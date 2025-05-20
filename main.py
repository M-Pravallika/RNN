from __future__ import unicode_literals, print_function, division
from io import open
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import wandb
import csv
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

START_token = 0
END_token = 1

class Language:
    def __init__(self, name):
        self.name = name
        self.charToKey = {}
        self.keyToChar = {0: "START", 1: "END"}
        self.totalTokens = 2

    # Add all characters in a word to Language dictionary
    def addWordToLanguage(self, word):
        for character in word:
            order = ord(character)
            if order not in self.charToKey:
                self.charToKey[order] = self.totalTokens
                self.keyToChar[self.totalTokens] = order
                self.totalTokens += 1

    def convertWordToTensor(self, word, device):
        indexes = [self.charToKey[ord(char)] for char in word]
        indexes.append(END_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

class LangDataLoader(nn.Module):
    def __init__(self, target_language, config, data_folder, data_split):
        self.config = config
        self.target_language = target_language
        self.data_folder = data_folder
        self.data_split = data_split

    def _read_language_data(self):
        file_path = f"{self.data_folder}/{self.target_language}/{self.target_language}_{self.data_split}.csv"
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            language_pairs = [[row[0], row[1]] for row in csv_reader]

        input_language = Language('eng')
        output_language = Language(self.target_language)

        return input_language, output_language, language_pairs
    
    def _filter_language_pairs(self, language_pairs, max_length):
        return [pair for pair in language_pairs if len(pair[0]) <= max_length and len(pair[1]) <= max_length]

    def _prepare_data(self):
        input_language, output_language, language_pairs = self._read_language_data()

        # Filter long pairs to ensure batch processing can handle it
        language_pairs = self._filter_language_pairs(language_pairs, self.config["max_length"])

        # Update word vocabularies
        for pair in language_pairs:
            input_language.add_word_to_language(pair[0])
            output_language.add_word_to_language(pair[1])

        print(f"Total words in {input_language.name}: {input_language.total_tokens}\nTotal words in {output_language.name}: {output_language.total_tokens}")

        return input_language, output_language, language_pairs

    def get_data_loader(self):
        input_language, output_language, language_pairs = self._prepare_data()
        max_length = self.config["max_length"]
        device = self.config["device"]
        num_pairs = len(language_pairs)

        # Initialize input and target data arrays
        input_data = np.zeros((num_pairs, max_length), dtype=np.int32)
        target_data = np.zeros((num_pairs, max_length), dtype=np.int32)

        # Populate input_data and target_data with word indexes
        for idx, (input_sentence, target_sentence) in enumerate(language_pairs):
            input_indexes = input_language.convert_word_to_indexes(input_sentence) + [END_token]
            target_indexes = output_language.convert_word_to_indexes(target_sentence) + [END_token]

            input_data[idx, :len(input_indexes)] = input_indexes
            target_data[idx, :len(target_indexes)] = target_indexes

        # Create a TensorDataset and DataLoader for batching
        dataset = TensorDataset(torch.LongTensor(input_data).to(device), torch.LongTensor(target_data).to(device))
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.config["batchSize"])
        return input_language, output_language, data_loader
    
class Encoder(nn.Module):
    def __init__(self, input_size, config):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config["hidden_layer_size"]
        self.embedding_size = config["embedding_size"]
        self.cell = getattr(nn, config["cell_type"])(self.embedding_size, self.hidden_size, batch_first=True, num_layers=config["num_layers"])
        self.dropout_layer = nn.Dropout(config["dropout"])
        self.embedding_layer = nn.Embedding(self.input_size, self.embedding_size)
        self.config = config

    def forward(self, input_tensor):
        embedded_input = self.dropout_layer(self.embedding_layer(input_tensor))
        output, hidden_state = self.cell(embedded_input)
        return output, hidden_state


class AttentionDecoder(nn.Module):
    def __init__(self, config, output_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.embedding_size = config["embedding_size"]
        self.dropout_layer = nn.Dropout(config["dropout"])
        self.cell = getattr(nn, config["cell_type"])(2 * self.hidden_size, self.hidden_size, batch_first=True, num_layers=config["num_layers"])
        self.device = config["device"]
        self.embedding_layer = nn.Embedding(output_size, self.hidden_size)
        self.attention_module = Attention(self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, output_size)
        self.config = config

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        decoder_input = torch.empty(encoder_outputs.size(0), 1, dtype=torch.long, device=self.device).fill_(START_token)
        decoder_outputs = []
        attention_weights_list = []

        for i in range(self.config["max_length"]):
            decoder_output, encoder_hidden, attention_weights = self.forward_step(
                decoder_input, encoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attention_weights_list.append(attention_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = F.log_softmax(torch.cat(decoder_outputs, dim=1), dim=-1)
        attention_weights = torch.cat(attention_weights_list, dim=1)

        decoder_hidden = encoder_hidden
        return decoder_outputs, decoder_hidden, attention_weights

    def forward_step(self, input_tensor, hidden_state, encoder_outputs):
        embedded_input = self.dropout_layer(self.embedding_layer(input_tensor))
        attention_hidden_state = hidden_state[-1].unsqueeze(0).permute(1, 0, 2)
        context_vector, attention_weights = self.attention_module(attention_hidden_state, encoder_outputs)
        cell_input = torch.cat((embedded_input, context_vector), dim=2)

        output, hidden_state = self.cell(cell_input, hidden_state)
        output = self.output_layer(output)

        return output, hidden_state, attention_weights


class Decoder(nn.Module):
    def __init__(self, config, output_size):
        super(Decoder, self).__init__()
        self.config = config
        self.hidden_size = config["hidden_layer_size"]
        self.embedding_size = config["embedding_size"]
        self.embedding_layer = nn.Embedding(output_size, self.embedding_size)
        self.cell = getattr(nn, config["cell_type"])(self.embedding_size, self.hidden_size, batch_first=True, num_layers=config["num_layers"])
        self.output_layer = nn.Linear(self.hidden_size, output_size)
        self.dropout_layer = nn.Dropout(config["dropout"])

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        decoder_input = torch.empty(encoder_outputs.size(0), 1, dtype=torch.long, device=self.config["device"]).fill_(START_token)
        decoder_outputs = []
        max_length = self.config["max_length"]

        for i in range(max_length):
            decoder_output, encoder_hidden = self.forward_step(decoder_input, encoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = F.log_softmax(torch.cat(decoder_outputs, dim=1), dim=-1)
        decoder_hidden = encoder_hidden
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input_tensor, hidden_state):
        embedded_input = self.dropout_layer(self.embedding_layer(input_tensor))
        cell_input = getattr(F, self.config["activation_fn"])(embedded_input)

        output, hidden_state = self.cell(cell_input, hidden_state)
        output = self.output_layer(output)
        return output, hidden_state


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight_matrix = nn.Linear(hidden_size, hidden_size)
        self.query_matrix = nn.Linear(hidden_size, hidden_size)
        self.attention_weight = nn.Linear(hidden_size, 1)

    def forward(self, previous_state, encoder_outputs):
        energy = self.attention_weight(torch.tanh(self.weight_matrix(previous_state) + self.query_matrix(encoder_outputs)))
        energy = energy.squeeze(2).unsqueeze(1)

        attention_scores = F.softmax(energy, dim=-1)

        context_vector = torch.bmm(attention_scores, encoder_outputs)
        return context_vector, attention_scores

# Method to parse the command line arguments
def parseArguments():
    p = argparse.ArgumentParser()
    p.add_argument('-wp', '--wandb_project', help="WandB project name", type=str, default='da6401-rnn')
    p.add_argument('-we', '--wandb_entity', help="WandB entity name", type=str, default='ns25z065')
    p.add_argument('-m', '--mode', help="Apply attention or not, choices: [Vanilla, Attention]", type=str, default='Attention')
    p.add_argument('-e', '--epochs', help="Max number of epochs to run", type=int, default=5)
    p.add_argument('-c', '--cell', help="Type of cell choices: [GRU, RNN, LSTM]", type=str, default='GRU')
    p.add_argument('-n', '--layers', help="Number of layers: [1, 2, 3]", type=int, default=3)
    p.add_argument('-hs', '--hidden_size', help="Hidden cell size", type=int, default=256)
    p.add_argument('-es', '--embedding_size', help="Embedding size", type=int, default=512)
    p.add_argument('-b', '--batchSize', help="Batch size", type=int, default=32)
    p.add_argument('-o', '--optimizer', help="Optimizer [RMSprop, Adam, NAdam]", type=str, default='RMSprop')
    p.add_argument('-lr', '--eta', help="Learning rate (eta)", type=float, default=0.001)
    p.add_argument('-a', '--activation', help="Activation function choices: [relu, gelu, tanh, selu, mish, leaky_relu]", type=str, default='tanh')
    p.add_argument('-do', '--dropout', help="Dropout probability p, if p = 0 no dropout", type=float, default=0)
    p.add_argument('-f', '--folder', help="Directory containing train and val folders of image dataset", type=str, default='data')
    p.add_argument('-l', '--lang', help="Language dataset [hin, tel, tam, etc.,]", type=str, default='hin')
    return p.parse_args()

def train(train_dataloader, encoder, decoder, config, inputLang, outputLang, folder):
    totalLoss = 0

    # Initialise optimizers and loss function
    encoderOptimizer = getattr(optim, config["optim"])(encoder.parameters(), lr=config["eta"])
    decoderOptimizer = getattr(optim, config["optim"])(decoder.parameters(), lr=config["eta"])
    criterion = nn.NLLLoss()

    for epoch in range(1, config["num_epochs"] + 1):
        loss = trainEpoch(train_dataloader, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion)
        totalLoss += loss

        # Calculate validation accuracy with validation data
        validationAccuracy = evaluateData(encoder, decoder, inputLang, outputLang, outputLang.name, 'valid', folder, config["device"])
        print('(%d %d%%) Accuracy: %.4f' % (epoch, epoch / config["num_epochs"] * 100, validationAccuracy))
        wandb.log({"epoch": epoch, "train_loss": totalLoss, "val_accuracy": validationAccuracy})
        totalLoss = 0

# Method to train a data epoch
def trainEpoch(dataloader, encoder, decoder, encoderOptimizer, decoderOptimizer, criterion):
    totalLoss = 0

    #For every batch, do forward, backpropagate and adjust
    for data in dataloader:
        inputTensor, targetTensor = data
        encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()

        # Forward propagate through the network for the batch input
        encoderOutputs, encoderHidden = encoder.forward(inputTensor)
        decoderOutputs, _, _ = decoder.forward(encoderOutputs, encoderHidden, targetTensor)

        # Calculate loss
        loss = criterion(
            decoderOutputs.view(-1, decoderOutputs.size(-1)),
            targetTensor.view(-1)
        )

        # Backpropagate and adjust parameters
        loss.backward()
        encoderOptimizer.step()
        decoderOptimizer.step()

        totalLoss += loss.item()

    return totalLoss / len(dataloader)

# Method to evaluate training/ validation/ test data based on 'filename' argument
def evaluateData(encoder, decoder, inputLang, outputLang, lang, filename, folder, device, write=False):
    # Set encoder and decoder to evaluation mode
    encoder.eval()
    decoder.eval()
    path = folder + '/' + lang + '/' + lang + '_' + filename + '.csv'

    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        pairs = [[row[0], row[1]] for row in csv_reader]

    total = len(pairs)
    score = 0

    # Evaluate each word
    for i in range(total):
        output, _ = evaluate(encoder, decoder, pairs[i][0], inputLang, outputLang, device)
        if output == pairs[i][1]:
            score += 1
        
        if write == True:
            write_to_csv(pairs[i][0], pairs[i][1], output, "output.csv")


    # Set back to training for next epoch
    encoder.train()
    decoder.train()
    return score/total * 100

def write_to_csv(input_word, target_word, prediction, filename):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([input_word, target_word, prediction])

def evaluate(encoder, decoder, word, inputLang, outputLang, device):
    attentions = None
    with torch.no_grad():
        try:
            inputTensor = inputLang.convertWordToTensor(word, device)
            
            # forward propagation
            encoderOutputs, encoderHidden = encoder.forward(inputTensor)
            decoderOutputs, decoderHidden, attentions = decoder.forward(encoderOutputs, encoderHidden)
    
            _, topi = decoderOutputs.topk(1)
            decodedTokens = topi.squeeze()
    
            # Convert tokens to actual output language string
            decodedWord = []
            for token in decodedTokens:
                if token.item() == END_token:
                    break
                decodedWord.append(outputLang.keyToChar[token.item()])
    
            decodedWord = ''.join([chr(char) for char in decodedWord])
        except:
            decodedWord = ''
        
    return decodedWord, attentions

def train_test_model(args):
    print(args)
    config = {
        "architecture": args.mode,
        "num_epochs": args.epochs,
        "cell_type": args.cell,
        "num_layers": args.layers,
        "hidden_layer_size": args.hidden_size,
        "embedding_size": args.embedding_size,
        "activation_fn": args.activation,
        "dropout": args.dropout,
        "eta": args.eta,
        "optim": args.optimizer,
        "batchSize": args.batchSize
    }

    additional_config = {
        "max_length": 25
    }

    with wandb.init(config = config, project=args.wandb_project):
        config = wandb.config

        # Select device dynamically
        gpu_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_built()
        device = "mps" if mps_available else "cuda" if gpu_available else "cpu"
        print("Running on device: " + device)
        additional_config["device"] = device

        config.update(additional_config)

        run_name_template = "{} / cellType {} / #layers {} / hiddenSize {} / embedSize {} / actFn {} / dropout {} / learnRate {} / optim {} / batchSize {}"
        run_name = run_name_template.format(config["architecture"], config["cell_type"],
                                            config["num_layers"], config["hidden_layer_size"],
                                            config["embedding_size"], config["activation_fn"],
                                            config["dropout"], config["eta"], config["optim"],
                                            config["batchSize"])

        print("Run name: ", run_name)
        wandb.run.name = run_name

        # Load the data
        langDataLoader = LangDataLoader(args.lang, config, args.folder, 'train')
        input_lang, output_lang, train_dataloader = langDataLoader.get()

        # Create encoder and decoder based on architecture
        encoder = Encoder(input_lang.totalTokens, config).to(device)
        if config["architecture"] == "Attention":
            decoder = AttentionDecoder(config, output_lang.totalTokens).to(device)
        else:
            decoder = Decoder(config, output_lang.totalTokens).to(device)

        # Train the model and check validation accuracy
        train(train_dataloader, encoder, decoder, config, input_lang, output_lang, args.folder)
        testAccuracy = evaluateData(encoder, decoder, input_lang, output_lang, output_lang.name, 'test', args.folder, device, True)
        print('Test Accuracy: %.4f' % (testAccuracy))
        wandb.log({"accuracy": testAccuracy})
        return encoder, decoder, input_lang, output_lang, device

if __name__ == "__main__":
    args = parseArguments()
    encoder, decoder, input_lang, output_lang, device = train_test_model(args)
    wandb.finish()