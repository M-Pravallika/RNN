# DA6401 Assignment 3

### Steps to run

The argument `-f` or `--folder` should be used to mention the folder inside root in which all the language folders are present.
Example: dataset for the below case

/dataset
 - hin

The argument `-l` or `--lang` should be used to mention the language to be trained on

```
python main.py --wandb_entity ns25z065 --wandb_project da6401-rnn --type Vanilla --noOfEpochs 5 --cellType GRU --noOfLayers 3 --hidden_size 256 --embedding_size 512 --batch_size 32 --optimizer RMSprop --eta 0.001 --activation leaky_relu --dropout 0 --folder dataset --lang hin
```

### Arguments supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | da6401-rnn | WandB project |
| `-we`, `--wandb_entity` | ns25z065 | WandB entity |
| `-m`,`--mode` | Attention | Apply attention or not: [Vanilla, Attention] |
| `-e`, `--epochs` | 5 | Max number of epochs to run |
| `-c`, `--cell` | GRU | Type of cell choices: [GRU, RNN, LSTM] |
| `-n`, `--layers` | 3 | Number of layers: [1, 2, 3] |
| `-hs`, `--hiddenSize` | 256 | Hidden cell size |
| `-es`, `--embeddingSize` | 512 | Embedding size |
| `-b`, `--batchSize` | 32 | Batch size |
| `-o`, `--optimizer` | RMSprop | Optimizer to be used [RMSprop, Adam, NAdam] |
| `-lr`, `--eta` | 0.001 | Learning rate |
| `-a`, `--activationFn` | tanh | Activation function type: [relu, gelu, tanh, selu, mish, leaky_relu] |
| `-do`, `--dropout` | 0 | Dropout probability p, for no dropout give p = 0 |
| `-f`, `--folder` | dataset | Directory containing train and val folders dataset |
| `-l`, `--lang` | hin | Language dataset to be trained on [hin, etc.,] |

### Steps to run for Attention based model

```--type Attention``` to train the RNN along with Attention mechanism.

```
python main.py --wandb_entity ns25z065 --wandb_project da6401-rnn --type Attention --noOfEpochs 5 --cellType GRU --noOfLayers 3 --hidden_size 256 --embedding_size 512 --batch_size 32 --optimizer RMSprop --eta 0.001 --activation tanh --dropout 0 --folder dataset --lang hin
```
