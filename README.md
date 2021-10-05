## MNIST Experiment

This repo contains a collection of computer vision experiments using different SOTA algorithms trained and evaluated on the MNIST dataset.
The experiment can train four different models:
1. Spatial Tranformer Networks
2. Spatial Transformer Networks with CoordConv
3. Group Equivariant Convolutional Networks
4. Visual Transformers

More detailed documentation is provided in [From_Idea_to_Implementation.pdf](https://github.com/gdimopoulos/mnist_experiment/blob/main/From_Idea_to_Implementation.pdf)

# Instructions
The main program that runs the experiments is `experiment.py`.
You can run it with `-h` to show a summary of the usage and a description of the different arguments that can be used:
```
python experiment.py -h

usage: experiment.py [-h] [--network {stnconv,stncoord,gconv,vit}] [--batch-size-train BATCH_SIZE_TRAIN] [--batch-size-test BATCH_SIZE_TEST] [--epochs EPOCHS] [--no-cuda] [--seed SEED] [--optimizer {sgd,adam}] [--learning-rate LEARNING_RATE] [--momentum MOMENTUM] [--log-interval LOG_INTERVAL]

MNIST Experiment with STN, CoordConv STN, Group Equivariant Convnet and Vision Transformer

optional arguments:
  -h, --help            show this help message and exit
  --network {stnconv,stncoord,gconv,vit}, -n {stnconv,stncoord,gconv,vit}
                        specify the type of network to use for training the model (default stnconv)
  --batch-size-train BATCH_SIZE_TRAIN
                        input batch size for training (default: 64)
  --batch-size-test BATCH_SIZE_TEST
                        input batch size for testing (default: 64)
  --epochs EPOCHS, -e EPOCHS
                        number of epochs to train (default: 30)
  --no-cuda             disables CUDA training
  --seed SEED, -s SEED  random seed (default: 1)
  --optimizer {sgd,adam}, -o {sgd,adam}
                        optimizer to use (default sgd)
  --learning-rate LEARNING_RATE, -l LEARNING_RATE
                        initial learning rate [default:0.01]
  --momentum MOMENTUM, -m MOMENTUM
                        initial learning rate [default: 0.5]
  --log-interval LOG_INTERVAL, -i LOG_INTERVAL
                        how many batches to wait before logging training status
``` 

Install the dependencies:
```
pip install -r requirements.txt
```

Usage example:
```
python experiment -n stnconv -e 10
```

The metrics of each experiment are tracked and stored with tesnorboardX under the `tensorboard_logs` directory.
You can load current or previous results by pointing tensorboard to the appropriate directory, for example:
```
tensorboard --logdir tensorboard_logs/vit
``` 