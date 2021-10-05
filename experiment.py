from __future__ import print_function
from os import environ, path
import argparse
import torch
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import torch.optim as optim
from six import moves
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import random


from stn_baseline import BaseNet
from stn_coorconv import CoordNet
from gconv import GconvNet
from vit_pytorch import ViT

import warnings

warnings.filterwarnings("ignore")


def seed_everything(seed):
    """Use the same seed for python and torch and use deterministic
    algorithms to improve reproducibility.

    """
    random.seed(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="MNIST Experiment with STN, CoordConv STN, Group Equivariant Convnet and Vision Transformer"
)
parser.add_argument(
    "--network",
    "-n",
    type=str,
    default="stnconv",
    choices=["stnconv", "stncoord", "gconv", "vit"],
    help="specify the type of network to use for training the model (default stnconv)",
)
parser.add_argument(
    "--batch-size-train",
    type=int,
    default=64,
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--batch-size-test",
    type=int,
    default=64,
    help="input batch size for testing (default: 64)",
)
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=30,
    help="number of epochs to train (default: 30)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", "-s", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "--optimizer",
    "-o",
    type=str,
    default="sgd",
    choices=["sgd", "adam"],
    help="optimizer to use (default sgd)",
)
parser.add_argument(
    "--learning-rate",
    "-l",
    type=float,
    default="0.01",
    help="initial learning rate [default:0.01]",
)
parser.add_argument(
    "--momentum",
    "-m",
    type=float,
    default="0.5",
    help="initial learning rate [default:  0.5]",
)
parser.add_argument(
    "--log-interval",
    "-i",
    type=int,
    default=10,
    help="how many batches to wait before logging training status",
)

args = parser.parse_args()

# Check if cuda is available on the system
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common seed to imporve reproducibility
seed_everything(args.seed, args.cuda)

# Set the number of workers and pin memory if cuda is available
kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

writer = SummaryWriter(log_dir="tensorboard_logs/" + args.network)

# Setup urllib requests to fetch the MNIST data
opener = moves.urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
moves.urllib.request.install_opener(opener)

# Setup the loader for the training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root=".",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size_train,
    shuffle=True,
    **kwargs
)

# Setup the loader for the test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root=".",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size_test,
    shuffle=True,
    **kwargs
)

# Select the type of network to use for training the model
if args.network == "stnconv":
    model = BaseNet().to(device)
elif args.network == "stncoord":
    model = CoordNet(use_cuda=args.cuda).to(device)
elif args.network == "gconv":
    model = GconvNet().to(device)
else:
    model = ViT(
        image_size=28,
        patch_size=4,
        num_classes=10,
        channels=1,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
    )

if args.cuda:
    model.cuda()

# Setup the optimizer
if args.optimizer == "sgd":
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
elif args.optimizer == "adam" or args.network == "vit":
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


def train(epoch):
    """The train function iterates over batches of the training set,
    performs a forward and backward pass and records the progress
    of the trained model in terms of negative likelihood log loss
    and accuracy.

    """
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if args.network == "vit":
            output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Log train/loss to TensorBoard at every iteration
            n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
            writer.add_scalar("train/loss", loss.data, n_iter)
            print(
                "\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data,
                ),
                end="",
            )
    # Log model parameters to TensorBoard at every epoch
    for name, param in model.named_parameters():
        layer, attr = path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(
            "{}/{}".format(layer, attr), param.clone().cpu().data.numpy(), n_iter
        )

    print(
        "\nTrain Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(epoch):
    """The test function iterates over batches of the test data
    and evaluates the performance of the trained model in terms
    of negative likelihood log loss and accuracy.

    """
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if args.network == "vit":
                output = F.log_softmax(output, dim=1)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * correct / len(test_loader.dataset)
        # Log test/loss and test/accuracy to TensorBoard at every epoch
        n_iter = epoch * len(train_loader)
        writer.add_scalar("test/loss", test_loss, n_iter)
        writer.add_scalar("test/accuracy", test_accuracy, n_iter)
        print(
            "\nTest Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                epoch,
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


# Iterate over epochs to train and test the model
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

# Close TensorBoardX summary writer
writer.close()
