import os, argparse
import torch, torchvision
import classification_model
from torchvision import datasets, transforms
from classification_model import ClassificationModel
from classification_cnn import train

parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

# Network settings
parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type to build: vggnet11, resnet18')

# Training settings
parser.add_argument('--n_epoch',
    type=int, required=True, help='Number of passes through the full training dataset')
parser.add_argument('--learning_rate',
    type=float, required=True, help='Step size to update parameters')
parser.add_argument('--learning_rate_decay',
    type=float, required=True, help='Scaling factor to decrease learning rate at the end of each decay period')
parser.add_argument('--learning_rate_period',
    type=float, required=True, help='Number of epochs before reducing/decaying learning rat')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path directory to save checkpoints and Tensorboard summaries')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()


if __name__ == '__main__':

    # Create transformations to apply to data during training
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    transforms_train = torchvision.transforms.Compose([
        # TODO: Include random brightness, contrast, saturation, flip
        # and other augmentations of your choice
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # torchvision.transforms.RandomRotation(10),
        torchvision.transforms.ToTensor()
        
    ])

    # TODO: Construct training dataset based on args.dataset variable
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(
		root='./data',
		train=True,
		download=True,
		transform=transforms_train
		)
    elif args.dataset == 'mnist':
       dataset_train = datasets.MNIST(
		root='./data',
		train=True,
		download=True,
		transform=transforms_train
		)
    else:
        dataset_train = None

    # TODO: Setup a dataloader (iterator) to fetch from the training set using
    # torch.utils.data.DataLoader and set shuffle=True, drop_last=True, num_workers=2
    dataloader_train = torch.utils.data.DataLoader(
		dataset_train,
		batch_size=args.n_batch,
		shuffle=True,
		drop_last=True,
		num_workers=2
	)

    # TODO: Define the possible classes in depending on args.dataset variable
    class_names = dataset_train.classes

    # TODO: Get number of classes in dataset
    n_class = len(class_names)

    '''
    Set up model and optimizer
    '''
    # TODO: Compute number of input features depending on args.dataset variable
    input_shape = dataset_train[0][0].shape
    n_input_feature = input_shape[0] * input_shape[1] * input_shape[2]

    # TODO: Instantiate network
    model = ClassificationModel(args.encoder_type)

    # TODO: Setup learning rate SGD optimizer and step function scheduler
    # https://pytorch.org/docs/stable/optim.html?#torch.optim.SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    '''
    Train network and store weights
    '''
    # TODO: Set network to training mode
    
    model.train()

    # TODO: Train network
    
    train(model, dataloader_train, args.n_epoch, optimizer, args.learning_rate_decay, args.learning_rate_period, args.checkpoint_path, args.device)
