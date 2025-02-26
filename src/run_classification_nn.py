import os, argparse
import torch, torchvision
import classification_model
from torchvision import datasets, transforms
from networks import NeuralNetwork
from classification_nn import evaluate

parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoint file')
parser.add_argument('--output_path',
    type=str, required=True, help='Path to save output')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()


if __name__ == '__main__':

    '''
    Set up dataloading
    '''
    # TODO: Create transformations to apply to data during testing
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
		torchvision.transforms.ToTensor()
	])

    # TODO: Construct testing dataset based on args.dataset variable
    if args.dataset == 'cifar10':
        dataset_test = datasets.CIFAR10(
		root='./data',
		train=False,
		download=True,
		transform=transforms_test
		)
    elif args.dataset == 'mnist':
       dataset_test = datasets.MNIST(
		root='./data',
		train=False,
		download=True,
		transform=transforms_test
		)
    else:
        dataset_test = None

    # TODO: Setup a dataloader (iterator) to fetch from the testing set using
    # torch.utils.data.DataLoader and set shuffle=False, drop_last=False, num_workers=2
    # Set batch_size to 25
    dataloader_test = torch.utils.data.DataLoader(
		dataset_test,
		batch_size=25,
		shuffle=False,
		drop_last=False,
		num_workers=2
	)

    '''
    Set up model
    '''
    # TODO: Compute number of input features based on args.dataset variable
    # n_input_feature = args.dataset ==
    input_shape = dataset_test[0][0].shape
    n_input_feature = input_shape[0] * input_shape[1] * input_shape[2]

    # TODO: Instantiate network
    net = NeuralNetwork(n_input_feature, len(dataset_test.classes))

    '''
    Restore weights and evaluate network
    '''
    # TODO: Load network from checkpoint
    checkpoint = net.load_state_dict(torch.load(args.checkpoint_path))

    # TODO: Set network to evaluation mode
    
    net.eval()

    # TODO: Evaluate network on testing set
    
    evaluate(net, dataloader_test, dataset_test.classes, args.output_path, args.device)
