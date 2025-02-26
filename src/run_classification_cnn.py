import os, argparse
import torch, torchvision
import classification_model
from torchvision import datasets, transforms
from classification_model import ClassificationModel
from classification_cnn import evaluate

parser = argparse.ArgumentParser()

# Data settings
parser.add_argument('--n_batch',
    type=int, required=True, help='Number of samples per batch')
parser.add_argument('--dataset',
    type=str, required=True, help='Dataset for training: cifar10, mnist')

# Network settings
parser.add_argument('--encoder_type',
    type=str, required=True, help='Encoder type to build: vggnet11, resnet18')

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
    transforms_test = transforms_test = torchvision.transforms.Compose([
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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
		batch_size=args.n_batch,
		shuffle=False,
		drop_last=False,
		num_workers=2
	)

    '''
    Set up model
    '''
    # TODO: Instantiate network
    # input_channels = 3 if args.dataset == 'cifar10' else 1
    input_channels = dataset_test[0][0].shape[0]
    model = ClassificationModel(args.encoder_type, input_channels=input_channels, device=args.device)

    '''
    Restore weights and evaluate network
    '''
    # TODO: Load network from checkpoint
    checkpoint = model.restore_model(args.checkpoint_path)

    # TODO: Set network to evaluation mode
    
    model.eval()

    # TODO: Evaluate network on testing set
    
    evaluate(model, dataloader_test, dataset_test.classes, args.output_path, args.device)
