import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from classification_model import ClassificationModel
from classification_cnn import train, evaluate
from sam2 import SAM2

class HyperspectralDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load labels and file paths from the label file
        with open(label_file, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                self.samples.append((os.path.join(data_dir, image_path), int(label)))

        # Initialize SAM2 for mask generation
        self.sam2 = SAM2()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        # Load the hyperspectral image (e.g., using numpy or spectral library)
        image = np.load(image_path)  # Replace with actual hyperspectral image loader

        # Generate masks using SAM2
        masks = self.sam2.generate_masks(image) # or SAM2AutomaticMaskGenerator(sam2).generate(image)?

        # Sample 5-6 bounding boxes from the masks
        patches = []
        for mask in masks[:5]:
            x, y, w, h = mask  # Assume mask provides bounding box coordinates
            patch = image[y:y+h, x:x+w]
            patches.append(patch)

        # Apply transformations to each patch
        if self.transform:
            patches = [self.transform(patch) for patch in patches]

        return patches, label

    @staticmethod
    def collate_fn(batch):
        patches, labels = zip(*batch)
        patches = [torch.stack(patch_set) for patch_set in patches]
        labels = torch.tensor(labels)
        return patches, labels

if __name__ == '__main__':
    # Arguments (replace with argparse if needed)
    data_dir = './data/drop-4/' # this prob needs to be updated to include all the images
    label_file = './data/labels.txt' # still need this from simon
    checkpoint_path = './checkpoints/hyperspectral/'
    output_path = './outputs/hyperspectral/'
    n_batch = 50
    n_epoch = 200
    learning_rate = 0.05
    learning_rate_decay = 0.5
    learning_rate_period = 25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust for hyperspectral data
    ])

    # Dataset and DataLoader
    dataset = HyperspectralDataset(data_dir, label_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True, collate_fn=HyperspectralDataset.collate_fn)

    # Model
    model = ClassificationModel(encoder_type='resnet18', input_channels=1, device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    train(model, dataloader, n_epoch, optimizer, learning_rate_decay, learning_rate_period, checkpoint_path, device)

    # Evaluate
    evaluate(model, dataloader, dataset.classes, output_path, device)