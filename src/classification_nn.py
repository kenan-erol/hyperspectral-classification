import torch, torchvision
import numpy as np
import log_utils


def train(net,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period,
          device):
    '''
    Trains the network using a learning rate scheduler

    Arg(s):
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        n_epoch : int
            number of epochs to train
        optimizer : torch.optim
            https://pytorch.org/docs/stable/optim.html
            optimizer to use for updating weights
        learning_rate_decay : float
            rate of learning rate decay
        learning_rate_decay_period : int
            period to reduce learning rate based on decay e.g. every 2 epoch
        device : str
            device to run on
    Returns:
        torch.nn.Module : trained network
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    # TODO: Move model to device using 'to(...)' function
    net = net.to(device)

    # TODO: Define cross entropy loss
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        # TODO: Decrease learning rate when learning rate decay period is met
        # e.g. decrease learning rate by a factor of decay rate every 2 epoch
        if epoch and epoch % learning_rate_decay_period == 0:

            # pass
            for param_group in optimizer.param_groups:
                param_group['lr'] *= learning_rate_decay

        for batch, (images, labels) in enumerate(dataloader):

            # TODO: Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # TODO: Vectorize images from (N, C, H, W) to (N, d)
            images = images.view(images.shape[0], -1)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Clear gradients so we don't accumlate them from previous batches
            optimizer.zero_grad()

            # TODO: Compute loss function
            loss = loss_func(outputs, labels)

            # TODO: Update parameters by backpropagation
            loss.backward()
            optimizer.step()

            # TODO: Accumulate total loss for the epoch
            total_loss +=loss.item()

        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch={}/{}  Loss: {:.3f}'.format(epoch + 1, n_epoch, mean_loss))

    return net

def evaluate(net, dataloader, class_names, output_path, device):
    '''
    Evaluates the network on a dataset

    Arg(s):
        net : torch.nn.Module
            neural network
        dataloader : torch.utils.data.DataLoader
            # https://pytorch.org/docs/stable/data.html
            dataloader for training data
        class_names : list[str]
            list of class names to be used in plot
        output_path: str
            path to save output visualization
        device : str
            device to run on
    '''

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    # TODO: Move model to device
    net = net.to(device)

    n_correct = 0
    n_sample = 0

    # Make sure we do not backpropagate
    with torch.no_grad():

        for (images, labels) in dataloader:

            # TODO: Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # TODO: Vectorize images from (N, H, W, C) to (N, d)
            C, H, W = images.shape[1:]
            images = images.view(images.shape[0], -1)

            # TODO: Forward through the network
            outputs = net(images)

            # TODO: Take the argmax over the outputs
            outputs = torch.argmax(outputs, dim=1)

            # Accumulate number of samples
            n_sample += labels.size(0)

            # TODO: Check if our prediction is correct
            n_correct += (outputs == labels).sum().item()

    # TODO: Compute mean accuracy
    mean_accuracy = 100.0 *  (n_correct / n_sample)

    print('Mean accuracy over {} images: {:.3f}%'.format(n_sample, mean_accuracy))

    # TODO: Convert the last batch of images back to original shape
    images = images.view(-1, C, H, W)

    # TODO: Move images back to cpu and to numpy array
    images = images.cpu().numpy()

    # TODO: torch.Tensor operate in (N x C x H x W), convert it to (N x H x W x C)
    images = images.transpose(0, 2, 3, 1) # [permute?]

    # TODO: Move the last batch of labels to cpu and convert them to numpy and
    # map them to their corresponding class labels
    labels = np.array([class_names[label] for label in labels.cpu().numpy()])

    # TODO: Move the last batch of outputs to cpu, convert them to numpy and
    # map them to their corresponding class labels
    outputs = np.array([class_names[output] for output in outputs.cpu().numpy()])

    # Convert images, outputs and labels to a lists of lists
    grid_size = 5

    images_display = []
    subplot_titles = []

    for row_idx in range(grid_size):
        # TODO: Get start and end indices of a row
        idx_start = row_idx * grid_size
        idx_end = idx_start + grid_size

        # TODO: Append images from start to end to image display array
        images_display.append(images[idx_start:idx_end])

        # TODO: Append text of 'output={}\nlabel={}' substituted with output and label to subplot titles
        for i in range(idx_start, idx_end):
            subplot_titles.append('output={}\nlabel={}'.format(outputs[i], labels[i]))


    # TODO: Plot images with class names and corresponding groundtruth label in a 5 by 5 grid
    log_utils.plot_images(images_display, grid_size, grid_size, subplot_titles,  output_path)
