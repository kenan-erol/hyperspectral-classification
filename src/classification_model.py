import torch
import networks
from matplotlib import pyplot as plt
import math
import log_utils

class ClassificationModel(object):
    '''
    Classification model class that supports VGG11 and ResNet18 encoders

    Arg(s):
        encoder_type : str
            encoder options to build: vggnet11, resnet18, etc.
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 encoder_type,
                 input_channels=3,
                 num_classes=1000,
                 device=torch.device('cuda')):

        self.device = device

        # TODO: Instantiate VGG11 and ResNet18 encoders and decoders based on
        # https://arxiv.org/pdf/1409.1556.pdf
        # https://arxiv.org/pdf/1512.03385.pdf
        # Decoder should use
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

        if encoder_type == 'vggnet11':
            self.encoder = networks.VGGNet11Encoder(input_channels=input_channels, n_filters=[64, 128, 256, 512, 512])
            self.decoder = torch.nn.Sequential(
                                               torch.nn.Flatten(), torch.nn.Linear(512, 4096), torch.nn.ReLU(inplace=True), torch.nn.Linear(4096, 4096), torch.nn.ReLU(inplace=True), torch.nn.Linear(4096, num_classes))

        elif encoder_type == 'resnet18':
            self.encoder = networks.ResNet18Encoder(input_channels=input_channels, n_filters=[64, 128, 256, 512, 512], use_batch_norm=True) # this needs to change for 3d
            self.decoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(), torch.nn.Linear(512, num_classes)) # paper says 1000 fc but the last layer has 512 so
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        # TODO: Move encoder and decoder to device
        
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def transform_input(self, images):
        '''
        Transforms input based on model arguments and settings

        Arg(s):
            images : torch.Tensor[float32]
                N x C x H x W images
        Returns:
            torch.Tensor[float32] : transformed N x C x H x W images
        '''

        # TODO: Perform normalization based on
        # https://arxiv.org/pdf/1409.1556.pdf
        # https://arxiv.org/pdf/1512.03385.pdf

        if self.encoder_type == 'vggnet11':
            # pass # paper says no normalisation?
            images = images/255.0
        elif self.encoder_type == 'resnet18':
            # pass batch normalization
            # torch.nn.transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225])(images)
            pass

        return images

    def forward(self, image):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
        Returns:
            torch.Tensor[float32] : N x K predicted class confidences
        '''

        # TODO: Implement forward function
        
        encoded = self.encoder(image)[0]
        output = self.decoder(encoded)

        return output

    def compute_loss(self, output, label):
        '''
        Compute cross entropy loss

        Arg(s):
            output : torch.Tensor[float32]
                N x K predicted class confidences
            label : torch.Tensor[int]
                ground truth class labels
        Returns:
            float : loss averaged over the batch
            dict[str, float] : dictionary of loss related tensors
        '''

        # TODO: Compute cross entropy loss
        loss = torch.nn.CrossEntropyLoss()(output, label)

        loss_info = {
            'loss' : loss
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        
        # params = list(self.encoder.parameters())
        # params.append(self.decoder.parameters())
        
        encodeds = list(self.encoder.parameters())
        decodeds = list(self.decoder.parameters())
        
        params = encodeds + decodeds

        return params

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device

        # TODO: Move encoder and decoder to device
        
        self.encoder.to(device)
        self.decoder.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        # TODO: Wrap encoder and decoder in
        # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
        
        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                lists of paths to model weights
            optimizer : torch.optim or None
                current optimizer
        Returns:
            int : training step
            torch.optim : restored optimizer or None if no optimizer is passed in
        '''

        # TODO: Restore the weights from checkpoint
        # Encoder and decoder are keyed using 'encoder_state_dict' and 'decoder_state_dict'
        # If optimizer is given, then save its parameters using key 'optimizer_state_dict'
        
        checkpoint = torch.load(restore_path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['step'], optimizer

    def save_model(self, checkpoint_path, step, optimizer=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                list path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        # TODO: Save the weights into checkpoint
        # Encoder and decoder are keyed using 'encoder_state_dict' and 'decoder_state_dict'
        # If optimizer is given, then save its parameters using key 'optimizer_state_dict'
        
        checkpoint = dict()
        
        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['decoder_state_dict'] = self.decoder.state_dict()
        checkpoint['step'] = step
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        
        

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image,
                    output,
                    ground_truth,
                    scalars={},
                    n_image_per_summary=16):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image : torch.Tensor[float32] 640 x 480
                image at time step
            output : torch.Tensor[float32]
                N
            label : torch.Tensor[float32]
                ground truth force measurements or ground truth bounding box and force measurements
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display
        '''

        with torch.no_grad():

            image_summary = image[0:n_image_per_summary, ...]

            # TODO: Move image_summary to CPU using cpu()
            image_summary = image_summary.cpu()

            # TODO: Convert image_summary to numpy using numpy() and swap dimensions from NCHW to NHWC
            n_batch, n_channel, n_height, n_width = image_summary.shape
            image_summary = image_summary.numpy().transpose((0, 2, 3, 1))

            # TODO: Create plot figure of size n x n using log_utils
            n = math.ceil(n_image_per_summary ** 0.5)
            images_display = []
            subplot_titles = []
            
            output = output[0:n_image_per_summary].cpu().detach().numpy()
            
            ground_truth = ground_truth[0:n_image_per_summary].cpu().detach().numpy()
            
            for i in range(n):
                start_idx = i * n
                end_idx = start_idx + n
                
                images_display.append(image_summary[start_idx:end_idx])
                
                subplot_titles.append([f'pred={output[i]}\nlabel={ground_truth[i]}' for i in range(start_idx, end_idx)])
                
            fig = log_utils.plot_images(images_display, n, n, subplot_titles)

            # TODO: Log image summary to Tensorboard with <tag>_image as its summary tag name using
            # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure
            
            summary_writer.add_figure(f'{tag}_image', fig, step)

            plt.tight_layout()

            plt.cla()
            plt.clf()
            plt.close()

            # TODO: Log scalars to Tensorboard with <tag>_<name> as its summary tag name using
            # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar
            for (name, value) in scalars.items():
                summary_writer.add_scalar(f'{tag}_{name}', value, step)
