import torch
import net_utils
from net_utils import ResNetBlock, VGGNetBlock, Conv2d

class NeuralNetwork(torch.nn.Module):
    '''
    Neural network class of fully connected layers

    Arg(s):
        n_input_feature : int
            number of input features
        n_output : int
            number of output classes
    '''

    def __init__(self, n_input_feature, n_output):
        super(NeuralNetwork, self).__init__()

        # Create your 6-layer neural network using fully connected layers with ReLU activations
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
        # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

        # TODO: Instantiate 5 fully connected layers and choose number of neurons i.e. 512
        self.fc1 = torch.nn.Linear(n_input_feature, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 64)

        # TODO: Define output layer
        self.output = torch.nn.Linear(64, n_output)

    def forward(self, x):
        '''
        Forward pass through the neural network

        Arg(s):
            x : torch.Tensor[float32]
                tensor of N x d
        Returns:
            torch.Tensor[float32]
                tensor of n_output predicted class
        '''

        # TODO: Implement forward function
        output_fc1 = torch.nn.functional.relu(self.fully_connected_layer1(x))
        output_fc2 = torch.nn.functional.relu(self.fully_connected_layer2(output_fc1))
        output_fc3 = torch.nn.functional.relu(self.fully_connected_layer3(output_fc2))
        output_fc4 = torch.nn.functional.relu(self.fully_connected_layer4(output_fc3))
        output_fc5 = torch.nn.functional.relu(self.fully_connected_layer5(output_fc4))

        output = self.output(output_fc5)

        return output

class ResNet18Encoder(torch.nn.Module):
    '''
    ResNet18 encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(ResNet18Encoder, self).__init__()

        assert len(n_filters) == 5

        activation_func = net_utils.activation_func(activation_func)

        # TODO: Implement ResNet encoder using ResNetBlock from net_utils.py
        # Based on https://arxiv.org/pdf/1512.03385.pdf
        # net utils has a ResNetBlock class that you can use
        
        self.conv1 = Conv2d(input_channels, n_filters[0], kernel_size=7, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        
        self.max_pool = torch.nn.MaxPool2d(3, 2)
        
        self.conv2_x = torch.nn.Sequential(ResNetBlock(n_filters[0], n_filters[1], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm),
                                         ResNetBlock(n_filters[1], n_filters[1], stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                                        )
        
        self.conv3_x = torch.nn.Sequential(ResNetBlock(n_filters[1], n_filters[2], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm),
                                         ResNetBlock(n_filters[2], n_filters[2], stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
		)
        
        self.conv4_x = torch.nn.Sequential(ResNetBlock(n_filters[2], n_filters[3], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm),
                                         ResNetBlock(n_filters[3], n_filters[3], stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
		)
        
        self.conv5_x = torch.nn.Sequential(ResNetBlock(n_filters[3], n_filters[4], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm),
                                         ResNetBlock(n_filters[4], n_filters[4], stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
		)

    def forward(self, x):
        '''
        Forward input x through a ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            list[torch.Tensor[float32]] : list of intermediate feature maps used for skip connections
        '''

        layers = [x]

        # TODO: Implement forward function
        
        x = self.conv1(x)
        x = self.max_pool(x)
        
        layers.append(x)
        
        x = self.conv2_x(x)
        
        layers.append(x)
        
        x = self.conv3_x(x)
        
        layers.append(x)
        
        x = self.conv4_x(x)
        
        layers.append(x)
        
        x = self.conv5_x(x)
        
        layers.append(x)

        # Return latent and intermediate features
        return layers[-1], layers[1:-1]

class VGGNet11Encoder(torch.nn.Module):
    '''
    VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(VGGNet11Encoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        # TODO: Implement VGGNet encoder using VGGNetBlock from net_utils.py
        # Based on https://arxiv.org/pdf/1409.1556.pdf
        
        self.conv1 = VGGNetBlock(input_channels, n_filters[0], n_convolution=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        
        # self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = VGGNetBlock(n_filters[0], n_filters[1], n_convolution=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        
        # self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        
        self.conv3 = VGGNetBlock(n_filters[1], n_filters[2], n_convolution=2, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        
        # self.maxpool3 = torch.nn.MaxPool2d(2, 2)
        
        self.conv4 = VGGNetBlock(n_filters[2], n_filters[3], n_convolution=2, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        
        # self.maxpool4 = torch.nn.MaxPool2d(2, 2)
        
        use_max_pool = (input_channels != 1)
        
        self.conv5 = VGGNetBlock(n_filters[3], n_filters[4], n_convolution=2, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_max_pool=use_max_pool)
        
        # self.maxpool5 = torch.nn.MaxPool2d(2, 2)
        
        

    def forward(self, x):
        '''
        Forward input x through a VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            list[torch.Tensor[float32]] : list of intermediate feature maps used for skip connections
        '''

        layers = [x]

        # TODO: Implement forward function
        
        x = self.conv1(x)
        
        # x = self.maxpool1(x)
        
        layers.append(x)
        
        x = self.conv2(x)
        
        # x = self.maxpool2(x)
        
        layers.append(x)
        
        x = self.conv3(x)
        
        # x = self.maxpool3(x)
        
        layers.append(x)
        
        x = self.conv4(x)
        
        # x = self.maxpool4(x)
        
        layers.append(x)
        
        x = self.conv5(x)
        
        # x = self.maxpool5(x)
        
        layers.append(x)
        

        # Return latent and intermediate features
        return layers[-1], layers[1:-1]
