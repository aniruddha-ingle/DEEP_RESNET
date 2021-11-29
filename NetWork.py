import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn.functional import relu

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        self.start_layer = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3, 3), 
                                stride=1, padding=1)
        
        ### YOUR CODE HERE
        block_fn = bottleneck_block
        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self,  num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.BN_RELU = nn.Sequential(
            nn.BatchNorm2d(self.num_features, self.eps, self.momentum),
            nn.ReLU()
        )
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        return self.BN_RELU(inputs)
        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        self.filters = filters
        self.projection_shortcut = projection_shortcut
        self.strides = strides
        self.first_num_filters = first_num_filters
        self.batch_norm_relu_block = batch_norm_relu_layer(self.filters, 1e-5, 0.997)
        
        if projection_shortcut is not None:
            if filters // 4 == first_num_filters:
                filters = self.filters // 4
            else:
                filters = self.filters // 2
                strides = strides * 2
    
        hidden_filters = self.filters // 4
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        self.BOTTLENECK = nn.Sequential(
            batch_norm_relu_layer(filters, 1e-5, 0.997),
            nn.Conv2d(in_channels = filters, out_channels = hidden_filters , kernel_size = 1, bias=True,
                                stride=strides, padding=0),
            batch_norm_relu_layer(hidden_filters, 1e-5, 0.997),
            nn.Conv2d(in_channels = hidden_filters, out_channels = hidden_filters, kernel_size = 3, bias=True, 
                                stride=self.strides, padding=1),
            batch_norm_relu_layer(hidden_filters, 1e-5, 0.997),
            nn.Conv2d(in_channels = hidden_filters, out_channels = self.filters, kernel_size = 1, bias=True, 
                                stride=self.strides, padding=0),
        )
        ### YOUR CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        outputs = self.BOTTLENECK(inputs)
        if self.projection_shortcut is not None:
            inputs = self.projection_shortcut(inputs)

        outputs += inputs
        
        return outputs
        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        
        filters_out = filters * 4
        
        ### END CODE HERE
        self.resnet_size = resnet_size
        self.blocks = nn.ModuleList()

        for i in range(resnet_size):
            projection_shortcut = None
            if first_num_filters == filters_out // 4 and i==0:
                projection_shortcut = nn.Conv2d(in_channels = filters_out//4, out_channels = filters_out, kernel_size = (1, 1), 
                                        stride=strides, padding=0)
            elif first_num_filters < filters_out and i==0:
                projection_shortcut = nn.Conv2d(in_channels = filters_out//2, out_channels = filters_out, kernel_size = (1, 1), 
                                        stride=strides, padding=0)                
            self.blocks.append(bottleneck_block(filters_out, projection_shortcut, 1, first_num_filters))
        
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        outputs = inputs
        for i in range(self.resnet_size):
            outputs = self.blocks[i](outputs)
        return outputs
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        ### END CODE HERE
        self.batchnorm_relu = batch_norm_relu_layer(filters, 1e-5, 0.997)
        self.num_classes = num_classes
        self.OUTPUT = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters, num_classes),
        )
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        if self.batchnorm_relu is not None:
          inputs = self.batchnorm_relu(inputs)

        outputs = self.OUTPUT(inputs)
        
        return outputs
        ### END CODE HERE