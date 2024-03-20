"""
This code is adapted from the U-Net paper. See details in the paper:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. 
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad

class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Module that implements 
            - a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        
        Input:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            mid_channels (int): number of channels to use in the intermediate layer    
        """
        super(DoubleConvHelper,self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels //2 # Default to out_channels if mid_channels is not provided
        
        # The first convolution block
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # The second convolution block
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Pass input through the first convolution block
        x = self.relu1(self.bn1(self.conv1(x)))
        # Pass result through the second convolution block
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    """ Downscale using the maxpool then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        # Define the maxpool operation
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Instantiate the DoubleConvHelper using the provided in_channels and out_channels
        self.double_conv = DoubleConvHelper(in_channels, out_channels)

    def forward(self, x):
        # Apply the maxpool operation
        x = self.maxpool(x)
        # Pass the result through the DoubleConvHelper
        x = self.double_conv(x)
        return x

class Decoder(nn.Module):
    """ Upscale using ConvTranspose2d then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        # Double convolution
        self.double_conv = DoubleConvHelper(in_channels, out_channels)
    
    
    def forward(self, x1, x2):
        """ 
        1) x1 is passed through either the upsample or the convtranspose2d
        2) The difference between x1 and x2 is calculated to account for differences in padding
        3) x1 is padded (or not padded) accordingly
        4) x2 represents the skip connection
        5) Concatenate x1 and x2 together with torch.cat
        6) Pass the concatenated tensor through a doubleconvhelper
        7) Return output
        """
        # Replace x1 with the upsampled version of x1
        x1 = self.up(x1)
       
        # Input is Channel Height Width, step 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Add padding
        x1 = pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate x1 and x2
        x = torch.cat([x2,x1],dim=1)

        # Pass the concatenated tensor through a doubleconvhelper
        x = self.double_conv(x)

        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders. 
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.
        
        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is 
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.

        Input:
            in_channels: number of input channels of the image
            of shape (batch, in_channels, width, height)
            out_channels: number of output channels of prediction,
            prediction is shape (batch, out_channels, width//scale_factor, height//scale_factor)
            n_encoders: number of encoders to use in the network (implementing this parameter is
            optional, but it is a good idea to have it as a parameter in case you want to experiment,
            if you do not implement it, you can assume n_encoders=2)
            embedding_size: number of channels to use in the first layer
            scale_factor: number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        """
        super(UNet, self).__init__()
        self.init_double_conv = DoubleConvHelper(in_channels, embedding_size)
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_feats = embedding_size
        for _ in range(n_encoders - 1):
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoders.append(DoubleConvHelper(in_feats, in_feats * 2))
            in_feats *= 2  # Double the number of features after each block


        self.decoders = nn.ModuleList()
        for _ in range(n_encoders - 1):
            self.decoders.append(Decoder(in_feats, in_feats // 2))
            in_feats //= 2  # Halve the number of features after each block

        # Final convolution (no ReLU, and possibly a 1x1 conv to reduce to out_channels)
        self.final_conv = nn.Conv2d(embedding_size, out_channels, kernel_size=1)
        
        # Scaling down the final output using MaxPool2D (this is unusual in U-Nets generally, 
        # you might skip this if your input and output sizes should match)
        self.downscale = nn.MaxPool2d(kernel_size=scale_factor)

    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        # Initial double convolution
        x = self.init_double_conv(x)
        
        # Encoder path
        residuals = []  # To store features for the skip connections
        for pool, encoder in zip(self.pools, self.encoders):
            residuals.append(x)
            x = pool(x)
            x = encoder(x)

        # Decoder path
        for decoder in self.decoders:
            skip_connection = residuals.pop()  # Get the last feature map from the residuals
            x = decoder(x, skip_connection)  # Pass the current output and the skip connection to the decoder

        # Final convolution
        x = self.final_conv(x)
        
        # Scale down the output if required (depends on your specific application)
        x = self.downscale(x)
        
        return x
