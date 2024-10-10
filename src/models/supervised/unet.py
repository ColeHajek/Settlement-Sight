"""
This code is adapted from the U-Net paper:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
"""

import torch
import torch.nn as nn
from torch.nn.functional import pad


class DoubleConvHelper(nn.Module):
    """
    A helper module that performs two consecutive convolutions followed by BatchNorm and ReLU activations.

    This module implements a sequence of:
        - Convolutional Layer
        - Batch Normalization
        - ReLU Activation
        - Another Convolutional Layer
        - Batch Normalization
        - ReLU Activation

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int, optional
        Number of channels to use in the intermediate convolution layer. If not provided, defaults to `out_channels // 2`.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConvHelper, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels // 2

        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution block
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    """
    Encoder block for downsampling and feature extraction in U-Net.

    This block consists of a MaxPooling layer to reduce spatial dimensions followed by a DoubleConvHelper
    for feature extraction.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConvHelper(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    """
    Decoder block for upsampling and feature combination in U-Net.

    This block consists of a ConvTranspose2d layer for upsampling followed by a DoubleConvHelper
    to refine the features by combining skip connections from the encoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConvHelper(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass for the decoder block.

        Parameters
        ----------
        x1 : torch.Tensor
            The input tensor from the previous layer, shape (batch, in_channels, H, W).
        x2 : torch.Tensor
            The skip connection tensor from the encoder, shape (batch, in_channels//2, 2H, 2W).

        Returns
        -------
        torch.Tensor
            Output tensor after upsampling and merging with the skip connection.
        """
        # Upsample x1
        x1 = self.up(x1)

        # Calculate padding to match the dimensions of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate x1 and x2 along the channel dimension
        x = torch.cat([x2, x1], dim=1)

        # Pass through the double convolution block
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net model implementation.

    U-Net is a type of Convolutional Neural Network (CNN) used primarily for segmentation tasks.
    The U-Net architecture consists of a symmetric encoder-decoder structure where the encoder
    progressively down-samples the input image and the decoder progressively up-samples the features
    to predict segmentation masks. Skip connections are used to concatenate feature maps from the encoder
    to the decoder, allowing for precise localization.

    Parameters
    ----------
    in_channels : int
        Number of input channels of the image (e.g., 3 for RGB images).
    out_channels : int
        Number of output channels (e.g., number of classes for segmentation).
    n_encoders : int, optional
        Number of encoder blocks, by default 2.
    embedding_size : int, optional
        Number of feature maps in the first encoder block, by default 64.
    scale_factor : int, optional
        Scaling factor for the final output, used to downscale the segmentation result, by default 50.
    """

    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        super(UNet, self).__init__()

        # Initial double convolution to project from in_channels to embedding_size
        self.init_double_conv = DoubleConvHelper(in_channels, embedding_size)

        # Encoder blocks: progressively downsample and increase feature channels
        self.encoders = nn.ModuleList()
        in_feats = embedding_size
        for _ in range(n_encoders - 1):
            self.encoders.append(Encoder(in_feats, in_feats * 2))
            in_feats *= 2  # Double the number of features after each encoder block

        # Decoder blocks: progressively upsample and combine features with skip connections
        self.decoders = nn.ModuleList()
        for _ in range(n_encoders - 1):
            self.decoders.append(Decoder(in_feats, in_feats // 2))
            in_feats //= 2  # Halve the number of features after each decoder block

        # Final 1x1 convolution to project back to out_channels (number of classes)
        self.final_conv = nn.Conv2d(embedding_size, out_channels, kernel_size=1)

        # Downscale the final output using MaxPool2D (this step is optional and depends on application)
        self.downscale = nn.MaxPool2d(kernel_size=scale_factor)

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Segmented output tensor.
        """
        # Initial projection using the first double convolution
        x = self.init_double_conv(x)

        # Encoder path: pass through each encoder and save feature maps for skip connections
        residuals = []
        for encoder in self.encoders:
            residuals.append(x)
            x = encoder(x)

        # Decoder path: use the saved feature maps from the encoder as skip connections
        for decoder in self.decoders:
            skip_connection = residuals.pop()
            x = decoder(x, skip_connection)

        # Final 1x1 convolution to get the required number of output channels
        x = self.final_conv(x)

        # Optionally, downscale the output if needed
        x = self.downscale(x)
        return x
