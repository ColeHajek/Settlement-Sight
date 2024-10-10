import torch
import torch.nn as nn
from typing import List

class Encoder(nn.Module):
    """
    Encoder block for Convolutional Neural Networks.

    This class represents a CNN block that projects from `in_channels` to a usually higher number
    of channels `out_channels`. It runs the image through `depth` layers of convolutions with 
    a specified `kernel_size`. Padding is used to keep the input and output dimensions the same.

    After the convolutional layers, a MaxPooling operation reduces the resolution of the input
    by a factor of `pool_size`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    depth : int
        Number of convolutional layers in the encoder block.
    kernel_size : int
        Size of the kernel for the convolutional layers.
    pool_size : int
        Size of the pooling layer kernel.
    """
    
    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size: int, pool_size: int):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize a list of convolutional layers
        conv_layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        ]

        # Append (depth - 1) additional convolutional layers with ReLU activations
        for _ in range(depth - 1):
            conv_layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.ReLU(inplace=True))
        
        # Group all convolutional layers into a sequential container
        self.convs = nn.Sequential(*conv_layers)
        
        # Initialize max pooling layer
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        """
        Forward pass through the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after passing through convolutions and pooling,
            of shape (batch_size, out_channels, height // pool_size, width // pool_size).
        """
        x = self.convs(x)
        x = self.pool(x)
        return x

class SegmentationCNN(nn.Module):
    """
    Convolutional Neural Network for Segmentation.

    This model takes an input image of shape (batch_size, in_channels, height, width) and outputs a
    segmented image of shape (batch_size, out_channels, height // prod(pool_sizes), width // prod(pool_sizes)),
    where `prod()` is the product of all the values in `pool_sizes`.

    The model uses multiple `Encoder` blocks to downsample the input and increase the feature channels,
    followed by a final `decoder` layer to project the features to the desired number of `out_channels`.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    out_channels : int
        Number of output channels (e.g., number of classes for segmentation).
    depth : int, optional
        Number of convolutional layers per encoder block, by default 2.
    embedding_size : int, optional
        Number of output channels for the first encoder block, by default 64.
    pool_sizes : List[int], optional
        List of pooling factors for each encoder block, by default [5, 5, 2].
    kernel_size : int, optional
        Size of the convolutional kernels, by default 3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        embedding_size: int = 64,
        pool_sizes: List[int] = [5, 5, 2],
        kernel_size: int = 3,
        **kwargs,
    ):
        super(SegmentationCNN, self).__init__()

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.embedding_size = embedding_size

        # Initialize the list of encoder blocks
        encoder_blocks = []

        # Create the first encoder with in_channels mapped to embedding_size
        encoder_blocks.append(
            Encoder(in_channels=in_channels, out_channels=embedding_size, depth=depth, kernel_size=kernel_size, pool_size=pool_sizes[0])
        )

        # Create subsequent encoders with progressively increasing number of output channels
        for pool_size in pool_sizes[1:]:
            encoder_blocks.append(
                Encoder(in_channels=embedding_size, out_channels=2 * embedding_size, depth=depth, kernel_size=kernel_size, pool_size=pool_size)
            )
            # Update embedding size for the next layer
            embedding_size *= 2

        # Convert the list of encoders to a ModuleList so they are registered as submodules
        self.encoders = nn.ModuleList(encoder_blocks)

        # Final decoder layer projects from the final embedding size back to the number of output channels
        self.decoder = nn.Conv2d(in_channels=embedding_size, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the entire network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height // prod(pool_sizes), width // prod(pool_sizes)).
        """
        # Pass input through each encoder block sequentially
        for encoder in self.encoders:
            x = encoder(x)
        
        # Final projection through the decoder layer
        y_pred = self.decoder(x)
        return y_pred