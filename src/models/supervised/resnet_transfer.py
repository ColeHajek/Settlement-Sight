from torchvision.models.segmentation import fcn_resnet101
import torch
from torch import nn

class FCNResnetTransfer(nn.Module):
    """
    A modified version of the FCN-ResNet101 model from the torchvision library for custom segmentation tasks.

    This class adapts the FCN-ResNet101 model by replacing its first and last convolutional layers to match
    custom input and output channels. Additionally, an average pooling layer is added at the end to resize
    the output based on a given scale factor.

    Parameters
    ----------
    input_channels : int
        Number of input channels in the input image (e.g., 3 for RGB or 57 for multispectral images).
    output_channels : int
        Number of output channels in the prediction (e.g., number of classes for segmentation).
    scale_factor : int, optional
        Scaling factor for the final output, used to downscale the segmentation result, by default 50.

    Note
    ----
    The original FCN-ResNet101 is designed for semantic segmentation tasks where the output is of the same
    resolution as the input. This class adds an additional pooling layer to downscale the final output.
    """

    def __init__(self, input_channels, output_channels, scale_factor=50, **kwargs):
        super(FCNResnetTransfer, self).__init__()

        # Load the FCN-ResNet101 model with pretrained weights
        fcn_model = fcn_resnet101(pretrained=True, pretrained_backbone=True)

        # Replace the first convolutional layer to accept the specified number of input channels
        # The original model expects 3 input channels; we adjust this to input_channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Extract the middle layers of the FCN-ResNet101 model
        # Use the backbone layers except for the first conv layer (replaced above) and the last layers
        self.classifier = nn.Sequential(*list(fcn_model.backbone.children())[1:-2])

        # Replace the last convolutional layer to output the desired number of classes (output_channels)
        # The original model outputs 21 classes (for the COCO dataset), but we set it to output_channels
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=output_channels, kernel_size=(1, 1), stride=(1, 1))

        # Final average pooling layer to scale down the output to the required resolution
        # This depends on the input image resolution and the scale_factor provided
        self.pool = nn.AvgPool2d(kernel_size=(25 // 4, 25 // 4))

    def forward(self, x):
        """
        Forward pass of the FCN-ResNet model with custom input and output layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_channels, width, height), where
            width and height should be divisible by `scale_factor`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, output_channels, width//scale_factor, height//scale_factor).
        """

        # Apply the custom first convolutional layer
        y_pred = self.conv1(x)

        # Pass through the modified middle layers (feature extraction)
        y_pred = self.classifier(y_pred)

        # Apply the custom output convolutional layer to get the desired number of classes
        y_pred = self.conv2(y_pred)

        # Pool the output to match the required resolution based on scale_factor
        y_pred = self.pool(y_pred)

        return y_pred
