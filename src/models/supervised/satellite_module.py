import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import torchmetrics
from torchmetrics import Accuracy,AUROC,F1Score
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from torchmetrics.detection import IntersectionOverUnion as IoU
from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer

class ESDSegmentation(pl.LightningModule):
    """
    LightningModule for training a segmentation model on the ESD dataset
    """
    def __init__(self, model_type, in_channels, out_channels, class_weights=None,
                 learning_rate=1e-3, model_params: dict = {}):
        """
        Initializes the model with the given parameters.

        Input:
        model_type (str): type of model to use, one of "SegmentationCNN",
        "UNet", or "FCNResnetTransfer"
        in_channels (int): number of input channels of the image of shape
        (batch, in_channels, width, height)
        out_channels (int): number of output channels of prediction, prediction
        is shape (batch, out_channels, width//scale_factor, height//scale_factor)
        learning_rate (float): learning rate of the optimizer
        model_params (dict): dictionary of parameters to pass to the model
        """
        super().__init__()
        self.save_hyperparameters()
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        print("Learning rate:",self.learning_rate)
        print("Class_weights:",self.class_weights)

        if model_type == "SegmentationCNN":
            self.model = SegmentationCNN(in_channels,out_channels, **model_params)
        elif model_type == "UNet":
            self.model = UNet(in_channels, out_channels, **model_params)
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        ## Performance Metrics ##
        
        # Accuracy
        self.avg_acc = Accuracy(task='multiclass',num_classes=out_channels)
        self.per_class_acc = Accuracy(task='multiclass',num_classes=out_channels,average=None)

        # Area Under Curve
        self.avg_AUC = MulticlassAUROC(num_classes=out_channels,average='macro',thresholds=None)
        self.per_class_AUC = MulticlassAUROC(num_classes=out_channels,average='none',thresholds=None)

        # F1 Score
        self.avg_F1Score = MulticlassF1Score(num_classes = out_channels)
        self.per_class_F1Score = MulticlassF1Score(num_classes=out_channels,average=None)


    
    def forward(self, X):
        """
        Run the input X through the model

        Input: X, a (batch, input_channels, width, height) image
        Ouputs: y, a (batch, output_channels, width/scale_factor, height/scale_factor) image
        """
        y_pred = self.model.forward(X)
        return y_pred # list of probabilitiels falls under each class 
   

        
    def training_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then uses CrossEntropyLoss to calculate
        the current loss.

        Note: CrossEntropyLoss requires mask to be of type
        torch.int64 and shape (batches, width, height), 
        it only has one channel as the label is encoded as
        an integer index. As these may not be this shape and
        type from the dataset, you might have to use
        torch.reshape or torch.squeeze in order to remove the
        extraneous dimensions, as well as using Tensor.to to
        cast the tensor to the correct type.

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            train_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, target, metadata = batch
        target = target.squeeze(1)
        target = target.to(torch.int64)

        preds = self.forward(sat_img)
        
        if self.class_weights is not None:
            loss = F.cross_entropy(preds, target, weight=self.class_weights)
        else:
            loss = F.cross_entropy(preds, target)
        
        ## Record Performance Metrics ##
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=True)

        # Accuracy
        acc = self.avg_acc(preds, target)
        per_class_acc = self.per_class_acc(preds,target)
        
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'train_class_' + str(c+1) + '_acc'
            self.log(label, per_class_acc[c], on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        
        # Area Under Curve
        auc = self.avg_AUC(preds,target)
        per_class_auc = self.per_class_AUC(preds,target)

        self.log('train_area_under_curve',auc,on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'train_class_' + str(c+1) + '_area_under_curve'
            self.log(label,per_class_auc[c],on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

        # F1 Score
        f1 = self.avg_F1Score(preds,target)
        per_class_f1 = self.per_class_F1Score(preds,target)
        
        self.log('train_f1',f1,on_step=False, on_epoch=True, prog_bar=True, logger=True,enable_graph=True)
        for c in range(4):
            label = 'train_class_' + str(c+1) + '_f1'
            self.log(label,per_class_f1[c],on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then evaluates the 

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            val_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Should be the cross_entropy_loss, as it is the main validation
            loss that will be tracked.
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, target, batch_metadata = batch
        target = target.squeeze(1)
        target = target.to(torch.int64)

        sat_img = sat_img.float()
        
        preds = self.forward(sat_img)

        loss = F.cross_entropy(preds, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, enable_graph=True)
        
        ## Record Performance Metrics ##

        # Accuracy
        acc = self.avg_acc(preds, target)
        per_class_acc = self.per_class_acc(preds,target)
        
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'val_class_' + str(c+1) + '_acc'
            self.log(label, per_class_acc[c], on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        
        # Area Under Curve
        auc = self.avg_AUC(preds,target)
        per_class_auc = self.per_class_AUC(preds,target)

        self.log('val_area_under_curve',auc,on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        for c in range(4):
            label = 'val_class_' + str(c+1) + '_area_under_curve'
            self.log(label,per_class_auc[c],on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

        # F1 Score
        f1 = self.avg_F1Score(preds,target)
        per_class_f1 = self.per_class_F1Score(preds,target)
        
        self.log('val_f1',f1,on_step=False, on_epoch=True, prog_bar=True, logger=True,enable_graph=True)
        for c in range(4):
            label = 'val_class_' + str(c+1) + '_f1'
            self.log(label,per_class_f1[c],on_step=False, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Loads and configures the optimizer. See torch.optim.Adam
        for a default option.

        Outputs:
            optimizer: torch.optim.Optimizer
                Optimizer used to minimize the loss
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Example scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # or 'step' for step-wise updates
                'frequency': 1,
            }
        }