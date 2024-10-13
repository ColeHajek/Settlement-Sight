import torch
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F

from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.segmentation_cnn import SegmentationCNN

class ESDSegmentation(pl.LightningModule):
    def __init__(
        self,
        model_type,
        in_channels,
        out_channels,
        learning_rate=1e-3,
        model_params: dict = {},
        lambda_l1=0.01,
        dropout_prob = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        self.dropout_prob = dropout_prob
        # Initialize model based on the model_type parameter
        if model_type.lower() == "unet":
            self.model = UNet(in_channels=in_channels, out_channels=out_channels,dropout_prob=dropout_prob, **model_params)
        elif model_type.lower() == "segmentation_cnn":
            self.model = SegmentationCNN(in_channels=in_channels, out_channels=out_channels, **model_params)
        elif model_type.lower() == "fcn_resnet_transfer":
            self.model = FCNResnetTransfer(in_channels=in_channels, out_channels=out_channels, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize the performance metrics for the semantic segmentation task
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels)
        self.train_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=out_channels, average="weighted")
        self.val_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=out_channels, average="weighted")

        # Class-wise metrics
        self.train_classwise_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels, average=None)
        self.val_classwise_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels, average=None)
        self.train_classwise_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=out_channels, average=None)
        self.val_classwise_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=out_channels, average=None)

    def forward(self, X):
        X = torch.nan_to_num(X)
        return self.model(X)

    def l1_regularization(self):
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        return self.lambda_l1 * l1_penalty

    def compute_and_log_metrics(self, preds, target, prefix):
        """
        Helper function to compute loss and log metrics.
        Args:
            preds: Predictions from the model.
            target: Ground truth labels.
            prefix: "train" or "val" to specify the phase for logging.
        """
        # Compute cross-entropy loss 
        loss = F.cross_entropy(preds, target)

        # If lambda_l1 is being used add the penalty
        if self.lambda_l1!=0:
            loss += self.l1_regularization()
        
        self.log(f'{prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Compute and log accuracy
        if prefix == "train":
            acc = self.train_accuracy(preds, target)
            f1 = self.train_f1_score(preds, target)
            classwise_acc = self.train_classwise_accuracy(preds, target)
            classwise_f1 = self.train_classwise_f1_score(preds, target)
        else:  # val
            acc = self.val_accuracy(preds, target)
            f1 = self.val_f1_score(preds, target)
            classwise_acc = self.val_classwise_accuracy(preds, target)
            classwise_f1 = self.val_classwise_f1_score(preds, target)

        # Log overall accuracy and F1 score
        self.log(f'{prefix}_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log class-wise metrics
        for class_idx in range(len(classwise_acc)):
            self.log(f'{prefix}_class_accuracy_{class_idx+1}', classwise_acc[class_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{prefix}_class_f1_score_{class_idx+1}', classwise_f1[class_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model. Computes and logs metrics for training.
        """
        sat_img, target = batch
        target = target.squeeze(1).to(torch.int64)  # Ensure correct shape and type
        preds = self(sat_img)  # Forward pass

        # Compute and log training loss and metrics
        loss = self.compute_and_log_metrics(preds, target, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model. Computes and logs metrics for validation.
        """
        sat_img, target = batch
        target = target.squeeze(1).to(torch.int64)  # Ensure correct shape and type
        preds = self(sat_img)  # Forward pass

        # Compute and log validation loss and metrics
        loss = self.compute_and_log_metrics(preds, target, prefix="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
            }
        }