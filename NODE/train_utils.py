import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import mean_absolute_error

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(MaskedCrossEntropyLoss, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        self.cross_entropy_crit = nn.CrossEntropyLoss(ignore_index = 0)
        
    def forward(self, inputs, targets):
        """Compute the CrossEntropyLoss of the next activity prediction 
        head while masking the predictions coresponding to padding events. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the unnormalized logits for each 
            activity class. Shape (batch_size, window_size, num_classes) 
            and dtype torch.float32.
        targets : torch.Tensor
            The activity labels, containing the indices. Shape 
            (batch_size, window_size), dtype torch.int64. 

        Returns
        -------
        loss: torch.Tensor
            The masked cross entropy loss for the activity prediction head. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size, num_classes)
        inputs = torch.reshape(input=inputs, shape=(-1, self.num_classes))
        # Reshape targets to shape (batch_size*window_size,)
        targets = torch.reshape(input=targets, shape=(-1,))

        # Compute masked loss 
        loss = self.cross_entropy_crit(inputs, targets) # scalar tensor

        return loss
    
class MaskedMeanAbsoluteErrorLoss(nn.Module):
    def __init__(self):
        super(MaskedMeanAbsoluteErrorLoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss in which the 
        target values of -100.0, corresponding to padded event tokens, 
        are ignored / masked and hence do not contribute to the input 
        gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the time 
            till next event target. Shape (batch_size, window_size, 1) 
            and dtype torch.float32.
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        loss: torch.Tensor
            The masked MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size,)
        inputs = torch.reshape(input=inputs, shape=(-1,))
        # Reshape targets to shape (batch_size*window_size,)
        targets= torch.reshape(input=targets, shape=(-1,))

        # Create mask to ignore time targets with value -100
        mask = (targets != -100).float()

        absolute_erros = torch.abs(inputs-targets) # (batch_size * window_size,)

        masked_absolute_erros = absolute_erros * mask # (batch_size * window_size,)

        # count: number of non-ignored targets 
        count = torch.sum(mask)

        # Compute masked loss 
        return torch.sum(masked_absolute_erros) / count
    
class RemainingRunTimeMAELoss(nn.Module):
    def __init__(self):
        super(RemainingRunTimeMAELoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss for the optional 
        remaining runtime predictions. This loss only takes into account 
        the predictions and labels on the first decoding step, since 
        we only want to make one remaining runtime prediction for each 
        prefix. The predictions and labels corresponding to the remaining 
        'window_size-1' decoding steps are ignored and do not contribute 
        to the input gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for either 
            the time till next event target, or the total remaining time 
            target. Shape (batch_size, window_size, 1) 
            and dtype torch.float32.
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        loss: torch.Tensor
            The masked MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Only select the predictions and labels of first dec step
        inputs = inputs[:, 0, 0] # (batch_size, )
        targets = targets[:, 0, 0] # (batch_size, )

        absolute_erros = torch.abs(inputs-targets) # (batch_size * window_size,)

        return torch.sum(absolute_erros) / absolute_erros.shape[0] # scalar 

class MultiOutputLoss(nn.Module):
    def __init__(self, num_classes, remaining_runtime_head, outcome_bool):
        """Composite loss function for the following three jointly 
        learned prediction tasks: 

        #. activity suffix prediction (default)

        #. time till next event suffix predicion (default)

        #. remaining runtime prediction
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        """
        super(MultiOutputLoss, self).__init__()
        self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
        self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()
        self.cont_loss_fn_rrt = RemainingRunTimeMAELoss()

    # def forward(self, cat_output, ttne_output, rrt_output, cat_target, ttne_target, rrt_target):
    def forward(self, outputs, labels):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            model's predictions for one of the three tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            labels for one of the three tasks.

        Returns
        -------
        loss : torch.Tensor
            Scalar tensor. Contains the composite loss that is used for 
            updating the gradients during training. Gradient tracking 
            turned on.
        cat_loss.item() : float
            Native python float. The (masked) cross entropy loss for 
            the next activity prediction head. Not used for gradient 
            updates during training, but for keeping track of the 
            different loss components during training and evaluation.
        cont_loss1.item() : float
            Native python float. The (masked) MAE loss for the time 
            till next event prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        cont_loss2.item() : float
            Native python float. The (masked) MAE loss for the complete 
            remaining runtime prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1])
        
        # Loss Time Till Next Event (ttne) suffix prediction
        cont_loss1 = self.cont_loss_fn_ttne(outputs[1], labels[0])

        # Loss remaining runtime (rrt) prediction
        cont_loss2 = self.cont_loss_fn_rrt(outputs[2], labels[1])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss1 + cont_loss2

        # Composite loss, act suffix loss, ttne loss, rrt loss
        return loss, cat_loss.item(), cont_loss1.item(), cont_loss2.item()