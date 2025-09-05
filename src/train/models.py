import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNReaonser(nn.Module):
    """
    GCN layer with residual network and classifier
    """
    def __init__(self, in_channels, hidden_channels, num_classes, training=True, dropout=0.2):
        super(GCNReaonser, self).__init__()
        self.dropout = dropout
        self.training = training
        
        # GCN conv
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels)
        ])
        
        # MLP
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def 
        
    def forward(self, batchs):
        x = batchs.x
        query = batchs.query
        query = query[batchs.batch]
        edge_index = batchs.edge_index

        x = query * x
        residuals = [x]  # residual
        
        # GCN + ReLU + Dropout
        for i, conv in enumerate(self.convs[:-1]):
            x_conv = conv(x, edge_index)
            
            if residuals[i].size(1) == x_conv.size(1):
                x_conv = x_conv + residuals[i]
            
            x = F.relu(x_conv)
            residuals.append(x)
        
        # the last layer
        x = self.convs[-1](x, edge_index)
        if residuals[-1].size(1) == x.size(1):
            x = x + residuals[-1]

        # final layer
        x = self.classifier(x)
        if self.training:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def save_model(self, path):
        """
        Save the entire model object to a file.
        
        Args:
            path (str): File path to save the model (e.g., 'model.pth')
        """
        torch.save(self, path)
        print(f"Model saved to {path}")
