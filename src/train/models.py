import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNReaonser(nn.Module):
    """
    GCN layer with residual network and classifier
    """
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.2):
        super(GCNReaonser, self).__init__()
        self.dropout = dropout
        
        # GCN conv
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels)
        ])
        
        # MLP
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, query):
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
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()
