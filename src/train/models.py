import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNReaonser(nn.Module):
    """
    GCN layer with residual network and classifier
    """
    def __init__(self, in_channels, hidden_channels, num_classes, 
                 training=True, 
                 dropout=0.2, 
                 top_k=5, 
                 instruction_num=4,
                 embed_dim=1024):
        super(GCNReaonser, self).__init__()
        self.dropout = dropout
        self.training = training
        self.top_k = top_k
        self.instruction_num = instruction_num
        self.embed_dim = embed_dim

        self.init_layers(in_channels, hidden_channels, num_classes)

    def init_vectors(self):
        self.instructions = []
        self.instruction = torch.zeros(self.instruction_num, self.embed_dim).to(self.device)

    def init_layers(self, in_channels, hidden_channels, num_classes):
        # GCN conv
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels)
        ])
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def init_nodes(self, x, query_pool, batch, top_k=5):
        """
        batchs x: node embeddings [batch_nodes_num, emb_dim]
        batchs query_pool: query [batch_query_num, emb_dim]
        batchs batch: node embedding index in current batch [batch_nodes_num]
        top_k: select top_k nodes as initial nodes
        """
        batch_size = query_pool.shape[0]
        num_nodes = x.shape[0]
        
        # expand query to each graph, then compute cosine similarity
        query_expanded = query_pool[batch]  # [batch_nodes_num, emb_dim]
        similarities = F.cosine_similarity(x, query_expanded, dim=1)  # [batch_nodes_num]

        # mask
        p = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        for i in range(batch_size):
            # i-th graph indices
            graph_indices = (batch == i).nonzero(as_tuple=True)[0]
            
            if len(graph_indices) == 0:
                continue
                
            graph_similarities = similarities[graph_indices]
            _, topk_local_indices = torch.topk(
                graph_similarities, 
                k=min(top_k, len(graph_indices))
            )
            
            # top_k indices in one graph
            topk_global_indices = graph_indices[topk_local_indices]
            p[topk_global_indices] = 1.0
        
        return p / top_k
    
    def get_pool_query(self, query, query_mask):
        masked_query = query * query_mask
    
        valid_token_count = query_mask.sum(dim=1)  # [batch_size, 1]
        valid_token_count = torch.clamp(valid_token_count, min=1.0)
        
        # pooling at dim 1
        sum_query = masked_query.sum(dim=1)  # 形状 [batch_size, emb_dim]
        mean_query = sum_query / valid_token_count
        return mean_query
        
    def forward(self, batchs):
        """
        batchs x: node embeddings [batch_nodes_num, emb_dim]
        batchs query: query [batch_query_num, seq_len, emb_dim]
        batchs query_mask: query [batch_query_num, seq_len, 1]
        batchs batch: node embedding index in current batch [batch_nodes_num]
        batchs edge_index: edge connections [batch_edge_num, 2]
        batchs edge_type: edge type [batch_edge_num]
        """
        
        x = batchs.x
        query = batchs.query
        batch_size = query.shape[0]
        query_mask = batchs.query_mask
        batch = batchs.batch
        edge_index = batchs.edge_index
        edge_type = batchs.edge_type
        query_pool = self.get_pool_query(query, query_mask)
        q = self.init_nodes(x, query_pool, batch, self.top_k)
        self.init_vectors()
        
        # please modify the following ....
        query = query[batchs.batch]
        exit()

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
