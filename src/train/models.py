import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
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
                 num_iter=3,
                 num_layers=3,
                 embed_dim=1024):
        super(GCNReaonser, self).__init__()
        self.dropout = dropout
        self.training = training
        self.top_k = top_k
        self.instruction_num = instruction_num
        self.num_iter = num_iter
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.init_layers(in_channels, hidden_channels, num_classes)

    def init_layers(self, in_channels, hidden_channels, num_classes):
        # GCN conv
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels)
        ])
        self.classifier = nn.Linear(hidden_channels, num_classes)

        for k in range(self.instruction_num):
            setattr(self, f'W_v_{k}', nn.Linear(4 * self.embed_dim, self.embed_dim, bias=False))
        self.W_u = nn.Linear(self.embed_dim, 1, bias=False)
        self.gru = nn.GRUCell(input_size=self.embed_dim, hidden_size=self.embed_dim)
        self.W_q = nn.Linear(4 * self.embed_dim, self.embed_dim, bias=False)

        
        for l in range(self.num_layers):
            setattr(self, f'W_x_{l}', nn.Linear(self.emb_dim, self.emb_dim, bias=False))
            setattr(self, f'W_h_{l}', nn.Linear((self.instruction_num + 1) * self.emb_dim, self.emb_dim, bias=False))
        
        self.w = nn.Parameter(torch.randn(self.emb_dim))

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
    
        valid_token_count = query_mask.sum(dim=1)  # [batch_query_num, 1]
        valid_token_count = torch.clamp(valid_token_count, min=1.0)
        
        # pooling at dim 1
        sum_query = masked_query.sum(dim=1)  # 形状 [batch_query_num, emb_dim]
        mean_query = sum_query / valid_token_count
        return mean_query
    
    def init_instructions(self, query, query_mask, query_pool):
        """
        Initialize instructions based on query embeddings with mask support.
        
        Returns:
            instructions: [batch_query_num, instruction_num, embed_dim] - initialized instruction vectors
        """
        batch_query_num, seq_len, embed_dim = query.shape
        device = query.device
        i_0 = torch.zeros(embed_dim)
        
        # Initialize i^(0) for each batch
        i_prev = i_0.expand(batch_query_num, 1, embed_dim).to(device)  # [batch_query_num, 1, embed_dim]
        q = query_pool.unsqueeze(1)  # [batch_query_num, 1, embed_dim]
        instructions = []
        
        for k in range(self.instruction_num):
            # Get the appropriate W_v for this iteration
            W_v = getattr(self, f'W_v_{k}')
            
            # Compute q^(k) = W_v * [i^(k-1) || q || q ⊙ i^(k-1) || q - i^(k-1)]
            concat_features = torch.cat([
                i_prev,          # i^(k-1) [batch_query_num, 1, embed_dim]
                q,               # q [batch_query_num, 1, embed_dim]
                q * i_prev,      # q ⊙ i^(k-1) [batch_query_num, 1, embed_dim]
                q - i_prev       # q - i^(k-1) [batch_query_num, 1, embed_dim]
            ], dim=-1)  # [batch_query_num, 1, 4 * embed_dim]
            
            q_k = W_v(concat_features)  # [batch_query_num, 1, embed_dim]
            
            # Compute attention weights u_j^(k)
            # q_k ⊙ b_j for each token j
            qk_expanded = q_k.expand(-1, seq_len, -1)  # [batch_query_num, seq_len, embed_dim]
            qk_dot_bj = qk_expanded * query  # [batch_query_num, seq_len, embed_dim]
            
            # Apply W_u and sum over embedding dimension to get attention scores
            attention_scores = self.W_u(qk_dot_bj).squeeze(-1)  # [batch_query_num, seq_len]
            
            # Apply mask: set padding positions to very negative value
            attention_scores = attention_scores.masked_fill(
                query_mask.squeeze(-1) == 0, -1e9
            )
            
            # Apply softmax over seq_len dimension to get attention weights
            u_j = F.softmax(attention_scores, dim=-1)  # [batch_query_num, seq_len] 
            
            # Compute i^(k) = sum_j u_j^(k) * b_j
            # Use mask to ensure padding tokens don't contribute
            weighted_query = u_j.unsqueeze(-1) * query * query_mask
            i_k = torch.sum(weighted_query, dim=1, keepdim=True)  # [batch_query_num, 1, embed_dim]
            
            # This normalization can be ignored
            # sum_weights = torch.sum(u_j.unsqueeze(-1) * query_mask, dim=1, keepdim=True)
            # i_k = i_k / (sum_weights + 1e-9)  # Avoid division by zero
            
            instructions.append(i_k)
            i_prev = i_k  # Update for next iteration
        
        # Stack all instructions along dimension 1
        instructions = torch.cat(instructions, dim=1)  # [batch_query_num, instruction_num, embed_dim]
        
        return instructions

    def update_instructions(self, p_0, node_embeddings, instructions):
        seed_mask = (p_0 > 0)
        seed_indices = seed_mask.nonzero(as_tuple=True)[0]
        seed_embeddings = node_embeddings[seed_indices]
        
        # Compute h_e by summing seed entity representations
        h_e = seed_embeddings.sum(dim=0)  # [emb_dim]
        h_e_expanded = h_e.unsqueeze(0).unsqueeze(0)  # [1, 1, emb_dim]
        
        # Process each instruction
        updated_instructions = []
        for k in range(instructions.size(1)):
            i_k = instructions[:, k, :]  # [batch_query_num, emb_dim]
            
            # Compute gate vector using GRU
            # Note: This assumes a GRU layer is defined elsewhere in the class
            _, g_k = self.gru(i_k.unsqueeze(0))  # g_k shape: [1, batch_query_num, emb_dim]
            g_k = g_k.squeeze(0)  # [batch_query_num, emb_dim]
            
            # Prepare the concatenated features
            concat_features = torch.cat([
                i_k,
                h_e_expanded.expand_as(i_k),
                i_k - h_e_expanded.expand_as(i_k),
                i_k * h_e_expanded.expand_as(i_k)
            ], dim=-1)  # [batch_query_num, 4 * emb_dim]
            
            # Apply linear transformation
            transformed = self.W_q(concat_features)  # [batch_query_num, emb_dim]
            
            # Apply gating mechanism
            updated_i_k = (1 - g_k) * i_k + g_k * transformed
            updated_instructions.append(updated_i_k)
        
        # Stack updated instructions
        updated_instructions = torch.stack(updated_instructions, dim=1)  # [batch_query_num, instruction_num, emb_dim]
        return updated_instructions
        
    def update_nodes(self, h_l, p_l, instructions, edge_index, batch_idx, layer_idx):
        """
        h_l: [batch_nodes_num, emb_dim]
        p_l: [batch_nodes_num]
        instructions: [batch_query_num, instruction_num, emb_dim]
        edge_index: [2, E]
        batch_idx: [batch_nodes_num]
        """
        W_x = getattr(self, f'W_x_{layer_idx}')
        W_h = getattr(self, f'W_h_{layer_idx}')

        src_nodes, dst_nodes = edge_index
        
        # [E, instruction_num, emb_dim]
        expanded_instructions = instructions[batch_idx[src_nodes]]
        # E, emb_dim]
        neighbor_h = W_x(h_l[src_nodes])
        # (I^(k) ⊙ W_X h_v') [E, instruction_num, emb_dim]
        messages = expanded_instructions * neighbor_h.unsqueeze(1)
        c_vk = F.relu(messages)  # [E, instruction_num, emb_dim]
        
        # [E, 1, 1]
        p_weights = p_l[src_nodes].view(-1, 1, 1)
        # [E, instruction_num, emb_dim]
        weighted_messages = p_weights * c_vk
        
        # [batch_nodes_num, instruction_num, emb_dim]
        aggregated = scatter(weighted_messages, dst_nodes, dim=0, 
                            dim_size=h_l.size(0), reduce='sum')
        
        # [batch_nodes_num, instruction_num * emb_dim]
        h_ins = aggregated.view(h_l.size(0), -1)  # [N, Q*D]
        combined = torch.cat([h_l, h_ins], dim=1)
        h_new = F.relu(W_h(combined))  # [N, D]
        
        # score for nodes
        scores = torch.matmul(h_new, self.u)
        p_new = F.softmax(scores, dim=0)
        
        return h_new, p_new
        
    def forward(self, batchs):
        """
        batchs x: node embeddings [batch_nodes_num, emb_dim]
        batchs query: query [batch_query_num, seq_len, emb_dim]
        batchs query_mask: query [batch_query_num, seq_len, 1]
        batchs batch: node embedding index in current batch [batch_nodes_num]
        batchs edge_index: edge connections [batch_edge_num, 2]
        batchs edge_type: edge type [batch_edge_num]
        """
        # basic propriety
        x = batchs.x
        query = batchs.query
        query_mask = batchs.query_mask
        batch_idx = batchs.batch
        edge_index = batchs.edge_index
        query_pool = self.get_pool_query(query, query_mask)

        p_0 = self.init_nodes(x, query_pool, batch_idx, self.top_k)
        instructions = self.init_instructions() # [batch_query_num, instruction_num, embed_dim]
        h_in = x

        for t in range(self.num_iter):
            p_l = p_0.clone()  # 每个 stage 都从种子实体开始推理
            h_current = h_in
            for l in range(self.num_layers):
            # instructions = self.update_instructions(p_0, node_embeddings, instructions)
                h_next, p_next = self.update_nodes(h_current, p_l, instructions, edge_index, batch_idx, l)
                h_current = h_next
                p_l = p_next
            h_out = h_current
            p_out = p_l
            instructions = self.update_instructions(instructions, h_out, p_0)
            h_in = h_out
        
        exit()
        return p_out

        # x = query * x
        # residuals = [x]  # residual
        
        # # GCN + ReLU + Dropout
        # for i, conv in enumerate(self.convs[:-1]):
        #     x_conv = conv(x, edge_index)
            
        #     if residuals[i].size(1) == x_conv.size(1):
        #         x_conv = x_conv + residuals[i]
            
        #     x = F.relu(x_conv)
        #     residuals.append(x)
        
        # # the last layer
        # x = self.convs[-1](x, edge_index)
        # if residuals[-1].size(1) == x.size(1):
        #     x = x + residuals[-1]

        # # final layer
        # x = self.classifier(x)
        # if self.training:
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # return x

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
