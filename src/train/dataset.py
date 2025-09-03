import os
import torch
from torch_geometric.data import Dataset

class GraphDataset(Dataset):
    """
    Custom dataset for lazy loading of graph data from .pt files.
    
    Args:
        queries: List of query vectors
        categories: List of category vectors (labels)
        pt_paths: List of paths to .pt files (relative to graph_embedding_dir)
        graph_embedding_dir: Directory containing .pt files
    """
    def __init__(self, queries, categories, pt_paths, graph_embedding):
        super(GraphDataset, self).__init__()
        self.queries = queries
        self.categories = categories
        self.pt_paths = pt_paths
        self.graph_embedding_dir = graph_embedding

    def len(self):
        return len(self.queries)

    def get(self, idx):
        # Load graph data from .pt file
        pt_path = os.path.join(self.graph_embedding_dir, self.pt_paths[idx] + '.pt')
        graph_data = torch.load(pt_path, weights_only=False)
        
        # Attach query and category to the Data object
        graph_data.query = torch.tensor(self.queries[idx]).view(1, -1)
        graph_data.category = torch.tensor(self.categories[idx])# .view(1, -1)

        # print("graph_data: ", graph_data)
        # print("graph_data.query;", graph_data.query.shape)
        # print("graph_data.category:", graph_data.category.shape)
        
        return graph_data