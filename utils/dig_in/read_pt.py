import torch
from torch_geometric.data import Data
from collections.abc import Mapping, Sequence

# Assuming 'data' is your Data object
def parse_data_object(data):
    print("Data object contents:")
    print("Keys:", data.keys)
    print("function: ", len(data["function"]))
    print("embeddings: ", data["embeddings"].shape)
    
    # Node features
    if hasattr(data, 'x'):
        print(f"Node features (x): shape={data.x.shape}")
    
    # Edge index
    if hasattr(data, 'edge_index'):
        print(f"Edge index: shape={data.edge_index.shape}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
    
    # Edge attributes
    if hasattr(data, 'node_type'):
        print(f"Edge attributes: shape={len(data.node_type)}")
    
    # Target labels
    if hasattr(data, 'edge_type'):
        print(f"Edge_type: shape={data.edge_type.shape}")

    if hasattr(data, 'function'):
        print(f"Function: shape={data.function}")

    if hasattr(data, 'embeddings'):
        print(f"Embeddings: shape={data.embeddings}")

# Example usage
data = torch.load('test_index/swe-bench-lite-function_sympy__sympy-19007.pt', map_location='cpu', weights_only=False)

parse_data_object(data)