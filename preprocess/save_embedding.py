import os
import argparse
import pickle
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from functools import partial
from config_file import NODE_TYPE_MAPPING, EDGE_TYPE_MAPPING

def get_filenames_in_dir(directory_path):
    filenames = []
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            filenames.append(filename.rsplit('.', 1)[0])
    return filenames

def embed_text_batch(tokenizer, model, texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=4096, padding="longest")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # [batch_size, embedding_dim]
    return embeddings.cpu().numpy()

def embed_codes(tokenizer, model, graph, batch_size=16):
    node_ids = list(graph.nodes)
    embedding_dim = model.config.hidden_size  # Get embedding dimension from model
    embeddings = np.empty((len(node_ids), embedding_dim), dtype=np.float32)
    
    indices_and_lengths = [(i, len(tokenizer.encode(graph.nodes[node].get('code', node), add_special_tokens=False))) 
                           for i, node in enumerate(node_ids)]
    indices_and_lengths.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in indices_and_lengths]
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(node_ids), batch_size):
            batch_indices = sorted_indices[i:i + batch_size]
            batch_texts = [graph.nodes[node_ids[idx]].get('code', node_ids[idx]) for idx in batch_indices]
            batch_embeddings = embed_text_batch(tokenizer, model, batch_texts)
            for j, idx in enumerate(batch_indices):
                embeddings[idx] = batch_embeddings[j]
            torch.cuda.empty_cache()  # Clear cache after each batch
    
    for i, node in enumerate(node_ids):
        graph.nodes[node]['embedding'] = embeddings[i]
    return graph

def embed_graph_single_process(graph_dir, graph_filename, save_dir, tokenizer, model, batch_size):
    """graph embedding in one process"""
    with open(os.path.join(graph_dir, graph_filename + '.pkl'), 'rb') as f:
        graph = pickle.load(f)
        graph = embed_codes(tokenizer, model, graph, batch_size)

        # node info
        node_ids = list(graph.nodes)
        node_type = [NODE_TYPE_MAPPING.get(graph.nodes[n]['type'], -1) for n in node_ids] # [num_nodes]
        embeddings_list = [graph.nodes[n]['embedding'] for n in node_ids]
        node_embedding = np.vstack(embeddings_list)
        node_embedding = torch.tensor(node_embedding, dtype=torch.float)  # [num_nodes, embedding_dim]

        # edge info
        edge_index = []
        edge_type = []
        for u, v, data in graph.edges(data=True):
            if isinstance(u, str):
                u = node_ids.index(u)
            if isinstance(v, str):
                v = node_ids.index(v)
            edge_index.append([u, v])
            edge_type.append(EDGE_TYPE_MAPPING.get(data['type'], -1))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
        edge_type = torch.tensor(edge_type, dtype=torch.long)  # [num_edges]

        # Data object
        data = Data(
            x=node_embedding,           # code embeddings
            edge_index=edge_index,      # edge index
            node_type=node_type,        # node type
            edge_type=edge_type         # edge type
        )

        # save as .pt file
        torch.save(data, os.path.join(save_dir, f"{graph_filename}.pt"))

def process_graph_chunk(graph_dir, graph_files, save_dir, tokenizer, model, device_id, batch_size):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    for graph_file in graph_files:
        embed_graph_single_process(graph_dir, graph_file, save_dir, tokenizer, model, batch_size)
        print(f"Process on GPU {device_id} finished processing {graph_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="czlll/SWE-bench_Lite")
    parser.add_argument("--index_dir", type=str, default="index_data")
    parser.add_argument("--graph_name", type=str, default="graph_index_v2.3")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument('--num_processes', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    dataset_name = args.dataset.split('/')[-1]
    args.graph_dir = f'{args.index_dir}/{dataset_name}/{args.graph_name}/'
    args.index_dir = f'{args.index_dir}/{dataset_name}/graph_embedding/'
    os.makedirs(args.index_dir, exist_ok=True)

    graph_files = get_filenames_in_dir(args.graph_dir)
    existing_set = set(get_filenames_in_dir(args.index_dir))
    graph_files = [f for f in graph_files if f not in existing_set]

    if not graph_files:
        print("No new graph files to process")
        return
    else:
        print(f"Processing {len(graph_files)} graph files")

    # If not gpu available
    if not torch.cuda.is_available():
        print("No GPU available, using CPU")
        device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        model = model.to(device)
        for graph_file in graph_files:
            embed_graph_single_process(args.graph_dir, graph_file, args.index_dir, tokenizer, model, "cpu")
        return
    
    # Multiple GPU available
    num_gpus = torch.cuda.device_count()
    num_processes = min(args.num_processes, num_gpus, len(graph_files))
    
    print(f"Using {num_processes} GPUs out of {num_gpus} available")
    print(f"Processing {len(graph_files)} graph files")
    
    # Divide graph files into chunks
    chunk_size = len(graph_files) // num_processes
    graph_chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else len(graph_files)
        graph_chunks.append(graph_files[start:end])
    
    # Process pools
    processes = []
    for i in range(num_processes):
        # independent tokenizer and model for each process
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        
        # create processes
        p = mp.Process(
            target=process_graph_chunk,
            args=(args.graph_dir, graph_chunks[i], args.index_dir, tokenizer, model, i, args.batch_size)
        )
        processes.append(p)
        p.start()
    
    # waiting...
    start_time = time.time()
    for p in processes:
        p.join()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"All processes completed, using {execution_time:.3f} s")


if __name__ == "__main__":
    main()