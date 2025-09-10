import os
import json
import pickle
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

def load_model(embedding_model_1, embedding_model_2):
    tokenizer_1 = AutoTokenizer.from_pretrained(embedding_model_1, trust_remote_code=True)
    model_1 = AutoModel.from_pretrained(embedding_model_1, trust_remote_code=True)
    model_1.to("cuda:0")
    model_1.eval()

    tokenizer_2 = AutoTokenizer.from_pretrained(embedding_model_2, trust_remote_code=True)
    model_2 = AutoModel.from_pretrained(embedding_model_2, trust_remote_code=True)
    model_2.to("cuda:1")
    model_2.eval()

    return tokenizer_1, model_1, tokenizer_2, model_2

def get_sample(dataset, id):
    dataset = load_from_disk(dataset)
    sample = dataset[id]
    x = sample['x']
    y = sample['y']
    image = sample['image']
    return x, y, image

def get_labels(graph_index, y, image):
    with open(os.path.join(graph_index, image + '.pkl'), 'rb') as file:
        code_map = pickle.load(file)
        nodes = code_map.nodes(data=False)
        selected_nodes = [node for node, flag in zip(nodes, y) if flag == 1.]
    return selected_nodes

def get_ori_labels(ori_dataset, id):
    with open(ori_dataset, 'r') as f:
        instances = json.load(f)
    edit_functions = instances["edit_functions"]
    ori_labels = edit_functions[id]
    return ori_labels

def init_nodes(x, query_pool, batch, top_k=5):
        """
        batchs x: node embeddings [batch_nodes_num, embed_dim]
        batchs query_pool: query [batch_query_num, embed_dim]
        batchs batch: node embedding index in current batch [batch_nodes_num]
        top_k: select top_k nodes as initial nodes
        """
        x = x.to(query_pool.device)
        batch_size = query_pool.shape[0]
        num_nodes = x.shape[0]
        
        # expand query to each graph, then compute cosine similarity
        query_expanded = query_pool[batch]  # [batch_nodes_num, embed_dim]
        similarities = F.cosine_similarity(x, query_expanded, dim=1)  # [batch_nodes_num]

        # mask
        p = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        for i in range(batch_size):
            # i-th graph indices
            graph_indices = (batch == i).nonzero(as_tuple=True)[0]
            
            if len(graph_indices) == 0:
                continue
                
            graph_similarities = similarities[graph_indices]
            topk_similarities, topk_local_indices = torch.topk(
                graph_similarities, 
                k=min(top_k, len(graph_indices))
            )
            
            # top_k indices in one graph
            topk_global_indices = graph_indices[topk_local_indices]
            p[topk_global_indices] = 1.0
            return topk_similarities.tolist(), topk_local_indices.tolist(), p
        
        # return p / top_k

def get_top_nodes(tokenizer, model, query, graph_path, top_k):
    graph_data = torch.load(graph_path, weights_only=False).to(model.device)
    query_tokens = tokenizer(query, padding=True, truncation=True, return_tensors='pt', max_length=2048)
    query_tokens = {k: v.to(model.device) for k, v in query_tokens.items()}
    query_embed = model(**query_tokens).last_hidden_state.mean(dim=1)
    topk_similarities, topk_local_indices, p = init_nodes(graph_data.x, query_embed, torch.zeros(graph_data.x.size(0), dtype=torch.int).to(model.device), top_k)
    return {k: v for k, v in zip(topk_local_indices, topk_similarities)}, p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model_1", type=str, default="Salesforce/SweRankEmbed-Small")
    parser.add_argument("--embedding_model_2", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--ori_dataset", type=str, default="train_index/SWE-smith/filtered_instances/python_instances.json")
    parser.add_argument("--graph_index", type=str, default="train_index/SWE-smith/graph_index_v1.0")
    parser.add_argument("--graph_embedding_1", type=str, default="train_index/SWE-smith/graph_embedding_SweRank")
    parser.add_argument("--graph_embedding_2", type=str, default="train_index/SWE-smith/graph_embedding_pool")
    parser.add_argument("--dataset", type=str, default="train_index/SWE-smith/question_and_labels/train_dataset")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--id", type=int, default=6)
    args = parser.parse_args()

    check_labels = False
    check_similarity = True
    tokenizer_1, model_1, tokenizer_2, model_2 = load_model(args.embedding_model_1, args.embedding_model_2)
    x, y, image = get_sample(args.dataset, args.id)
    x = """
a. Function: Response serialization functions (e.g., JSON encoding methods)

Original Intended Function: These functions convert the response data dictionary into a JSON string for the response body. They should maintain the field order and include all provided fields.
Why Error Occurs: The error might be due to:

Serialization logic that filters out fields with None or empty values, incorrectly removing device_code if it is not generated.
Incorrect configuration of JSON serialization that alters the field order or omits fields unexpectedly.
"""
    print("x: ", x)
    labels = get_labels(args.graph_index, y, image)
    labels_index = np.where(np.array(y) == 1)[0].tolist()
    print("labels: ", labels)
    print("y index: ", labels_index)
    if check_labels:
        ori_labels = get_ori_labels(args.ori_dataset, args.id)
        print("ori_labels: ", ori_labels)
        print()
        # print("ori_labels: ", ori_labels)
    if check_similarity: 
        query_prefix = 'Represent this query for searching relevant code: '
        top_nodes_dic_1, p_1 = get_top_nodes(tokenizer_1, model_1, x, os.path.join(args.graph_embedding_1, image + '.pt'), args.top_k)
        top_nodes_dic_2, p_2 = get_top_nodes(tokenizer_2, model_2, query_prefix + x, os.path.join(args.graph_embedding_2, image + '.pt'), args.top_k)
        preds_1 = get_labels(args.graph_index, p_1, image)
        preds_2 = get_labels(args.graph_index, p_2, image)
        print("top_nodes_dic_1: ", top_nodes_dic_1)
        print("preds_1: ", preds_1)
        print()
        print("top_nodes_dic_2: ", top_nodes_dic_2)
        print("preds_2: ", preds_2)
        print()
        print("labels not predicted in model 1: ", list(set(labels_index) - set(top_nodes_dic_1.keys())))
        print("labels not predicted in model 2: ", list(set(labels_index) - set(top_nodes_dic_2.keys())))


if __name__ == "__main__":
    main()
