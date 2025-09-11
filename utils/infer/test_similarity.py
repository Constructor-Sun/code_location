import os
import json
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel

def get_query(path):
    with open(path, 'rb') as file:
        for line in file:
            if line.strip():
                it = json.loads(line)
                return it["text"]
            
def load_model(embedding_model_1):
    tokenizer_1 = AutoTokenizer.from_pretrained(embedding_model_1, trust_remote_code=True)
    model_1 = AutoModel.from_pretrained(embedding_model_1, trust_remote_code=True)
    model_1.to("cuda:0")
    model_1.eval()
    return tokenizer_1, model_1

def convert_to_index(project_path, label):
    project_data = torch.load(project_path, weights_only=False)
    project_embeddings = project_data["function"]
    label_index = [project_embeddings.index(item) for item in label]
    return label_index, project_embeddings

def get_labels(project_embeddings, p):
    selected_nodes = [node for node, flag in zip(project_embeddings, p) if flag == 1.]
    return selected_nodes

def get_ori_labels(y_path):
    df = pd.read_csv(os.path.join(y_path, "qrels", "test.tsv"), sep='\t', header=0)
    corpus_ids = df['corpus-id'].tolist()
    print(corpus_ids)

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
            
            topk_global_indices = graph_indices[topk_local_indices]
            p[topk_global_indices] = 1.0
            return topk_similarities.tolist(), topk_local_indices.tolist(), p

def get_top_nodes(tokenizer, model, query, graph_path, top_k):
    project_data = torch.load(graph_path, weights_only=False)
    project_embeddings = project_data["embeddings"].to(model.device)
    query_tokens = tokenizer(query, padding=True, truncation=True, return_tensors='pt', max_length=2048)
    query_tokens = {k: v.to(model.device) for k, v in query_tokens.items()}
    query_embed = model(**query_tokens).last_hidden_state.mean(dim=1)
    topk_similarities, topk_local_indices, p = init_nodes(project_embeddings, query_embed, torch.zeros(project_embeddings.shape[0], dtype=torch.int).to(model.device), top_k)
    return {k: v for k, v in zip(topk_local_indices, topk_similarities)}, p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="test_index")
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite")
    parser.add_argument("--model", type=str, default="Salesforce/SweRankEmbed-Small")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="swe-bench-lite-function_astropy__astropy-12907")
    args = parser.parse_args()
    
    query = get_query(os.path.join(args.datasets, args.target, "queries.jsonl"))
    project_path = os.path.join(args.embed_dir, args.target + ".pt")
    labels = get_ori_labels(os.path.join(args.datasets, args.target))
    labels_index, project_embeddings = project_embeddings(project_path, labels)
    tokenizer, model = load_model(args.model)
    top_nodes_dic, p = get_top_nodes(tokenizer, model, query, 
                                     project_path, 
                                     top_k=args.top_k)
    preds = get_labels(project_embeddings, p)
    print("labels: ", labels)
    print("label index: ", labels_index)
    print()
    print("top_nodes_dic: ", top_nodes_dic)
    print("preds: ", preds)
    print()
    print("labels not predicted in model 1: ", list(set(labels_index) - set(top_nodes_dic.keys())))

if __name__ == "__main__":
    main()