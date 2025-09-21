import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import gc
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

def load_model(retrieval_model_1):
    tokenizer_1 = AutoTokenizer.from_pretrained(retrieval_model_1, trust_remote_code=False)
    model_1 = AutoModel.from_pretrained(retrieval_model_1, trust_remote_code=False)
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

def get_ori_label(y_path):
    df = pd.read_csv(os.path.join(y_path, "qrels", "test.tsv"), sep='\t', header=0)
    corpus_ids = df['corpus-id'].tolist()
    return corpus_ids

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

def get_top_nodes(tokenizer, retrieval_model, query, graph_path, top_k):
    project_data = torch.load(graph_path, weights_only=False)
    project_embeddings = torch.from_numpy(project_data["embeddings"])
    query_tokens = tokenizer(query, padding="longest", truncation=True, return_tensors='pt', max_length=2048)
    query_tokens = {k: v.to(retrieval_model.device) for k, v in query_tokens.items()}
    query_embed = retrieval_model(**query_tokens).last_hidden_state.mean(dim=1)
    topk_similarities, topk_local_indices, p = init_nodes(project_embeddings, query_embed, torch.zeros(project_embeddings.shape[0], dtype=torch.int).to(retrieval_model.device), top_k)
    return {k: v for k, v in zip(topk_local_indices, topk_similarities)}, p

def retrieve(args):
    target = args.dataset + '-function_' + args.target
    query = get_query(os.path.join(args.test_dir, target, "queries.jsonl"))
    project_path = os.path.join(args.embed_dir, target + ".pt")
    labels = get_ori_label(os.path.join(args.test_dir, target))
    labels_index, project_embeddings = convert_to_index(project_path, labels)
    tokenizer, retrieval_model = load_model(args.retrieval_model)
    try:
        top_nodes_dic, p = get_top_nodes(tokenizer, retrieval_model, query, 
                                         project_path, 
                                         top_k=args.top_k)
        preds = get_labels(project_embeddings, p)
    finally:
        retrieval_model.to("cpu")
        del retrieval_model
        gc.collect()
        torch.cuda.empty_cache()

    return query, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="test_index")
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite") # swe-bench-lite
    parser.add_argument("--retrieval_model", type=str, default="Salesforce/SweRankEmbed-Large")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="sympy__sympy-23262")
    args = parser.parse_args()

    args.target = args.dataset + '-function_' + args.target
    query = get_query(os.path.join(args.test_dir, args.target, "queries.jsonl"))
    project_path = os.path.join(args.embed_dir, args.target + ".pt")
    labels = get_ori_label(os.path.join(args.test_dir, args.target))
    labels_index, project_embeddings = convert_to_index(project_path, labels)
    tokenizer, retrieval_model = load_model(args.retrieval_model)
    top_nodes_dic, p = get_top_nodes(tokenizer, retrieval_model, query, 
                                     project_path, 
                                     top_k=args.top_k)
    preds = get_labels(project_embeddings, p)
    print("query: ", query)
    print("labels: ", labels)
    print("label index: ", labels_index)
    print()
    print("top_nodes_dic: ", top_nodes_dic)
    print("preds: ", preds)
    print()
    print("labels not predicted in retrieval_model 1: ", list(set(labels_index) - set(top_nodes_dic.keys())))

if __name__ == "__main__":
    main()