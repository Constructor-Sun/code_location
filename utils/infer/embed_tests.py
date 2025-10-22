import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import time
import json
import argparse
import torch
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def get_filenames_in_dir(directory_path):
    filenames = []
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            filenames.append(filename.rsplit('.', 1)[0])
    return filenames

def get_example_names(dataset_path, dataset):
    folder_names = []
    if dataset == "swe-bench-lite":
        failed_file = "swe-bench-lite-instances.json"
    elif dataset == "loc-bench":
        failed_file = "loc-bench-instances.json"
    else:
        raise KeyError("no matched dataset!")
    failed_path = os.path.join(dataset_path, failed_file)
    with open(failed_path, "r") as file:
        failed_instances = json.load(file)
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.startswith(dataset) and item.split('_', 1)[1] in failed_instances:
            folder_names.append(item)
    return folder_names

def embed_text_batch(tokenizer, model, texts, max_length=2048):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=max_length, padding="longest")
    if inputs['input_ids'].shape[-1] > max_length:
        print("some inputs exceed max length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch_size, max_length, 1]
        masked_embeddings = outputs.last_hidden_state * attention_mask
        embeddings = masked_embeddings.sum(dim=1)  # [batch_size, embedding_dim]
        seq_lengths = attention_mask.sum(dim=1)  # [batch_size, 1]
        seq_lengths = torch.clamp(seq_lengths, min=1)
        embeddings = embeddings / seq_lengths
    
    return embeddings.cpu().numpy()


def embed_codes(tokenizer, model, content_list, batch_size, max_length=2048):
    embedding_dim = model.config.hidden_size  # Get embedding dimension from model
    print("embeding_dim_:", embedding_dim)
    embeddings = np.empty((len(content_list), embedding_dim), dtype=np.float32)
    
    indices_and_lengths = [(i, len(tokenizer.encode(content, add_special_tokens=False)))
                           for i, content in enumerate(content_list)]
    indices_and_lengths.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in indices_and_lengths]
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(content_list), batch_size), desc="batch in one item"):
            batch_indices = sorted_indices[i:i + batch_size]
            batch_texts = [content_list[i] for i in batch_indices]
            batch_embeddings = embed_text_batch(tokenizer, model, batch_texts, max_length)
            for j, idx in enumerate(batch_indices):
                embeddings[idx] = batch_embeddings[j]
            torch.cuda.empty_cache()  # Clear cache after each batch
    return embeddings

def embed_item_single_process(dataset_path, item, save_dir, tokenizer, model, batch_size):
    func_list = []
    content_list = []
    with open(os.path.join(dataset_path, item, "corpus.jsonl"), 'rb') as file:
        for line in file:
            if line.strip():
                it = json.loads(line)
                func_list.append(it["_id"])
                content_list.append(it["text"])
        embeddings = embed_codes(tokenizer, model, content_list, batch_size, max_length=2048)
        saving = {
            "function": func_list,
            "embeddings": embeddings
        }
        torch.save(saving, os.path.join(save_dir, item + ".pt"))

def process_chunk(dataset_path, items, save_dir, tokenizer, model, device_id, batch_size):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    for item in items:
        embed_item_single_process(dataset_path, item, save_dir, tokenizer, model, batch_size)
        print(f"Process on GPU {device_id} finished processing {item}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite")
    parser.add_argument("--save_dir", type=str, default="test_index")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--model_path", type=str, default="Salesforce/SweRankEmbed-Large")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir)
    test_items = get_example_names(dataset_path, args.dataset)
    print("test_items: ", len(test_items))
    existing_set = set(get_filenames_in_dir(args.save_dir))
    test_items = [f for f in test_items if f not in existing_set]

    num_gpus = torch.cuda.device_count()
    num_processes = min(args.num_processes, num_gpus, len(test_items))
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Using {num_processes} GPUs out of {num_gpus} available")
    print(f"Processing {len(test_items)} files")

    chunk_size = len(test_items) // num_processes
    items_chunk = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else len(test_items)
        items_chunk.append(test_items[start:end])

    processes = []
    for i in range(num_processes):
        # independent tokenizer and model for each process
        print("Using: ", args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path)
        
        # create processes
        p = mp.Process(
            target=process_chunk,
            args=(dataset_path, items_chunk[i], args.save_dir, tokenizer, model, i, args.batch_size)
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
