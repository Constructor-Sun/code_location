import os
import torch
import argparse
import json
import time
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp
from multiprocessing import Pool
from transformers import AutoTokenizer, AutoModel

# save as Parquet
def save_to_parquet(data, filename):
    # ensure that the lengths are the same
    assert len(data['x']) == len(data['y']) == len(data['image'])
    # use DataFrame
    df = pd.DataFrame({
        'x': [x.tolist() if hasattr(x, 'tolist') else x for x in data['x']],
        'y': [json.dumps(y) for y in data['y']],
        'image': data['image']
    })
    
    df.to_parquet(filename, engine='pyarrow', index=False, compression='snappy')
    print(f"parquet saving at {filename}")

def embed_question_batch(tokenizer, model, question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=4096, padding="longest")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # [batch_size, embedding_dim]
    return embeddings.cpu().numpy()

def embed_questions_wrapper(args):
    model_path, questions_chunk, device_id, batch_size = args
    # analyze args from pool map
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    
    # compute batch embeddings
    batch_embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(questions_chunk), batch_size):
            batch_questions = questions_chunk[i:i + batch_size]
            batch_embeddings = embed_question_batch(tokenizer, model, batch_questions)
            batch_embeddings_list.append(batch_embeddings)
            torch.cuda.empty_cache()
    
    return np.concatenate(batch_embeddings_list, axis=0)

def indices_to_class(indices_list: list[list[int]], lengths: list[int]) -> list[list[int]]:
    """
    convert index into class
    """
    result = []
    
    for indices, length in zip(indices_list, lengths):
        binary_list = [0] * length
        for idx in indices:
            if idx < length:
                binary_list[idx] = 1
            else:
                print("error occurs idex:{}, length:{}".format(idx, length))
        result.append(binary_list)
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="train_index/SWE-smith/filtered_instances/python_instances.json")
    parser.add_argument("--save_path", type=str, default="train_index/SWE-smith/question_and_labels/train.parquet")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.load_path, "r") as f:
        json_data = json.load(f)
        problems, y = json_data["problem_statement"], json_data["edit_functions_ord"]
        image_name, image_num = json_data["image_name"], json_data["image_node_num"]

        if not torch.cuda.is_available():
            print("No GPU available, using CPU")
            device = torch.device("cpu")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
            model = model.to(device)
            model.eval()
            for problem in problems:
                embed_problems = embed_question_batch(tokenizer, model, problem)
        else:
            num_gpus = torch.cuda.device_count()
            num_processes = num_gpus

            print(f"Using {num_processes} GPUs out of {num_gpus} available")
            print(f"Processing {len(problems)} graph files")

            chunk_size = len(problems) // num_processes
            question_chunks = []
            for i in range(num_processes):
                start = i * chunk_size
                end = start + chunk_size if i < num_processes - 1 else len(problems)
                question_chunks.append(problems[start:end])

            task_args = [
                (args.model_path, question_chunks[i], i, args.batch_size)
                for i in range(len(question_chunks))
            ]

            start_time = time.time()
            with Pool(processes=num_processes) as pool:
                results = pool.map(embed_questions_wrapper, task_args)
            embed_problems = np.concatenate(results, axis=0)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"All processes completed, using {execution_time:.3f} s")

        converted_y = indices_to_class(y, image_num)
        data = {
            "x": embed_problems,
            "y": converted_y,
            "image": image_name
        }
        save_to_parquet(data, args.save_path)

if __name__ == "__main__":
    main()