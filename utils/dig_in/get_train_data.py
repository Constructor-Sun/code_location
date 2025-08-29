import os
import torch
import argparse
import json
import time
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool
from datasets import load_dataset, Features, Array2D, Sequence, Value
from transformers import AutoTokenizer, AutoModel

# 保存为 Parquet
def save_to_parquet(data, filename):
    # 转换为 DataFrame
    df = pd.DataFrame({
        'x': [x.numpy().tolist() for x, _, _ in data],  # torch.Tensor 转 list
        'y': [json.dumps(y) for _, y, _ in data],      # 序列化变长列表
        'image': [image for _, _, image in data]       # 直接存储字符串
    })
    
    # 保存为 Parquet
    df.to_parquet(filename, engine='pyarrow', index=False, compression='snappy')
    print(f"数据已保存到 {filename}")

# 加载 Parquet 并转换为 Dataset
def load_from_parquet(filename):
    # 使用 load_dataset 加载
    dataset = load_dataset('parquet', data_files=filename, split='train')
    
    # 反序列化 y，转换 x 为 torch.Tensor
    def process_example(example):
        example['y'] = json.loads(example['y'])  # 反序列化 y
        example['x'] = torch.tensor(example['x'], dtype=torch.float32)  # 转回 Tensor
        return example
    
    dataset = dataset.map(process_example)
    
    # 定义 Features（优化类型）
    features = Features({
        'x': Array2D(shape=(1024,), dtype='float32'),  # 1024 维 embedding
        'y': Sequence(Value('int32')),                 # 变长整数列表
        'image': Value('string')                       # 字符串
    })
    dataset = dataset.cast(features)
    
    return dataset

def embed_question_batch(tokenizer, model, question):
    inputs = tokenizer(question, return_tensors="pt", max_length=4096, padding="longest")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # [batch_size, embedding_dim]
    return embeddings.cpu().numpy()

# def embed_questions(tokenizer, model, questions, device_id, batch_size):
#     device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()

#     for question_batch in questions:

def embed_questions_wrapper(args):
    model_path, questions_chunk, device_id, batch_size = args
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    # 使用列表收集每个批次的numpy数组
    batch_embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(questions_chunk), batch_size):
            batch_questions = questions_chunk[i:i + batch_size]
            batch_embeddings = embed_question_batch(tokenizer, model, batch_questions)
            batch_embeddings_list.append(batch_embeddings)
            torch.cuda.empty_cache()
    
    # 将所有批次的numpy数组合并成一个大的numpy数组
    return np.concatenate(batch_embeddings_list, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="train_index/SWE-smith/filtered_instances/python_instances.json")
    parser.add_argument("--save_path", type=str, default="train_index/SWE-smith/question_and_labels")
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    with open(args.load_path, "r") as f:
        json_data = json.load(f)
        problems, y = json_data["problem_statement"], json_data["edit_functions_ord"]
        images, image_num = json_data["image_name"], json_data["image_node_num"]

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
                for i in len(range(question_chunks))
            ]
            
            start_time = time.time()
            with Pool(processes=num_processes) as pool:
                results = pool.map(embed_questions_wrapper, task_args)
            embed_problems = np.concatenate(results, axis=0)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"All processes completed, using {execution_time:.3f} s")

        # TODO: finish label converting here


        

if __name__ == "__main__":
    main()