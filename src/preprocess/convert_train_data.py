import torch
import json
import argparse
from datasets import load_dataset
from datasets import load_dataset, Features, Array2D, Sequence, Value

# convert Parquet to Dataset
def load_from_parquet(filename):
    def process_example(example):
        example['x'] = torch.tensor(example['x'], dtype=torch.float32)  # list -> tensor float32
        example['x_mask'] = torch.tensor(example['x_mask'], dtype=torch.int32)
        example['y'] = torch.tensor(json.loads(example['y']), dtype=torch.int32)  # JSON -> tensor int32
        return example
    
    dataset = load_dataset('parquet', data_files=filename, split='train')
    dataset = dataset.map(process_example)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="train_index/SWE-smith/question_and_labels/train.parquet")
    parser.add_argument("--save_path", type=str, default="train_index/SWE-smith/question_and_labels/train_dataset")
    args = parser.parse_args()

    dataset = load_from_parquet(args.load_path)
    dataset.save_to_disk(args.save_path)

if __name__ == "__main__":
    main()