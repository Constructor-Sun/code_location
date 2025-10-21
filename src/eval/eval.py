import os
import json
import argparse
from datasets import load_dataset

def convert_label(label):
    parts = label.split(':')
    left = parts[0]
    right = parts[1]

    # if there is a class name only, meaning __init__ methods shoule be modified
    if '.' not in right and not right.startswith('_') and any(char.isupper() for char in right): # and not right.startswith('_')
        right = os.path.join(right, "__init__")
    
    right = right.replace('.', '/')
    return os.path.join(left, right)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite")
    parser.add_argument("--saving", type=str, default="rank32-retrieval.json") # result-Qwen3-30B-subset, result-api-subset, result-xai-subset
    args = parser.parse_args()
    args.saving = os.path.join(args.test_dir, args.dataset + '-' + args.saving)
    if args.dataset == "swe-bench-lite":
        label_path = "czlll/SWE-bench_Lite"
    elif args.dataset == "loc-bench":
        label_path = "czlll/Loc-Bench_V1"
    dataset = load_dataset(label_path, split='test')
    id_to_edit_funcs = {}
    for item in dataset:
        instance_id = item['instance_id']
        edit_functions = item['edit_functions']
        id_to_edit_funcs[instance_id] = [convert_label(func) for func in edit_functions]
    
    with open(args.saving, "r") as file:
        preds = json.load(file)
    false_example = []
    null_example = [] # answer is null
    none_example = [] # label is null
    for key in preds:
        if "retrieval" in args.saving.split('/')[-1]:
            top_k = preds[key]["preds"]
        else:
            top_k = preds[key]
        if not isinstance(top_k, list): 
            null_example.append(key)
            continue
        if len(top_k) != 10:
            top_k_set = set(top_k)
            # consider deminish overlapping part 
            if len(top_k_set) <= 10:
                top_k = top_k_set
            # consider deminish the last one in group 1 and 3
            else:
                top_k = [item for i, item in enumerate(top_k) if i not in [3, 11]]
        ground_truth = id_to_edit_funcs.get(key)
        if ground_truth is None:
            none_example.append(key)
            continue
        for label in ground_truth:
            if label not in top_k:
                false_example.append(key)
                break
    print("false_example:", len(false_example))
    print(false_example)
    print("null_example:", len(null_example))
    print(null_example)
    print("none_example:", len(none_example))
    print(none_example)

    with open("datasets/swe-bench-lite-subset-summary.json", "r") as file:
        dict = json.load(file)
    if args.saving not in dict:
        if args.saving == "datasets/swe-bench-lite-result-Qwen3-30B-subset.json":
            key = "Qwen3-Coder-30B"
        if args.saving == "datasets/swe-bench-lite-result-api-subset.json":
            key = "Qwen3-Coder-480B"
        if args.saving == "datasets/swe-bench-lite-result-xai-subset.json":
            key = "Grok-Code-Fast-1"
        if args.saving == "datasets/swe-bench-lite-rank32-retrieval.json":
            key = "SweRank-32B"
        dict[key] = false_example
    with open("datasets/swe-bench-lite-subset-summary.json", "w") as file:
        json.dump(dict, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
