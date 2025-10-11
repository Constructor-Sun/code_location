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
    parser.add_argument("--saving", type=str, default="result-sample.json") # rank32-retrieval, result-api
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
        # top_k = [item for i, item in enumerate(top_k) if i not in [11, 7]]
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

if __name__ == "__main__":
    main()
