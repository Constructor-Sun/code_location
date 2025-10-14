import os
import json

def main():
    ref_dict_path = "datasets/instances.json"
    target_dict_path = "datasets/swe-bench-lite-result-api.json"
    target_split = target_dict_path.split("/")
    target_save_path = os.path.join(target_split[0], target_split[1].split(".")[0] + "-subset.json")

    with open(target_dict_path, "r") as file:
        target_dict = json.load(file)
    with open(ref_dict_path, "r") as file:
        ref_dict = json.load(file)
    
    save_dict = {}
    for key in ref_dict.keys():
        if key in target_dict:
            save_dict[key] = target_dict[key]
    
    print("ref: ", len(ref_dict.keys()))
    print("save: ", len(save_dict.keys()))

    with open(target_save_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()