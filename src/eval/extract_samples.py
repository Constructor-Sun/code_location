import os
import json

def get_swe_bench_folders(folder_path, prefix):
    swe_bench_folders = {}
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            item = item.split('function_')[-1]
            swe_bench_folders[item] = ""
    
    return swe_bench_folders

if __name__ == "__main__":
    target_folder = "datasets"
    prefix = 'swe-bench-lite'
    output_file = prefix + "-instances.json"
    
    folders_dict = get_swe_bench_folders(target_folder, prefix)
    output_file = os.path.join(target_folder, output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(folders_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Total: {len(folders_dict)}")
    print(f"saved in: {output_file}")