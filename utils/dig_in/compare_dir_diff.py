import pandas as pd
import os

# 读取Parquet文件并提取instance_id
def extract_instance_ids():
    file_path = "czlll/SWE-bench_Lite/data/test-00000-of-00001.parquet"
    df = pd.read_parquet(file_path)
    instance_ids = df['instance_id'].tolist()
    return set(instance_ids)  # 使用集合便于后续比较

# 读取graph_index目录中的所有文件名
def get_graph_index_filenames():
    directory_path = "index_data/SWE-bench_Lite/graph_index_v2.3"
    filenames = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            filenames.append(filename.split(".")[0])
    
    return set(filenames)

# 比较两个集合的差异
def compare_sets(set1, set2, name1="instance_ids", name2="filenames"):
    # 交集 - 两个集合中都存在的元素
    intersection = set1 & set2
    # 只在set1中存在的元素
    only_in_set1 = set1 - set2
    # 只在set2中存在的元素
    only_in_set2 = set2 - set1
    
    print(f"=== 比较结果 ===")
    print(f"{name1} 总数: {len(set1)}")
    print(f"{name2} 总数: {len(set2)}")
    print(f"相同元素数量: {len(intersection)}")
    print(f"只在 {name1} 中的数量: {len(only_in_set1)}")
    print(f"只在 {name2} 中的数量: {len(only_in_set2)}")
    
    # 如果有差异，可以选择显示部分差异项
    if only_in_set1:
        print(f"\n前10个只在 {name1} 中的元素:")
        print(list(only_in_set1)[:10])
    
    if only_in_set2:
        print(f"\n前10个只在 {name2} 中的元素:")
        print(list(only_in_set2)[:10])
    
    return intersection, only_in_set1, only_in_set2

# 主函数
def main():
    print("正在提取instance_id...")
    instance_ids = extract_instance_ids()
    print(f"成功提取 {len(instance_ids)} 个instance_id")
    
    print("\n正在读取graph_index目录中的文件名...")
    filenames = get_graph_index_filenames()
    print(f"成功读取 {len(filenames)} 个文件名")
    
    print("\n开始比较...")
    intersection, only_in_instance, only_in_files = compare_sets(
        instance_ids, filenames, "instance_ids", "graph_index文件"
    )
    
    # 如果需要保存差异结果到文件
    if only_in_instance:
        with open("only_in_instance_ids.txt", "w") as f:
            for item in only_in_instance:
                f.write(f"{item}\n")
        print("已将只在instance_ids中的元素保存到 only_in_instance_ids.txt")
    
    if only_in_files:
        with open("only_in_filenames.txt", "w") as f:
            for item in only_in_files:
                f.write(f"{item}\n")
        print("已将只在graph_index文件中的元素保存到 only_in_filenames.txt")

if __name__ == "__main__":
    main()