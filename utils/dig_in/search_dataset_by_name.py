from datasets import load_dataset

def find_sample_by_image_name(dataset, target_filename):
    """
    查找数据集中image_name字段的文件名等于指定字符串的样例
    
    Args:
        dataset: 加载的数据集
        target_filename: 要查找的文件名（不包含路径）
    """
    found_samples = []
    
    # 遍历数据集中的所有样本
    for sample in dataset:
        # 获取image_name字段并提取文件名
        image_name = sample.get('image_name', '')
        if isinstance(image_name, str):
            # 使用split('\\')分割路径并获取最后一部分（文件名）
            filename = image_name.split('/')[-1]
            
            # 如果文件名匹配目标字符串
            if filename == target_filename:
                found_samples.append(sample)
                print("=" * 50)
                print(f"找到匹配的样例 (image_name: {image_name}):")
                print("=" * 50)
                
                # 打印所有属性
                for key, value in sample.items():
                    print(f"{key}: {value}")
                print()
    
    # 如果没有找到匹配的样例
    if not found_samples:
        print(f"没有找到文件名为 '{target_filename}' 的样例")
    else:
        print(f"总共找到 {len(found_samples)} 个匹配的样例")

# 使用示例
if __name__ == "__main__":
    # 加载数据集（根据实际情况修改数据集名称和配置）
    dataset = load_dataset('SWE-bench/SWE-smith', split='train')
    
    # 指定要查找的image
    target_filename = "swesmith.x86_64.hips_1776_autograd.ac044f0d"
    
    # 执行查找
    find_sample_by_image_name(dataset, target_filename)