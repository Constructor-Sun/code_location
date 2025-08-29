from datasets import load_dataset
import json
import os
import re
from unidiff import PatchSet
import io

def is_python_or_config_file(file_path):
    """
    检查文件是否为Python文件或常见配置文件
    """
    # 常见配置文件扩展名
    config_extensions = {'.yml', '.yaml', '.json', '.toml', '.cfg', '.ini', '.txt', 
                        '.md', '.rst', '.cfg', '.conf', '.config'}
    
    # 获取文件扩展名（转换为小写）
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # 检查是否是Python文件或配置文件
    if file_extension == '.py':
        return True
    # elif file_extension in config_extensions:
    #     return True
    
    # 检查一些特殊的配置文件（无扩展名但常见）
    config_filenames = {'dockerfile', 'makefile', 'requirements', 'setup'}
    filename = os.path.basename(file_path).lower()
    if filename in config_filenames:
        return True
    
    return False

def is_code_file(file_path):
    """
    检查文件是否为代码文件(非Python和非配置文件)
    """
    code_extensions = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.java', '.go', 
                      '.cs', '.js', '.ts', '.rb', '.php', '.rs', '.swift', '.kt'}
    
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in code_extensions


def extract_edited_functions(patch_content):
    """
    从patch内容中提取被修改的函数信息
    返回两个列表：被修改的函数和新增的函数
    """
    edited_functions = set()
    added_functions = set()
    
    try:
        # 将patch内容转换为PatchSet对象
        patch = PatchSet(io.StringIO(patch_content))
        
        for patched_file in patch:
            file_path = patched_file.path
            if not file_path.endswith('.py'):  # 只处理Python文件
                continue
                
            # 跟踪当前的类名
            current_class = None
            # 用于跟踪删除和添加的函数
            deleted_funcs = set()
            added_funcs = set()
            
            for hunk in patched_file:
                for line in hunk:
                    line_content = line.value.strip()
                    
                    # 检测类定义
                    if line_content.startswith('class '):
                        current_class = line_content.split('class ')[1].split('(')[0].strip(':')
                    
                    # 检测函数定义
                    elif line_content.startswith('def '):
                        func_name = line_content.split('def ')[1].split('(')[0]
                        
                        if current_class:
                            full_name = f"{file_path}:{current_class}.{func_name}"
                        else:
                            full_name = f"{file_path}:{func_name}"
                            
                        if line.is_removed:
                            deleted_funcs.add(full_name)
                        elif line.is_added:
                            added_funcs.add(full_name)
            
            # 找出同时在删除和添加列表中的函数（这些是被修改的函数）
            modified = deleted_funcs.intersection(added_funcs)
            edited_functions.update(modified)
            
            # 找出只在添加列表中的函数（这些是新增的函数）
            only_added = added_funcs - deleted_funcs
            added_functions.update(only_added)
            
    except Exception as e:
        print(f"Warning: Failed to parse patch: {str(e)}")
        
    return ",".join(sorted(edited_functions)), ",".join(sorted(added_functions))


def main():
    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("SWE-bench/SWE-smith")
    
    # 存储只修改Python文件或配置文件的instance_id
    valid_instance_ids = []
    valid_image = []
    edit_functions_list = []
    added_functions_list = []
    edited_functions_list = []
    added_functions_list = []
    
    # 遍历数据集中的每个实例
    for instance in dataset["train"]:
        instance_id = instance.get("instance_id", "")
        image = instance.get("image_name", "").split("/")[-1]
        patch = instance.get("patch", "")
        
        if not patch or not instance_id:
            continue
        
        # 分割patch文本为单独的行
        lines = patch.split('\n')
        has_valid_files = False
        has_code_files = False
        current_file_path = None

        edited_funcs, added_funcs = extract_edited_functions(patch)
        edited_functions_list.append(edited_funcs if edited_funcs else "")
        added_functions_list.append(added_funcs if added_funcs else "")
        
        # 遍历每一行寻找diff行
        for line in lines:
            # 检查是否是diff行（包含文件路径）
            if line.startswith("diff --git"):
                # 提取文件路径部分
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                # 提取a/开头的文件路径（通常是第二个部分）
                a_file_path = parts[2]
                if a_file_path.startswith('a/'):
                    # 移除'a/'前缀
                    current_file_path = a_file_path[2:]
                    
                    # 检查是否是代码文件（非Python）
                    if is_code_file(current_file_path):
                        has_code_files = True
                        break
                    
                    # 检查是否是Python文件或配置文件
                    if is_python_or_config_file(current_file_path):
                        has_valid_files = True
        
        # 如果只包含Python/配置文件且不包含其他代码文件
        if has_valid_files and not has_code_files:
            valid_instance_ids.append(instance_id)
            valid_image.append(image)
    
    # 打印结果到控制台
    print(f"\n总计有 {len(valid_instance_ids)} 个样例只修改了Python文件或配置文件")
    print("前10个符合条件的instance_id:", valid_instance_ids[:10])
    
    # 创建结果字典
    result_data = {
        "total_count": len(valid_instance_ids),
        "instance_ids": valid_instance_ids,
        "image_name": valid_image,
        "edit_functions": edited_functions_list,
        "added_functions": added_functions_list,
        "description": "Instances that only modify Python files or configuration files, with modified and added functions"
    }
    
    # 保存到JSON文件
    output_file = "python_instances.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 {output_file} 文件中")

if __name__ == "__main__":
    main()