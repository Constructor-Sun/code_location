import os
import re
import unidiff
import argparse
import json
import pickle
import networkx as nx
from unidiff import PatchSet
from collections import defaultdict
from datasets import load_dataset


# def analyze_patch(patch_str):
#     def _analyze_hunk(full_name, hunk):
#         # Check hunk for changes
#         has_additions = any(line.line_type == '+' for line in hunk)
#         has_deletions = any(line.line_type == '-' for line in hunk)
        
#         if has_additions and has_deletions:
#             modified_functions.add(full_name)
#         elif has_additions and not has_deletions:
#             added_functions.add(full_name)

#     def _split_by_def(hunk):
#         result = []
#         func_names = []
#         current_section = []
#         prev_func_name = None
#         current_func_name = None
        
#         for line in hunk:
#             stripped_line = str(line).strip()
#             if stripped_line.startswith('+') or stripped_line.startswith('-'):
#                 stripped_line = stripped_line[1:].strip()
            
#             if stripped_line.startswith('def '):
#                 # get func name
#                 func_name = stripped_line.split('def ')[1].split('(')[0].strip()
#                 current_func_name = func_name
#                 # if not consistent and we have a current section
#                 if current_section and func_name != prev_func_name:
#                     result.append(current_section)
#                     func_names.append(prev_func_name)
#                     current_section = []
#                 # mark current name    
#                 prev_func_name = func_name

#             current_section.append(line)
        
#         # add final section
#         if current_section:
#             result.append(current_section)
#             func_names.append(current_func_name if current_func_name is not None else prev_func_name)

#         return result, func_names

#     def _find_func_header(hunk):
#         for line in hunk:
#             line_str = str(line)
#             func_match = re.search(r'^\s*[+-]?\s*def\s+(\w+)\s*\(', line_str)
#             if func_match:
#                 return func_match.group(1)
#         return None
    
#     def _find_class_header(hunk):
#         for line in hunk:
#             line_str = str(line)
#             class_match = re.search(r'^\s*[+-]?\s*class\s+(\w+)', line_str)
#             if class_match:
#                 return class_match.group(1)
#         return None

#     try:
#         patch = PatchSet.from_string(patch_str)
#     except unidiff.UnidiffParseError:
#         return "", ""
    
#     modified_functions = set()
#     added_functions = set()
#     if_py = True
    
#     for patched_file in patch:
#         file_path = patched_file.path
#         if_py = file_path.split(".")[-1] == "py"
#         if not if_py:
#             break
#         for hunk in patched_file:
#             # print("hunk: ", hunk)
#             header = hunk.section_header
#             # print("header: ", header)
#             if header is None:
#                 print("no header is found in: \n", hunk)
#                 continue
#             # Extract function name from context line
#             # Typically, context line is like: @@ -lineno,count +lineno,count @@ function_signature
#             match = re.match(r'def\s+(\w+)\s*\(', header)
#             if match:
#                 func_name = match.group(1)
#                 full_name = f"{file_path}:{func_name}"
#                 # print("func_name: ", func_name)
#                 _analyze_hunk(full_name, hunk)
#             else:
#                 match = re.search(r'class\s+(\w+)', header)
#                 if match:
#                     class_name = match.group(1)
#                     # print("class_name: ", class_name)
#                     sub_hunks, func_names = _split_by_def(hunk)
#                     for sub_hunk, func_name in zip(sub_hunks, func_names):
#                         if func_name == None:
#                             continue
#                         full_name = f"{file_path}:{class_name}:{func_name}"
#                         _analyze_hunk(full_name, sub_hunk)
#                 else:
#                     func_name = _find_func_header(hunk)
#                     if func_name:
#                         full_name = f"{file_path}:{func_name}"
#                         _analyze_hunk(full_name, hunk)
#                         # continue
#                     class_name = _find_class_header(hunk)
#                     if class_name:
#                         sub_hunks, func_names = _split_by_def(hunk)
#                         for sub_hunk, func_name in zip(sub_hunks, func_names):
#                             if func_name == None:
#                                 continue
#                             full_name = f"{file_path}:{class_name}:{func_name}"
#                             _analyze_hunk(full_name, sub_hunk)
#     return if_py, list(modified_functions), list(added_functions)

# def check_existence(image_path, modified_file):
#     with open(image_path, 'rb') as file:
#         data = pickle.load(file)
    
#     nodes = list(data.nodes(data=False))
#     # 创建节点到索引的映射字典
#     node_to_index = {node: idx for idx, node in enumerate(nodes)}
    
#     result = [0] * len(nodes)
    
#     for path in modified_file:
#         if path not in node_to_index:
#             return None
#         position = node_to_index[path]
#         result[position] = 1
    
#     return result

def analyze_patch(code_map_path, patch):
    with open(code_map_path, 'rb') as file:
        code_map = pickle.load(file)

    patch_set = unidiff.PatchSet(patch)
    modified_functions = set()
    added_functions = set()
    modified_node_indices = []
    added_node_indices = []
    
    # 首先收集所有文件的修改信息
    file_modifications = {}
    if_not_py = False
    for patched_file in patch_set:
        file_path = patched_file.path
        if_not_py = file_path.split(".")[-1] != "py"
        deleted_lines = set()
        added_lines = set()
        
        for hunk in patched_file:
            for line in hunk:
                if line.is_removed:
                    deleted_lines.add(line.source_line_no)
                elif line.is_added:
                    added_lines.add(line.target_line_no)
        
        if deleted_lines or added_lines:
            file_modifications[file_path] = {
                'deleted_lines': deleted_lines,
                'added_lines': added_lines
            }
    
    # 如果没有修改，直接返回空结果
    if not file_modifications or if_not_py:
        return None
    
    # 遍历所有节点，检查哪些函数受到影响
    node_num = len(code_map.nodes())
    for idx, node in enumerate(code_map.nodes(data=True)):
        node_name, attrs = node
        if attrs.get('type') != 'function':
            continue
            
        # 提取函数所在的文件路径
        file_path = node_name.split(':')[0]
        
        # 检查这个文件是否有修改
        if file_path not in file_modifications:
            continue
            
        mod_info = file_modifications[file_path]
        start_line = attrs.get('start_line')
        end_line = attrs.get('end_line')
        
        if start_line is None or end_line is None:
            continue
        
        # 检查是否有删除行在函数范围内
        has_deleted = any(start_line <= line <= end_line for line in mod_info['deleted_lines'])
        # 检查是否有新增行在函数范围内
        has_added = any(start_line <= line <= end_line for line in mod_info['added_lines'])
        
        if has_deleted:
            # 有删除行，说明是修改
            modified_functions.add(node_name)
            modified_node_indices.append(idx)
        elif has_added:
            # 只有新增行，说明是新增
            added_functions.add(node_name)
            added_node_indices.append(idx)
    
    return {
        'image_node_num': node_num,
        'modified_list': list(modified_functions),
        'added_list': list(added_functions),
        'modified_node_indices': modified_node_indices,
        'added_node_indices': added_node_indices
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SWE-bench/SWE-smith")
    parser.add_argument("--save_path", type=str, default="python_instances.json")
    parser.add_argument("--image_dir", type=str, default="train_index/SWE-smith/graph_index_v1.0")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    valid_instance_ids = []
    valid_image = []
    valid_image_node_num = []
    valid_problems = []
    edit_functions = []
    added_functions = []
    edit_functions_ord = []
    added_functions_ord = []

    cur = 0
    for instance in dataset["train"]:
        instance_id = instance.get("instance_id", "")
        image = instance.get("image_name", "").split("/")[-1]
        problem = instance.get("problem_statement")
        if problem == "":
            continue
        patch = instance.get("patch", "")

        result_dic = analyze_patch(os.path.join(args.image_dir, image + '.pkl'), patch)
        if result_dic:
            valid_instance_ids.append(instance_id)
            valid_image.append(image)
            valid_problems.append(problem)
            valid_image_node_num.append(result_dic['image_node_num'])
            edit_functions.append(result_dic['modified_list'])
            added_functions.append(result_dic['added_list'])
            edit_functions_ord.append(result_dic['modified_node_indices'])
            added_functions_ord.append(result_dic['added_node_indices'])
            cur += 1

        # if check_existence(os.path.join(args.image_dir, image + ".pkl"), modified_list):
        #     valid_instance_ids.append(instance_id)
        #     valid_image.append(image)
        #     valid_problems.append(problem)
        #     edit_functions.append(modified_list)
        #     added_functions.append(added_list)
        #     cur += 1
        # else:
        #     print("instance_id: ", instance_id)
        #     print("image: ", image)
        #     print("modified_list: ", modified_list)
        #     print("patch: ", patch)
        #     exit()

        if cur % 100 == 0:
            print("now cur: ", cur)

    result_data = {
        "description": "Instances that only modify Python files or configuration files, with modified and added functions",
        "total_count": len(valid_instance_ids),
        "instance_ids": valid_instance_ids,
        "image_name": valid_image,
        "image_node_num": valid_image_node_num,
        "edit_functions": edit_functions,
        "added_functions": added_functions,
        "edit_functions_ord": edit_functions_ord,
        "added_functions_ord": added_functions_ord,
        "problem_statement": valid_problems
    }
    # print(f"modified_str:{modified_str} \nadded_str:{added_str}")
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"file saved successful")

if __name__ == '__main__':
    main()

# patch = """
# diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
# --- a/django/contrib/admin/options.py
# +++ b/django/contrib/admin/options.py
# @@ -2037,10 +2037,13 @@ def __init__(self, parent_model, admin_site):
#          self.opts = self.model._meta
#          self.has_registered_model = admin_site.is_registered(self.model)
#          super().__init__()
# +        if self.verbose_name_plural is None:
# +            if self.verbose_name is None:
# +                self.verbose_name_plural = self.model._meta.verbose_name_plural
# +            else:
# +                self.verbose_name_plural = format_lazy('{}s', self.verbose_name)
#          if self.verbose_name is None:
#              self.verbose_name = self.model._meta.verbose_name
# -        if self.verbose_name_plural is None:
# -            self.verbose_name_plural = self.model._meta.verbose_name_plural
 
#      @property
#      def media(self):
#     """
