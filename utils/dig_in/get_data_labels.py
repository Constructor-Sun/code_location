import re
import unidiff
import argparse
import json
from unidiff import PatchSet
from collections import defaultdict
from datasets import load_dataset


def analyze_patch(patch_str):
    def _analyze_hunk(full_name, hunk):
        # Check hunk for changes
        has_additions = any(line.line_type == '+' for line in hunk)
        has_deletions = any(line.line_type == '-' for line in hunk)
        
        if has_additions and has_deletions:
            modified_functions.add(full_name)
        elif has_additions and not has_deletions:
            added_functions.add(full_name)

    def _split_by_def(hunk):
        result = []
        func_names = []
        current_section = []
        prev_func_name = None
        current_func_name = None
        
        for line in hunk:
            stripped_line = str(line).strip()
            if stripped_line.startswith('+') or stripped_line.startswith('-'):
                stripped_line = stripped_line[1:].strip()
            
            if stripped_line.startswith('def '):
                # get func name
                func_name = stripped_line.split('def ')[1].split('(')[0].strip()
                current_func_name = func_name
                # if not consistent and we have a current section
                if current_section and func_name != prev_func_name:
                    result.append(current_section)
                    func_names.append(prev_func_name)
                    current_section = []
                # mark current name    
                prev_func_name = func_name

            current_section.append(line)
        
        # add final section
        if current_section:
            result.append(current_section)
            func_names.append(current_func_name if current_func_name is not None else prev_func_name)

        return result, func_names

    def _find_func_header(hunk):
        for line in hunk:
            line_str = str(line)
            func_match = re.search(r'^\s*[+-]?\s*def\s+(\w+)\s*\(', line_str)
            if func_match:
                return func_match.group(1)
        return None
    
    def _find_class_header(hunk):
        for line in hunk:
            line_str = str(line)
            class_match = re.search(r'^\s*[+-]?\s*class\s+(\w+)', line_str)
            if class_match:
                return class_match.group(1)
        return None

    try:
        patch = PatchSet.from_string(patch_str)
    except unidiff.UnidiffParseError:
        return "", ""
    
    modified_functions = set()
    added_functions = set()
    if_py = True
    counter = 0
    
    for patched_file in patch:
        file_path = patched_file.path
        # print("file_path: ", file_path)
        if_py = file_path.split(".")[-1] == "py"
        if not if_py:
            break
        for hunk in patched_file:
            # print("hunk: ", hunk)
            header = hunk.section_header
            # print("header: ", header)
            if header is None:
                print("no header is found in: \n", hunk)
                continue
            # Extract function name from context line
            # Typically, context line is like: @@ -lineno,count +lineno,count @@ function_signature
            match = re.match(r'def\s+(\w+)\s*\(', header)
            if match:
                func_name = match.group(1)
                full_name = f"{file_path}:{func_name}"
                # print("func_name: ", func_name)
                _analyze_hunk(full_name, hunk)
            else:
                match = re.search(r'class\s+(\w+)', header)
                if match:
                    class_name = match.group(1)
                    # print("class_name: ", class_name)
                    sub_hunks, func_names = _split_by_def(hunk)
                    for sub_hunk, func_name in zip(sub_hunks, func_names):
                        if func_name == None:
                            continue
                        full_name = f"{file_path}:{class_name}:{func_name}"
                        _analyze_hunk(full_name, sub_hunk)
                else:
                    func_name = _find_func_header(hunk)
                    if func_name:
                        full_name = f"{file_path}:{func_name}"
                        _analyze_hunk(full_name, hunk)
                        # continue
                    class_name = _find_class_header(hunk)
                    if class_name:
                        sub_hunks, func_names = _split_by_def(hunk)
                        for sub_hunk, func_name in zip(sub_hunks, func_names):
                            if func_name == None:
                                continue
                            full_name = f"{file_path}:{class_name}:{func_name}"
                            _analyze_hunk(full_name, sub_hunk)
                        # continue
                    # counter += 1
                    # if counter <= 15:
                    #     continue
                    # print("still not solved header: ", header)
                    # print("still not solved hunk: ", hunk)
                    # raise ValueError("Your custom error message here.")
            
    modified_str = ','.join(modified_functions)
    added_str = ','.join(added_functions)
    return if_py, modified_str, added_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SWE-bench/SWE-smith")
    parser.add_argument("--save_path", type=str, default="python_instances.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    valid_instance_ids = []
    valid_image = []
    valid_problems = []
    edit_functions = []
    added_functions = []

    for instance in dataset["train"]:
        instance_id = instance.get("instance_id", "")
        image = instance.get("image_name", "").split("/")[-1]
        problem = instance.get("problem_statement")
        patch = instance.get("patch", "")

        if_py, modified_str, added_str = analyze_patch(patch)
        if not if_py or modified_str == "":
            continue

        valid_instance_ids.append(instance_id)
        valid_image.append(image)
        valid_problems.append(problem)
        edit_functions.append(modified_str)
        added_functions.append(added_str)

    result_data = {
        "description": "Instances that only modify Python files or configuration files, with modified and added functions",
        "total_count": len(valid_instance_ids),
        "instance_ids": valid_instance_ids,
        "image_name": valid_image,
        "edit_functions": edit_functions,
        "added_functions": added_functions,
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
