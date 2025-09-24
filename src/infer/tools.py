import os
import ast
import json
import subprocess
import difflib
from typing import Dict, List, Union
from pydantic import BaseModel, Field

MAX_NUM = 20

class GetFunctionsInput(BaseModel):
    func_paths: str = Field(
        ...,
        description="""
            Reveal targeted function contents.
            Input: a JSON string containing a list of function paths. The key is "func_paths".
            Note: Paths should reference specific functions/methods, NOT just files or directories.
        """,
        examples=[
            '{"func_paths": ["sklearn/base.py/DensityMixin/score", "build_tools/github/vendor.py/main"]}'
            ]
    )

class GetClassInput(BaseModel):
    func_paths: str = Field(
        ...,
        description="""
            Reveal targeted class contents.
            Input: a JSON string containing a valid class paths. The key is "class_path".
            Note: Paths should reference specific class, NOT just files or directories.
        """,
        examples=[
            '{"class_path": "sklearn/base.py/DensityMixin"}'
            ]
    )

class GetFileInput(BaseModel):
    file_path: str = Field(
        ...,
        description="""
            Reveal all functions within this file.
            Input: as a JSON string containing a valid file path. The key is "file_path".
            Note: Should not include specific function name.
        """,
        examples=[
            '{"file_path": "build_tools/github/vendor.py"}'
            ]
    )

class ListFunctionDirectoryInput(BaseModel):
    file_path: str = Field(
        ...,
        description="""
            Reveal all functions with this file. But if too much functions here, it will only list modules' names.
            Input: a JSON string containing a valid file path or directory. The key is "file_path".
            Note: "file_path" should be a path for a specific .py file.
        """,
        examples=[
            '{"file_path": "sklearn/pipeline.py"}'
            ]
    )

class GetCallGraphInput(BaseModel):
    params: str = Field(
        ...,
        description="""
            Reveal the call graph of target function within its directory.
            Input: a JSON string containing a valid file path. The key is "target_function".
            Note: "target_function" should be a path for a specific function.
        """,
        examples=[
            '{"target_function": "examples/intermediate/coupled_cluster.py/get_CC_operators"}', 
            '{"target_function": "src/_pytest/cacheprovider.py/cache"}'
        ]
    )

def _split_path(path):
    # find file first. Always assume there is a .py file in the path.
    py_index = path.find('.py')
    if py_index == -1:
        return path, ""
    
    # dir + file
    first_part = path[:py_index + 3]  # include ".py"
    
    # (module) + function
    second_part = path[py_index + 3:]
    if second_part.startswith('/'):
        second_part = second_part[1:]

    # code2flow requires '.' to connect module and function
    second_part = second_part.replace('/', '.')
    return first_part, second_part

def find_similar_path(corpus_dict: Dict[str, str], path: str) -> str:
    similar_paths = difflib.get_close_matches(path, corpus_dict.keys(), n=3, cutoff=0.7)
    if similar_paths:
        similar_paths_str = " or ".join(f"'{p}'" for p in similar_paths)
        suggestions = f"{path} not found! Do you mean {similar_paths_str}? Ensure the path points to a specific function."
    else:
        suggestions = f"{path} not found! Ensure the path points to a specific function, e.g., 'dir/file.py/Module/function'."
    return suggestions

def _get_imports(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        import_strings = []
        
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                import_str = f"import {', '.join(modules)}"
                import_strings.append(import_str)
            elif isinstance(node, ast.ImportFrom):
                # 处理 from ... import 语句（包括相对导入）
                names = [alias.name for alias in node.names]
                
                # 构建模块名：处理相对导入级别
                level = '.' * (node.level or 0)  # 相对导入的点号
                module_part = node.module or ''  # 可能为None（如 from . import something）
                
                if level and module_part:
                    module_str = f"{level}{module_part}"
                elif level:
                    module_str = level
                else:
                    module_str = module_part
                
                # 构建完整的from import语句
                if module_str:
                    from_str = f"from {module_str} import {', '.join(names)}"
                else:
                    # 处理特殊情况：from . import something
                    from_str = f"from {level} import {', '.join(names)}"
                
                import_strings.append(from_str)
        
        return 'This file has global import relations: \n' + '\n'.join(import_strings)
    except SyntaxError as e:
        raise ValueError(f"File has syntax error: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not Found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Runtime error: {e}")

def check_module(corpus_dict: Dict[str, str], path: str) -> List[str]:
    dir_file, module_function = _split_path(path)
    module_func_split = module_function.split(".")
    # module + func is not allowed here
    if len(module_func_split) == 2:
        return []
    else: # len == 1: false func, true module, or false module
        module = module_func_split[0]
        module_path = os.path.join(dir_file, module)
        module_dict = {}
        for keys in corpus_dict:
            if keys.startswith(module_path):
                module_dict[keys] = corpus_dict[keys]
        if len(module_dict) <= 3:
            return list(module_dict.values())
        else:
            return list(module_dict.keys())

def get_corpus(test_dir: str, dataset: str, instance_id: str) -> Dict[str, str]:
    if '-function_' in instance_id:
        corpus_path = os.path.join(test_dir, instance_id, "corpus.jsonl")
    else:
        corpus_path = os.path.join(test_dir, dataset + '-function_' + instance_id, "corpus.jsonl")
    corpus_dict = {}
    # convert corpus to python dicts. key: func_path, value: func_content
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            _id = data.get("_id")
            text = data.get("text")
            
            if _id is not None and text is not None:
                corpus_dict[_id] = text
    return corpus_dict

def get_functions(corpus_dict: Dict[str, str], func_paths: str) -> List[str]:
    try:
        # Parse the JSON string, expecting an object with 'func_paths' key
        input_data = json.loads(func_paths)
        if not isinstance(input_data, dict) or "func_paths" not in input_data:
            return "Input must be a valid JSON sting with a 'func_paths' key"
        paths_list = input_data["func_paths"]
        if not isinstance(paths_list, list):
            return "'func_paths' value must be a list"
    except json.JSONDecodeError as _:
        return "JSON failed to load. Please check the format."

    result = []
    for path in paths_list:
        if not isinstance(path, str):
            result.append(f"Invalid path: {path}. Paths must be strings.")
            continue
        # If path is a valid function path, directly append its' content to the result.
        if path in corpus_dict:
            result.append(corpus_dict[path])
        else:
            # If path is a valid class path.
            suggestions = check_module(corpus_dict, path)
            if suggestions != []:
                # suggestions = "It seems that you entered a class name, this class contain the following functions: " + ", ".join(suggestions)
                suggestions = "It seems that you entered a class name, try use 'get_class' tool to reveal this class"
                result.append(suggestions)
            else:
                # If path is neither a valid function nor the module, return similar function paths.
                suggestions = find_similar_path(corpus_dict, path)
                result.append(suggestions)
    return result

def get_class(corpus_dict: Dict[str, str], class_path: str) -> List[str]:
    try:
        # Parse the JSON string, expecting an object with 'class_path' key
        input_data = json.loads(class_path)
        if not isinstance(input_data, dict) or "class_path" not in input_data:
            return "Input must be a valid JSON sting with a 'class_path' key"
        path = input_data["class_path"]
        if not isinstance(path, str):
            return "'class_path' value must be a str"
    except json.JSONDecodeError as _:
        return "JSON failed to load. Please check the format."
    
    result = []
    if path.endswith(".py"):
        return f"It seems that {path} is a file! Try use get_file tool to reveal its content!"
    if path in corpus_dict:
        return f"It seems that {path} is a valid function! Try use get_functions tool to reveal its content!"
    for key in corpus_dict.keys():
        if key.startswith(path):
            result.append(corpus_dict[key])
    if result == []:
        return "Not a valid class name. Try enter a valid class name."
    else:
        return result

def get_file(corpus_dict: Dict[str, str], instance_path: str, file_path: str) -> List[str]:
    try:
        # Parse the JSON string, expecting an object with 'file_path' key
        input_data = json.loads(file_path)
        if not isinstance(input_data, dict) or "file_path" not in input_data:
            return "Input must be a valid JSON sting with a 'file_path' key"
        path = input_data["file_path"]
        if not isinstance(path, str):
            return "'file_path' value must be a string."
    except json.JSONDecodeError as _:
        return "JSON failed to load. Please check the format."

    result = {}
    for func_path in corpus_dict.keys():
        if func_path.startswith(path):
            result[func_path] = corpus_dict[func_path]
    if not result:
        return f"No functions found for file path '{path}'. Please check if the file path exists in the corpus."
    
    try:
        import_result = _get_imports(os.path.join(instance_path, path))
    except Exception as e:
        return f"{e}"
    if len(result) > MAX_NUM:
        return list_function_directory(corpus_dict, file_path) + '\n' + import_result
        # return "Too many functions in this file. Here this tool will give you a list of function names contained in this file. If you wanna check more details about a few of them, please call 'get_functions'. " + str(result.keys())
    return ', '.join(result.values()) + '\n' + import_result

def list_function_directory(corpus_dict: Dict[str, str], file_path: str) -> List[str]:
    try:
        input_data = json.loads(file_path)
        if not isinstance(input_data, dict) or "file_path" not in input_data:
            return "Input must be a valid JSON sting with a 'file_path' key"
        path = input_data["file_path"]
        if not isinstance(path, str):
            return "'file_path' value must be a string."
    except json.JSONDecodeError as _:
        return "JSON failed to load. Please check the format."

    result = [func_path for func_path in corpus_dict.keys()
              if func_path.startswith(path)]
    
    # TODO: Add a *.py here to avoid checking all functions within a dir.
    # TODO: However, if it is a specific .py file, list all functions here 
    # TODO: to avoid some critical functions never known to LLM.
    if len(result) > MAX_NUM:
        class_paths_set = set()
        module_level_function_paths_set = set()
        for func_path in result:
            file_part, module_func_part = _split_path(func_path)
            
            if module_func_part:
                parts = module_func_part.split('.')
                if len(parts) > 1:
                    module_path = os.path.join(file_part, parts[0])
                    class_paths_set.add(module_path)
                else:
                    function_path = os.path.join(file_part, parts[0])
                    module_level_function_paths_set.add(function_path)
        
        class_paths = list(class_paths_set)
        
        return (
            f"Too many functions in {path}. Instead, we have a brief intro. \n"
            f"This file contains modules: {str(class_paths)} \n"
            f"This file also contains module-level functions: {str(module_level_function_paths_set)}"
            )

    if not result:
        result.append(f"No functions found for file path '{path}'.")
    return result

def transform_graph(call_graph):
    graph_data = call_graph["graph"]
    
    # node uid -> name
    nodes = graph_data["nodes"]
    node_name_map = {}
    for node_id, node_info in nodes.items():
        node_name_map[node_id] = node_info["name"]
    
    # edges: change uid to name
    transformed_edges = []
    for edge in graph_data["edges"]:
        transformed_edge = {
            "source": node_name_map[edge["source"]],
            "target": node_name_map[edge["target"]]
        }
        transformed_edges.append(transformed_edge)
    
    # create a new json
    result = {
        "directed": graph_data["directed"],
        "edges": transformed_edges
    }
    
    return result

# def _get_call_graph(instance_path: str, scopes: List[str], target_function: str) -> Dict[str, str]:
#     targets = [os.path.join(instance_path, scope) for scope in scopes]
#     modified_targets = []
#     for target in targets:
#         if target.endswith('.py'):
#             if '/' in target:
#                 dir_part = target.rsplit('/', 1)[0]
#                 modified_targets.append(f"{dir_part}/*.py")
#             else:
#                 modified_targets.append("*.py")
#         else:
#             target = target if target.endswith('/') else target + '/'
#             modified_targets.append(target + '*.py')
#     modified_targets = " ".join(modified_targets)
#     code2flow_cmd = f"code2flow {modified_targets} --target-function {target_function} --upstream-depth=1 --downstream-depth=5 --output tmp/call_graph.json"
#     print("code2flow_cmd: ", code2flow_cmd)

#     try:
#         # TODO: sometimes it cannot find the existing node, strange
#         _ = subprocess.run(
#             code2flow_cmd,
#             shell=True,
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         with open("tmp/call_graph.json", "r") as file:
#             call_graph = json.load(file)
#             simple_call_graph = transform_graph(call_graph)
#             with open('tmp/transformed_call_graph.json', 'w', encoding='utf-8') as f:
#                 json.dump(simple_call_graph, f, indent=2, ensure_ascii=False)
#             return simple_call_graph
#     except subprocess.CalledProcessError as e:
#         # print(f"Error running code2flow: {e}")
#         # print(f"stderr: {e.stderr}")
#         # TODO: fix it using: 1. use list_function_directory; 2. do not limit target_function
#         try:
#             fixed_cmd = f"code2flow {modified_targets} --output tmp/call_graph.json"
#             _ = subprocess.run(
#                 fixed_cmd,
#                 shell=True,
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )
#             with open("tmp/call_graph.json", "r") as file:
#                 call_graph = json.load(file)
#                 simple_call_graph = transform_graph(call_graph)
#                 with open('tmp/transformed_call_graph.json', 'w', encoding='utf-8') as f:
#                     json.dump(simple_call_graph, f, indent=2, ensure_ascii=False)
#                 return f"""Cannot find {target_function} in {modified_targets}.
#                             Instead, this tool reveal all call graphs in target_function:
#                             """ + str(simple_call_graph)
#         except subprocess.CalledProcessError as e2:
#             # print(f"Error running code2flow: {e}")
#             # print(f"stderr: {e.stderr}")
#             return "Failed to geneate call graph. Please check if file or function exists."
#     except FileNotFoundError as e:
#         raise RuntimeError(
#             "code2flow not found. Please install with: pip install code2flow"
#             ) from e
    
# def get_call_graph(instance_path: str, params: str) -> Dict[str, str]:
#     try:
#         params_dict = json.loads(params)
#         return _get_call_graph(
#             instance_path,
#             params_dict['scopes'],
#             params_dict['target_function']
#         )
#     except json.JSONDecodeError:
#         return "Error: Invalid JSON format. Please provide valid JSON string."
#     except KeyError as e:
#         return f"Error: Missing required parameter: {e}"
#     except Exception as e:
#         return f"Error: {str(e)}"

def _get_call_graph(corpus_dist: Dict[str, str], instance_path: str, target_function: str) -> str:
    def _check_valid(path):
        if path in corpus_dist:
            return ""
        else:
            return find_similar_path(corpus_dist, path)
    
    def _load_json_call_graph():
        with open("tmp/call_graph.json", "r") as file:
            call_graph = json.load(file)
            simple_call_graph = transform_graph(call_graph)
            with open('tmp/transformed_call_graph.json', 'w', encoding='utf-8') as f:
                json.dump(simple_call_graph, f, indent=2, ensure_ascii=False)
            return simple_call_graph

    suggestions = _check_valid(target_function)
    if suggestions != "":
        return suggestions
    else:
        dir_file, module_function = _split_path(target_function)
        scope = os.path.join(instance_path, dir_file)
        scope_dir = os.path.dirname(scope)

        scope_strategies = [
            scope_dir,
            os.path.join(scope_dir, "*.py"),
            scope
        ]
        scope_strategies = list(dict.fromkeys([s for s in scope_strategies if s]))

        for i, scope in enumerate(scope_strategies):
            try:
                code2flow_cmd = f"code2flow {scope} --target-function {module_function} --upstream-depth=1 --downstream-depth=5 --output tmp/call_graph.json"
                _ = subprocess.run(
                    code2flow_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True
                )
                simple_call_graph = _load_json_call_graph()
                edge_count = len(simple_call_graph['edges'])
                if edge_count <= 75 and i < len(scope_strategies) - 1:
                    return f"Call graph of {module_function} in the scope {scope} is: \n" + str(simple_call_graph)
                else:
                    continue
            except subprocess.CalledProcessError as e:
                print(f"Finding {module_function} in {scope} has runtime error: {type(e)}")
                continue
            except subprocess.TimeoutExpired:
                print(f"Finding {module_function} in {scope} cost too much time. This is not expected.")
                continue
            except FileNotFoundError as e:
                raise RuntimeError(
                    "code2flow not found. Please install with: pip install code2flow"
                    ) from e
            except Exception as e:
                # raise RuntimeError(
                #     f"Finding {module_function} in {scope} has unexpected errors: {e}. Please check the tool."
                #     ) from e
                print(f"Finding {module_function} in {scope} has unexpected errors: {e}. Please check the tool.")
                continue
    return f"get_call_graph cannot analyze this function due to its inner flaws. This tool then use 'get_functions' to reveal the content of {target_function}: " + str(get_functions(corpus_dist, '{"func_paths": ["' + target_function + '"]}'))
    
def get_call_graph(corpus_dist: Dict[str, str], instance_path: str, target_function: str) -> str:
    try:
        print("target_function: ", target_function)
        target = json.loads(target_function)["target_function"]
        return _get_call_graph(
            corpus_dist,
            instance_path,
            target
        )
    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Please provide valid JSON string."
    except KeyError as e:
        return f"Error: Missing required parameter: {e}"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    dataset = "swe-bench-lite"
    instance_id = "django__django-13158"
    corpus = get_corpus("datasets", dataset, instance_id)
    instance_path = os.path.join(dataset, instance_id)
    test_funcs = False
    test_class = False
    test_file = False
    test_call_graph = True
    test_lists = False
    if test_funcs:
        example_func_path = '{"func_paths": ["sympy/matrices/expressions/matexpr.py/Identity/_entry"]}'
        funcs = get_functions(corpus, example_func_path)
        print("funcs: ", funcs)
    if test_class:
        example_func_path = '{"class_path": "sympy/matrices/expressions/matexpr.py/Identity"}'
        funcs = get_class(corpus, example_func_path)
        print("class: ", funcs)
    if test_file:
        file_path = '{"file_path": "test_path_error.py"}'
        funcs = get_file(corpus, instance_path, file_path)
        print("file: ", funcs)
    if test_call_graph:
        params = '{"target_function": "django/forms/models.py/ModelMultipleChoiceField/to_python"}'
        result = get_call_graph(corpus, instance_path, params)
        print("result: ", result)
    if test_lists:
        file_path = '{"file_path": "sympy/matrices/expressions/matexpr.py"}'
        funcs = list_function_directory(corpus, file_path)
        print("funcs: ", funcs)
    # if test_lists:
    #     keywords = '{"keywords": ["repr"]}'
    #     result = get_keyword_search(corpus, keywords)
    #     print("result: ", result)

if __name__ == "__main__":
    main()