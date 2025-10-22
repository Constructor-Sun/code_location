import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import re
import gc
import json
import signal
import argparse
import random
import time
import numpy as np
import torch
from functools import partial
from langchain_community.llms import VLLM
from openai import OpenAI

from retrieval import retrieve_batch
from prompt import *
from tools import *
from tools import _split_path

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def load_inference_model(model_name):
    llm = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    return llm

def execute_xai(llm, corpus, target, query, preds, args):
    def _inner_loop(tools, messages):
        response = messages[-1]
        while response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                if function_name not in tools_map:
                    messages.append({
                            "role": "tool",
                            "content": json.dumps({"error": f"Function {function_name} not found"}),
                            "tool_call_id": tool_call.id
                        })
                    continue
                function_args = json.loads(tool_call.function.arguments)
                result = tools_map[function_name](**function_args)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tool_call.id
                    })
            response = llm.chat.completions.create(
                model=args.inference_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                top_p=1.0,
                seed=42
            ).choices[0].message
            messages.append(response)
        return 

    tool_func = {
        "type": "function",
        "function": {
            "name": "get_functions",
            "description": "Retrieve function code from corpus dictionary for one or multiple function paths",
            "parameters": GetFunctionsInput.model_json_schema(),
        }
    }
    tool_class = {
        "type": "function",
        "function": {
            "name": "get_class",
            "description": "Retrieve class content from corpus dictionary",
            "parameters": GetClassInput.model_json_schema(),
        }
    }
    tool_file = {
        "type": "function",
        "function": {
            "name": "get_file",
            "description": "Retrieve all functions from a specific file path in the corpus dictionary.",
            "parameters": GetFileInput.model_json_schema(),
        }
    }
    tool_list = {
        "type": "function",
        "function": {
            "name": "list_function_directory",
            "description": "Retrieve all functions from a specific file path in the corpus dictionary.",
            "parameters": ListFunctionDirectoryInput.model_json_schema(),
        }
    }
    tool_graph = {
        "type": "function",
        "function": {
            "name": "get_call_graph",
            "description": "Retrieve all functions from a specific file path in the corpus dictionary.",
            "parameters": GetCallGraphInput.model_json_schema(),
        }
    }

    tools_1 = [tool_func, tool_file, tool_graph, tool_list, tool_class]
    tools_2 = [tool_func, tool_class, tool_file]

    tools_map = {
        "get_functions": partial(get_functions, corpus),
        "get_class": partial(get_class, corpus),
        "get_file": partial(get_file, corpus, os.path.join(args.dataset, target)),
        "list_function_directory": partial(list_function_directory, corpus),
        "get_call_graph": partial(get_call_graph, corpus, os.path.join(args.dataset, target))
    }

    messages_1 = [{"role": "user", "content": FIRST_PROMPT.format(query=query, preds=str(preds))}]
    response = llm.chat.completions.create(
        model=args.inference_model,
        messages=messages_1,
        tools=tools_1,
        tool_choice="auto",
        temperature=0,
        top_p=1.0,
        seed=42
    )
    messages_1.append(response.choices[0].message)
    _inner_loop(tools_1, messages_1)
    # print("final response: ", messages_1[-1].content)

    try:
        json_matches = re.findall(r'\{[^{}]*\}', messages_1[-1].content, re.DOTALL)
        json_str = json_matches[-1]
        response_final_1 = json.loads(json_str)
        candidate_methods = response_final_1["methods_tobe_modified"]
        count = sum(1 for method in candidate_methods if ".py/" in method)
        if count < 7:
            raise ValueError(f"Only {count} valid path at the first stage, use the initial candidates.")
    except:
        candidate_methods = preds
    
    class_list = []
    file_list = []
    module_func_list = []
    for path in candidate_methods:
        left, right = _split_path(path)
        file_list.append(left)
        if "." in right:
            class_list.append(right.split(".")[0])
        if "." not in right:
            if not right.startswith('_') and any(char.isupper() for char in right):
                class_list.append(right)
            else:
                module_func_list.append(right)
    class_num = len(set(class_list))
    file_num = len(set(file_list))
    module_func_num = len(set(module_func_list))
    print("class_num: ", class_num)
    print("file_num: ", file_num)
    print("module_func_num: ", module_func_num)
    if file_num == 1 or class_num <= 2 and file_num <= 2:
        print("ori: ", json.dumps(preds, indent=4))
        print("init answer: \n", json.dumps(candidate_methods, indent=4))
        return candidate_methods
    
    # Step 2
    partition = partition_methods(candidate_methods)
    total = []
    for _, methods in enumerate(partition):
        files = set([_split_path(path)[0] for path in methods])
        files = ", ".join(list(files))
        formatted_second_prompt = SECOND_PROMPT.format(query=query, candidates=methods, scope=files)
        messages_2 = [{"role": "user", "content": formatted_second_prompt}]
        response = llm.chat.completions.create(
            model="x-ai/grok-code-fast-1",
            messages=messages_2,
            tools=tools_2,
            tool_choice="auto",
            temperature=0,
            top_p=1.0,
            seed=42,
        )
        if response == None:
            print("response none here")
        messages_2.append(response.choices[0].message)
        _inner_loop(tools_2, messages_2)
        json_matches = re.findall(r'\{[^{}]*\}', messages_2[-1].content, re.DOTALL)
        if json_matches:
            json_str = json_matches[-1]
            print("json_str:\n", json_str)
            response_final_2 = json.loads(json_str)
            current = response_final_2["methods_tobe_modified"]
            total.extend(current)
        else:
            intermediate_steps = messages_2[:-1]
            valid = False
            for step in reversed(intermediate_steps):
                if (step and len(step) >= 1 and hasattr(step[0], 'log') and step[0].log):
                    json_matches = re.findall(r'\{[^{}]*\}', step[0].log, re.DOTALL)
                    if json_matches:
                        json_str = json_matches[-1]
                        response_final_2 = json.loads(json_str)
                        current = response_final_2["methods_tobe_modified"]
                        total.extend(current)
                        valid = True
                        break
            if not valid:
                exception = Exception()
                exception.response = messages_2[:-1]
                raise exception
                    
    print("original preds: ", json.dumps(preds, indent=4))
    print("init answer: \n", json.dumps(candidate_methods, indent=4))
    for i, group in enumerate(partition):
        print(f"\n第{i+1}组 ({len(group)}个方法):")
        for method in group:
            print(f"  {method}")
    print("final answer: \n", json.dumps(total, indent=4))
    return total

def execute(inference_model, target, query, preds, args):
    corpus = get_corpus(args.test_dir, args.dataset, target)
    total = execute_xai(inference_model, corpus, target, query, preds, args)
    return total
    
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out after 5 minutes")

def main():
    set_global_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="test_index")
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite") # loc-agent
    parser.add_argument("--retrieval_model", type=str, default="Salesforce/SweRankEmbed-Large")
    parser.add_argument("--inference_model", type=str, default="x-ai/grok-code-fast-1") 
    # Qwen/Qwen3-Coder-30B-A3B-Instruct, qwen3-coder-480b-a35b-instruct, x-ai/grok-code-fast-1, openai/gpt-oss-20b
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="instances.json")
    parser.add_argument("--retrieval", type=str, default="embed32-retrieval.json")
    parser.add_argument("--saving", type=str, default="result-xai.json")
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)
    args.saving = os.path.join(args.test_dir, args.dataset + '-' + args.saving)
    # args.target = os.path.join(args.test_dir, args.dataset + '-' + args.target)
    args.retrieval = os.path.join(args.test_dir, args.dataset + '-' + args.retrieval)
    args.target = os.path.join(args.test_dir, args.target)
    print("inference_model: ", args.inference_model)
    print("target: ", args.target)
    print("retrieval: ", args.retrieval)
    print("to save in: ", args.saving)
    
    if not os.path.exists(args.retrieval):
        retrieve_batch(args)
    with open(args.retrieval, 'r', encoding='utf-8') as f:
        retrieval_dict = json.load(f)
    try:
        with open(args.saving, 'r', encoding='utf-8') as f:
            saving_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        saving_dict = {}

    # execute
    # inference_model = load_inference_model(args.inference_model)
    inference_model = None
    RESTART_FREQUENCY = 10
    target_processed_count = 0 
    with open(args.target, 'r', encoding='utf-8') as f:
        data = json.load(f)
        keys = list(data.keys())
        print("total: ", len(keys))
        results = saving_dict
        for target in keys:
            if target not in retrieval_dict:
                continue
            if target in results and results[target] is not None and isinstance(results[target], list):
                continue
            # if target != "sympy__sympy-11870":
            #     continue

            if inference_model is None or target_processed_count % RESTART_FREQUENCY == 0:
                if inference_model is not None:
                    print(f"--- 达到 {RESTART_FREQUENCY} 例重启点，销毁旧 VLLM 实例 ---")
                    del inference_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(5)
                
                inference_model = load_inference_model(args.inference_model) 

            print("current target: ", target)
            query = retrieval_dict[target]["query"]
            preds = retrieval_dict[target]["preds"]

            max_retries = 1
            retries = 0
            answer = None

            while retries < max_retries and answer is None:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                try:
                    answer = execute(inference_model, target, query, preds, args)
                    signal.alarm(0)
                    if retries > 0:
                        print(f"Retry successful for {target}")
                except TimeoutError:
                    signal.alarm(0)
                    del inference_model
                    inference_model = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if retries < max_retries:
                        print(f"Timeout: {target} execution exceeded 5 minutes, retrying...")
                        retries += 1
                    else:
                        print(f"Timeout: {target} execution failed after {max_retries} retry")
                        answer = None
                except Exception as e:
                    signal.alarm(0)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"Other error occurred for {target}: {type(e).__name__}: {e}")
                    print(f"{str(e)}")
                    print(f"{repr(e)}")
                    if retries < max_retries:
                        print(f"Retrying due to {type(e).__name__}...")
                        retries += 1
                    else:
                        print(f"Failed after {max_retries} retries due to {type(e).__name__}")
                        answer = None

            # saving immediately
            results[target] = answer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            target_processed_count += 1
            with open(args.saving, 'w', encoding='utf-8') as save_file:
                json.dump(results, save_file, indent=2, ensure_ascii=False)
            print(f"Saved result for {target}")

if __name__ == "__main__":
    main()
