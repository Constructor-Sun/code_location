import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import VLLM
from langchain_community.chat_models import ChatOpenAI
from langchain_xai import ChatXAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate

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
    if model_name == "Qwen/Qwen3-Coder-30B-A3B-Instruct":
        llm = VLLM(
            model=model_name,
            max_new_tokens=80960*2,
            temperature=0, # 0.7
            top_p=0.8,
            top_k=30,
            do_sample=False, # True
            repetition_penalty=1.2,
            return_full_text=False,
            tensor_parallel_size=2,
            vllm_kwargs={
                "max_model_len": 80960*2,
                "gpu_memory_utilization": 0.8,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 80960
            }
        )
    elif model_name == "qwen3-coder-480b-a35b-instruct":
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key="sk-5368f4efcc9a464ca1a787f011186efa",
            max_tokens=65536,
            temperature=0,
            model_kwargs={"seed": 42},
            request_timeout=60,
            max_retries=2 
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1", # "https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key="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", # "sk-5368f4efcc9a464ca1a787f011186efa",
            max_tokens=65536,
            temperature=0,
            model_kwargs={"seed": 42},
            request_timeout=60,
            max_retries=2 
        )
    return llm

def create_agent(llm, corpus, target, args):
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    tool_func = StructuredTool.from_function(
        func=partial(get_functions, corpus),
        name="get_functions",
        description="Retrieve function code from corpus dictionary for one or multiple function paths",
        args_schema=GetFunctionsInput,
        return_direct=False,
        handle_tool_error=True,
    )
    tool_class = StructuredTool.from_function(
        func=partial(get_class, corpus),
        name="get_class",
        description="Retrieve class content from corpus dictionary",
        args_schema=GetClassInput,
        return_direct=False,
        handle_tool_error=True,
    )
    tool_file = StructuredTool.from_function(
        func=partial(get_file, corpus, os.path.join(args.dataset, target)),
        name="get_file",
        description="Retrieve all functions from a specific file path in the corpus dictionary.",
        args_schema=GetFileInput,
        return_direct=False,
        handle_tool_error=True,
    )
    tool_list = StructuredTool.from_function(
        func=partial(list_function_directory, corpus),
        name="list_function_directory",
        description="Reveal all function names within this file.",
        args_schema=ListFunctionDirectoryInput,
        return_direct=False,
        handle_tool_error=True,
    )
    tool_graph = StructuredTool.from_function(
        func=partial(get_call_graph, corpus, os.path.join(args.dataset, target)),
        name="get_call_graph",
        description="Generate a call graph for a target function using code2flow. Analyzes function dependencies within specified scopes.",
        args_schema=GetCallGraphInput,
        return_direct=False,
        handle_tool_error=True,
    )

    tools_1 = [tool_func, tool_file, tool_graph, tool_list, tool_class]
    tools_2 = [tool_func, tool_class, tool_file]

    agent_1 = create_react_agent(
        tools=tools_1,
        llm=llm,
        prompt=prompt,
        stop_sequence=False
    )
    executor_1 = AgentExecutor(
        agent=agent_1,
        tools=tools_1,
        memory=memory,
        max_iterations=30,
        # verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    agent_2 = create_react_agent(
        tools=tools_2,
        llm=llm,
        prompt=prompt,
        stop_sequence=False
    )
    executor_2 = AgentExecutor(
        agent=agent_2,
        tools=tools_2,
        memory=memory,
        max_iterations=20,
        # verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return {
        "first": executor_1,
        "second": executor_2
    }

def execute(inference_model, target, query, preds, args):
    corpus = get_corpus(args.test_dir, args.dataset, target)

    formatted_first_prompt = FIRST_PROMPT.format(query=query, preds=str(preds))
    agent = create_agent(inference_model, corpus, target, args)

    # Step 1
    step_input = {"input": formatted_first_prompt, "chat_history": []}
    response_1 = agent["first"].invoke(step_input)

    # Before step 2, make a judgement
    try:
        json_match = re.search(r'\{.*?\}', response_1["output"], re.DOTALL)
        json_str = json_match.group()
        response_final_1 = json.loads(json_str)
        candidate_methods = response_final_1["updated_methods"]

        # Aim to avoid path analysis error at the 1st stage
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
    # TODO: in fact there is another condition: almost module-level functions while no classes
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
        step_input = {"input": formatted_second_prompt, "chat_history": []}
        response_2 = agent["second"].invoke(step_input)
        json_match = re.search(r'\{.*\}', response_2["output"], re.DOTALL)
        # print("response_2 intermediate_steps:\n", response_2['intermediate_steps'])
        if json_match:
            json_str = json_match.group()
            response_final_2 = json.loads(json_str)
            current = response_final_2["methods_tobe_modified"]
            total.extend(current)
        else:
            intermediate_steps = response_2['intermediate_steps']
            valid = False
            for step in reversed(intermediate_steps):
                if (step and len(step) >= 1 and hasattr(step[0], 'log') and step[0].log):
                    json_match = re.search(r'\{.*\}', step[0].log, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        response_final_2 = json.loads(json_str)
                        current = response_final_2["methods_tobe_modified"]
                        total.extend(current)
                        valid = True
                        break
            if not valid:
                exception = Exception()
                exception.response = response_2
                raise exception
                    

    print("original preds: ", json.dumps(preds, indent=4))
    print("init answer: \n", json.dumps(candidate_methods, indent=4))
    for i, group in enumerate(partition):
        print(f"\n第{i+1}组 ({len(group)}个方法):")
        for method in group:
            print(f"  {method}")
    print("final answer: \n", json.dumps(total, indent=4))
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
    parser.add_argument("--inference_model", type=str, default="qwen3-coder-480b-a35b-instruct") 
    # Qwen/Qwen3-Coder-30B-A3B-Instruct, qwen3-coder-480b-a35b-instruct, x-ai/grok-code-fast-1
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="instances.json")
    parser.add_argument("--retrieval", type=str, default="embed32-retrieval.json")
    parser.add_argument("--saving", type=str, default="result-api.json")
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)
    args.saving = os.path.join(args.test_dir, args.dataset + '-' + args.saving)
    args.target = os.path.join(args.test_dir, args.dataset + '-' + args.target)
    args.retrieval = os.path.join(args.test_dir, args.dataset + '-' + args.retrieval)
    # args.target = os.path.join(args.test_dir, args.target)
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
    inference_model = None # 初始化为 None
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
                    time.sleep(5) # 增加延迟等待资源完全释放
                
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
