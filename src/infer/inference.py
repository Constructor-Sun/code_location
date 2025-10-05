import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import re
import gc
import sys
import json
import argparse
import random
import tempfile
import numpy as np
import torch
import psutil
from functools import partial
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import VLLM
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
    llm = VLLM(
        model=model_name,
        max_new_tokens=80960*2,
        temperature=0, # 0.7
        top_p=0.8,
        top_k=20,
        do_sample=False, # True
        repetition_penalty=1.2,
        return_full_text=False,
        tensor_parallel_size=2,
        vllm_kwargs={
            "max_model_len": 80960*2,
            "gpu_memory_utilization": 0.9,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 65536
        }
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
        prompt=prompt
    )
    executor_1 = AgentExecutor(
        agent=agent_1,
        tools=tools_1,
        memory=memory,
        max_iterations=None,
        # verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    agent_2 = create_react_agent(
        tools=tools_2,
        llm=llm,
        prompt=prompt
    )
    executor_2 = AgentExecutor(
        agent=agent_2,
        tools=tools_2,
        memory=memory,
        max_iterations=None,
        # verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return {
        "first": executor_1,
        "second": executor_2
    }

def save_conversation_history(agent):
    chat_history = agent.memory
    print(agent.memory.load_memory_variables({}))
    print("saving successful!")

def execute(target, query, preds, args):
    inference_model = load_inference_model(args.inference_model)
    corpus = get_corpus(args.test_dir, args.dataset, target)

    formatted_first_prompt = FIRST_PROMPT.format(query=query, preds=str(preds))
    agent = create_agent(inference_model, corpus, target, args)

    # Step 1
    step_input = {"input": formatted_first_prompt, "chat_history": []}
    response_1 = agent["first"].invoke(step_input)

    # Step 2
    json_match = re.search(r'\{.*\}', response_1["output"], re.DOTALL)
    if json_match:
        json_str = json_match.group()
        response_final_1 = json.loads(json_str)
        candidate_methods = response_final_1["updated_methods"]
        partition = partition_methods(candidate_methods)
        print("init answer: \n", json.dumps(candidate_methods, indent=4))
        for i, group in enumerate(partition):
            print(f"\n第{i+1}组 ({len(group)}个方法):")
            for method in group:
                print(f"  {method}")
        # exit()
        total = []
        for key, methods in enumerate(partition):
            files = set([_split_path(path)[0] for path in methods])
            files = ", ".join(list(files))
            formatted_second_prompt = SECOND_PROMPT.format(query=query, candidates=methods, file=files)
            step_input = {"input": formatted_second_prompt, "chat_history": []}
            response_2 = agent["second"].invoke(step_input)
            # print("Files:", files)
            # print(f"updated_methods_{key}: ", response_2["output"])
            json_match = re.search(r'\{.*\}', response_2["output"], re.DOTALL)
            if json_match:
                json_str = json_match.group()
                response_final_1 = json.loads(json_str)
                current = response_final_1["methods_tobe_modified"]
                total.extend(current)

        print("original preds: ", json.dumps(preds, indent=4))
        print("init answer: \n", json.dumps(candidate_methods, indent=4))
        for i, group in enumerate(partition):
            print(f"\n第{i+1}组 ({len(group)}个方法):")
            for method in group:
                print(f"  {method}")
        print("final answer: \n", json.dumps(total, indent=4))
        return total
    else:
        return "Not found!"
    
def terminate_process_and_children(process):
    """Forcefully terminate a process and its children, ensuring GPU cleanup."""
    try:
        parent = psutil.Process(process.pid)
        # Terminate children first
        for child in parent.children(recursive=True):
            try:
                child.terminate()  # Try SIGTERM first for graceful shutdown
                child.wait(timeout=3)  # Wait briefly for termination
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                child.kill()  # Force SIGKILL if needed
        # Terminate parent
        parent.terminate()
        parent.wait(timeout=3)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        parent.kill()
    finally:
        process.wait() 

def reset_cuda():
    """Reset CUDA context and clear memory."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            torch.cuda.empty_cache()  # Release memory back to CUDA pool
            # Force reset of CUDA context (use cautiously)
            torch.cuda.init()
    except RuntimeError as e:
        print(f"CUDA reset error: {e}")

def main():
    set_global_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="test_index")
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite") # loc-agent
    parser.add_argument("--retrieval_model", type=str, default="Salesforce/SweRankEmbed-Large")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="instances.json")
    parser.add_argument("--retrieval", type=str, default="embed32-retrieval.json")
    parser.add_argument("--saving", type=str, default="result.json")
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)
    args.retrieval = os.path.join(args.test_dir, args.dataset + '-' + args.retrieval)
    args.saving = os.path.join(args.test_dir, args.dataset + '-' + args.saving)
    # args.target = os.path.join(args.test_dir, args.dataset + '-' + args.target)
    args.target = os.path.join(args.test_dir, args.target)
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
    with open(args.target, 'r', encoding='utf-8') as f:
        data = json.load(f)
        keys = list(data.keys())
        results = saving_dict
        for target in keys:
            # if target not in retrieval_dict:
            #     continue
            if target in results and results[target] is not None:
                continue
            # if target != "django__django-13158":
            #     continue
            query = retrieval_dict[target]["query"]
            preds = retrieval_dict[target]["preds"]

            max_retries = 1
            retries = 0
            answer = None

            while retries < max_retries and answer is None:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                    temp_path = f.name

                params = {
                    'target': target,
                    'query': query,
                    'preds': preds,
                    'args': vars(args)
                }
                params_json = json.dumps(params, ensure_ascii=False)

                process = subprocess.Popen(
                    ['python', 'src/infer/inference_single.py', '--result-file', temp_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )

                try:
                    # Communicate with the process, passing params and waiting for timeout
                    stdout, stderr = process.communicate(input=params_json, timeout=60)
                    if process.returncode == 0:
                        print(stdout)
                        print(stderr)
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(f"Content of {temp_path}: {content}")
                            f.seek(0)
                            answer_data = json.load(f)
                            if isinstance(answer_data, dict) and "error" in answer_data:
                                print(f"Error from inference_single: {answer_data['error']}")
                            else:
                                answer = answer_data
                    else:
                        print(f"Subprocess failed with return code {process.returncode}")
                        print(f"Stderr: {stderr}")
                except subprocess.TimeoutExpired:
                    print("Subprocess timed out after 240 seconds")
                    print(process.stdout)
                    print(process.stderr)
                    terminate_process_and_children(process)
                    reset_cuda()
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            results[target] = answer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with open(args.saving, 'w', encoding='utf-8') as save_file:
                json.dump(results, save_file, indent=2, ensure_ascii=False)
            
            print(f"Saved result for {target}")

if __name__ == "__main__":
    main()