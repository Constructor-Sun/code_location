import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import re
import gc
import json
import argparse
import random
import numpy as np
import torch
import asyncio
from concurrent.futures import ProcessPoolExecutor, TimeoutError
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
            "gpu_memory_utilization": 0.9
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
        verbose=True,
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
        verbose=True,
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

# def execute(inference_model, target, query, preds, args):
#     corpus = get_corpus(args.test_dir, args.dataset, target)

#     formatted_first_prompt = FIRST_PROMPT.format(query=query, preds=str(preds))
#     agent = create_agent(inference_model, corpus, target, args)

#     # Step 1
#     step_input = {"input": formatted_first_prompt, "chat_history": []}
#     response_1 = agent["first"].invoke(step_input)

#     # Step 2
#     json_match = re.search(r'\{.*\}', response_1["output"], re.DOTALL)
#     if json_match:
#         json_str = json_match.group()
#         response_final_1 = json.loads(json_str)
#         candidate_methods = response_final_1["updated_methods"]
#         partition = partition_methods(candidate_methods)
#         total = []
#         for key, methods in enumerate(partition):
#             files = set([_split_path(path)[0] for path in methods])
#             files = ", ".join(list(files))
#             formatted_second_prompt = SECOND_PROMPT.format(query=query, candidates=methods, file=files)
#             step_input = {"input": formatted_second_prompt, "chat_history": []}
#             response_2 = agent["second"].invoke(step_input)
#             json_match = re.search(r'\{.*\}', response_2["output"], re.DOTALL)
#             if json_match:
#                 json_str = json_match.group()
#                 response_final_1 = json.loads(json_str)
#                 current = response_final_1["methods_tobe_modified"]
#                 total.extend(current)

#         # print("init answer::")
#         # for i, group in enumerate(partition):
#         #     print(f"\n第{i+1}组 ({len(group)}个方法):")
#         #     for method in group:
#         #         print(f"  {method}")
#         print("final answer: \n", json.dumps(total, indent=4))
#         return total
#     else:
#         return "Not found!"

async def execute(inference_model, target, query, preds, args, timeout=120):
    corpus = get_corpus(args.test_dir, args.dataset, target)

    formatted_first_prompt = FIRST_PROMPT.format(query=query, preds=str(preds))
    agent = create_agent(inference_model, corpus, target, args)

    try:
        # Step 1: Execute with timeout
        step_input = {"input": formatted_first_prompt, "chat_history": []}
        response_1 = await asyncio.wait_for(agent["first"].ainvoke(step_input), timeout=timeout)

        # Step 2
        json_match = re.search(r'\{.*\}', response_1["output"], re.DOTALL)
        if json_match:
            json_str = json_match.group()
            response_final_1 = json.loads(json_str)
            candidate_methods = response_final_1["updated_methods"]
            partition = partition_methods(candidate_methods)
            total = []
            for key, methods in enumerate(partition):
                files = set([_split_path(path)[0] for path in methods])
                files = ", ".join(list(files))
                formatted_second_prompt = SECOND_PROMPT.format(query=query, candidates=methods, file=files)
                step_input = {"input": formatted_second_prompt, "chat_history": []}
                response_2 = await asyncio.wait_for(agent["second"].ainvoke(step_input), timeout=timeout)
                json_match = re.search(r'\{.*\}', response_2["output"], re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    response_final_1 = json.loads(json_str)
                    current = response_final_1["methods_tobe_modified"]
                    total.extend(current)

            print("final answer: \n", json.dumps(total, indent=4))
            return total
        else:
            return "Not found!"
    except asyncio.TimeoutError:
        print(f"Timeout occurred for target {target} after {timeout} seconds")
        return "Timeout"

async def main():
    set_global_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="test_index")
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite") # loc-agent
    parser.add_argument("--retrieval_model", type=str, default="Salesforce/SweRankEmbed-Large")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="datasets/instances.json")
    parser.add_argument("--retrieval", type=str, default="retrieval.json")
    parser.add_argument("--saving", type=str, default="result.json")
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)
    args.saving = os.path.join(args.test_dir, args.dataset + '-' + args.saving)
    
    retrieval_path = os.path.join(args.test_dir, args.retrieval)
    if not os.path.exists(retrieval_path):
        retrieve_batch(args)
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        retrieval_dict = json.load(f)
    try:
        with open(args.saving, 'r', encoding='utf-8') as f:
            saving_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        saving_dict = {}

    # execute
    inference_model = load_inference_model(args.inference_model)
    with open(args.target, 'r', encoding='utf-8') as f:
        data = json.load(f)
        keys = list(data.keys())
        results = saving_dict
        for target in keys:
            if results[target] != "Timeout":
                continue
            query = retrieval_dict[target]["query"]
            preds = retrieval_dict[target]["preds"]

            max_retries = 1
            retries = 1
            answer = None

            while retries <= max_retries:
                try:
                    answer = await execute(inference_model, target, query, preds, args)
                    break
                except asyncio.TimeoutError:
                    print(f"Timeout occurred for target {target} on attempt {retries + 1}")
                    retries += 1
                    timeout += 100
                    torch.cuda.empty_cache()
                    gc.collect()
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"Error occurred for target {target} on attempt {retries + 1}: {str(e)}")
                    retries += 1

            # saving immediately
            results[target] = answer if answer is not None else "Timeout"
            torch.cuda.empty_cache()
            gc.collect()
            with open(args.saving, 'w', encoding='utf-8') as save_file:
                json.dump(results, save_file, indent=2, ensure_ascii=False)
            
            print(f"Saved result for {target}")

if __name__ == "__main__":
    asyncio.run(main())
