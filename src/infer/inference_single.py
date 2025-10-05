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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', required=True)
    sub_args = parser.parse_args()
    
    try:
        input_data = json.loads(sys.stdin.read())
        
        args_obj = argparse.Namespace()
        for key, value in input_data['args'].items():
            setattr(args_obj, key, value)
        
        result = execute(
            input_data['target'],
            input_data['query'],
            input_data['preds'],
            args_obj
        )
        
        with open(sub_args.result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            
    except Exception as e:
        print("error: ", e)

if __name__ == "__main__":
    main()