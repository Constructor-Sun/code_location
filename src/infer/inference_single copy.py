import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from functools import partial
from langchain import hub
from langchain.agents import AgentExecutor, ConversationalAgent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import VLLM
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from retrieval import retrieve
from prompt import *
from tools import *

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def load_inference_model(model_name, device="cuda:1"):
    llm = VLLM(
        model=model_name,
        max_new_tokens=102400,
        temperature=0, # 0.7
        top_p=0.8,
        top_k=20,
        do_sample=False, # True
        repetition_penalty=1.2,
        return_full_text=False,
        tensor_parallel_size=2,
        vllm_kwargs={
            "max_model_len": 102400,
            "gpu_memory_utilization": 0.9,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 65536
        }
    )
    return llm

def create_agent(llm, tools):
    # prompt = ChatPromptTemplate.from_messages([
    #     SystemMessage(content=SYSTEM_PROMPT),
    #     HumanMessage(content="{input}"),
    # ])
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # agent = ConversationalAgent.from_llm_and_tools(
    #     llm=llm,
    #     tools=tools,
    #     prompt=prompt,
    #     memory=memory,
    #     input_variables=["input", "chat_history"]
    # )
    agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        max_iterations=None,
        verbose=True,
        handle_parsing_errors=True
    )

    return executor

def save_conversation_history(agent, output_dir="tmp"):
    # Generate a unique filename using timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"conversation_history.json")
    messages = agent.memory.chat_memory.messages
    
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "human"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "unknown"
        history.append({
            "role": role,
            "content": msg.content,
            "timestamp": msg.additional_kwargs.get("timestamp", datetime.now().isoformat())
        })
    
    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"conversation_history": history}, f, ensure_ascii=False, indent=2)
    
    return output_path

def main():
    set_global_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="test_index")
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite") # loc-agent
    parser.add_argument("--retrieval_model", type=str, default="Salesforce/SweRankEmbed-Large")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="scikit-learn__scikit-learn-25500")
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)

    query, preds = retrieve(args)
    # print("query: ", query)
    # print("preds: ", preds)

    corpus = get_corpus(args.test_dir, args.dataset, args.target)
    inference_model = load_inference_model(args.inference_model)
    # prompt = create_prompt_template()
    # chain = prompt | inference_model | StrOutputParser()

    # input_data = {
    #     "query": query,
    #     "preds": str(preds)
    # }
    # response = chain.invoke(input_data)
 
    tools = [
        # StructuredTool.from_function(
        #     func=partial(get_thought),
        #     name="get_thought",
        #     description="You should put your thought as parameters",
        #     args_schema=GetThoughtInput,
        #     return_direct=False,
        #     handle_tool_error=True,
        # ),
        StructuredTool.from_function(
            func=partial(check_validation, corpus),
            name="check_validation",
            description="Check if all functions are valid at the final step",
            args_schema=CheckValidationInput,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=partial(get_functions, corpus),
            name="get_functions",
            description="Retrieve function code from corpus dictionary for one or multiple function paths",
            args_schema=GetFunctionsInput,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=partial(get_class, corpus),
            name="get_class",
            description="Retrieve class content from corpus dictionary",
            args_schema=GetClassInput,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=partial(get_file, corpus, os.path.join(args.dataset, args.target)),
            name="get_file",
            description="Retrieve all functions from a specific file path in the corpus dictionary.",
            args_schema=GetFileInput,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=partial(list_function_directory, corpus),
            name="list_function_directory",
            description="Reveal all function names within this file.",
            args_schema=ListFunctionDirectoryInput,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=partial(get_call_graph, corpus, os.path.join(args.dataset, args.target)),
            name="get_call_graph",
            description="Generate a call graph for a target function using code2flow. Analyzes function dependencies within specified scopes.",
            args_schema=GetCallGraphInput,
            return_direct=False,
            handle_tool_error=True,
        )
    ]

    formatted_first_prompt = FIRST_PROMPT.format(
                                query=query,
                                preds=str(preds)
                                )
    agent = create_agent(inference_model, tools)

    input_data = {
        "input": formatted_first_prompt,
        "chat_history": []
    }
    try:
        print("results:")
        response = agent.invoke(input_data)
        print(response["output"])
        history_path = save_conversation_history(agent)
        print("history saved at: ", history_path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
    

# This is used for API calling
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_functions",
#             "description": "Retrieve function code from corpus dictionary for one or multiple function paths",
#             "parameters": {
#                 "type": "object",
#                 "required": ["func_paths"],
#                 "properties": {
#                     "func_paths": {
#                         "type": "array",
#                         "items": {
#                             "type": "string"
#                         },
#                         "description": "List of function paths to retrieve from the corpus dictionary. Can be a single string or list of strings."
#                     }
#                 }
#             }
#         }
#     }
# ]
