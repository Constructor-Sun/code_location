import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
import torch.distributed as dist
from datetime import datetime
from functools import partial
from langchain.agents import AgentExecutor, ConversationalAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import VLLM
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from retrieval import retrieve
from prompt import *
from tools import (
    get_corpus, 
    get_functions, 
    get_file, 
    get_call_graph_json,
    GetFunctionsInput,
    GetFileInput,
    GetCallGraphInput
)

def load_inference_model(model_name, device="cuda:1"):
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    #     torch_dtype="auto",
    #     device_map=device
    # )

    # hf_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=2048,
    #     temperature=0.7,
    #     top_p=0.8,
    #     top_k=20,
    #     do_sample=True,
    #     repetition_penalty=1.05,
    #     return_full_text=False
    # )
    
    # inference_model = HuggingFacePipeline(pipeline=hf_pipeline)
    # return inference_model

    llm = VLLM(
        model=model_name,
        max_new_tokens=65536,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        do_sample=True,
        repetition_penalty=1.05,
        return_full_text=False,
        vllm_kwargs={
            "max_model_len": 65536,
            "gpu_memory_utilization": 0.9
        }
    )
    return llm

def create_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEMP_PROMPT),
        HumanMessage(content="{input}"),
    ])
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    agent = ConversationalAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        prompt=prompt,
        memory=memory,
        input_variables=["input", "chat_history"]
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        max_iterations=20,
        verbose=True,
        handle_parsing_errors=True
    )

    return executor

def save_conversation_history(agent, output_dir="tmp"):
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"conversation_history_{timestamp}.json")
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
    print("query: ", query)
    print("preds: ", preds)

    corpus = get_corpus(args.test_dir, args.dataset, args.target)
    # funcs=["sklearn/calibration.py/CalibratedClassifierCV/predict_proba", "sklearn/calibration.py/IsotonicRegression/predict", "sklearn/_config.py/set_config"]
    # print("sklearn/calibration.py/CalibratedClassifierCV/predict_proba: ", get_functions(corpus, funcs))
    # exit()
    inference_model = load_inference_model(args.inference_model)
    # prompt = create_prompt_template()
    # chain = prompt | inference_model | StrOutputParser()

    # input_data = {
    #     "query": query,
    #     "preds": str(preds)
    # }
    # response = chain.invoke(input_data)
 
    tools = [
        StructuredTool.from_function(
            func=partial(get_functions, corpus),
            name="get_functions",
            description="Retrieve function code from corpus dictionary for one or multiple function paths",
            args_schema=GetFunctionsInput,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=partial(get_file, corpus),
            name="get_file",
            description="Retrieve all functions from a specific file path in the corpus dictionary.",
            args_schema=GetFileInput,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=partial(get_call_graph_json, os.path.join(args.dataset, args.target)),
            name="get_call_graph_json",
            description="Generate a call graph for a target function using code2flow. Analyzes function dependencies within specified scopes.",
            args_schema=GetCallGraphInput,
            return_direct=False,
            handle_tool_error=True,
        )
    ]

    formatted_first_prompt = FIRST_PROMPT.format(query=query, preds=str(preds))
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
