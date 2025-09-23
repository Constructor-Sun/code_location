import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import argparse
import torch.distributed as dist
from datetime import datetime
from functools import partial
from langchain.agents import AgentExecutor, ConversationalAgent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import VLLM
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from retrieval import retrieve
from prompt import *
from tools import (
    get_corpus, 
    get_functions, 
    get_file, 
    list_function_directory,
    get_call_graph,
    GetFunctionsInput,
    GetFileInput,
    ListFunctionDirectoryInput,
    GetCallGraphInput
)

# Custom callback to track iterations and trigger reflection
class ReflectionCallback(BaseCallbackHandler):
    def __init__(self, reflection_interval=5):
        self.iteration_count = 0
        self.reflection_interval = reflection_interval
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a critical reviewer tasked with evaluating the agent's responses. Review the conversation history, critique the agent's performance, and provide specific recommendations for improvement. Focus on clarity, accuracy, relevance, and effectiveness of the responses. Suggest how the agent can better address the user's query."),
            MessagesPlaceholder(variable_name="chat_history")
        ])
        self.llm = None
        self.memory = None

    def set_llm_and_memory(self, llm, memory):
        self.llm = llm
        self.memory = memory

    def on_agent_action(self, action, **kwargs):
        """Called on each agent action (iteration)."""
        self.iteration_count += 1
        if self.iteration_count % self.reflection_interval == 0:
            print("perform reflection!")
            self.perform_reflection()

    def perform_reflection(self):
        """Perform reflection by critiquing the conversation history."""
        if self.llm is None or self.memory is None:
            print("Reflection skipped: LLM or memory not set.")
            return

        try:
            # Get conversation history
            chat_history = self.memory.chat_memory.messages
            print(f"Reflection triggered at iteration {self.iteration_count}. History length: {len(chat_history)}")

            # Invoke reflection prompt
            reflection_chain = self.reflection_prompt | self.llm
            reflection_output = reflection_chain.invoke({"chat_history": chat_history})

            # Handle reflection output (string or AIMessage)
            reflection_content = reflection_output if isinstance(reflection_output, str) else getattr(reflection_output, 'content', str(reflection_output))

            # Append reflection as a HumanMessage
            reflection_message = HumanMessage(
                content=f"Reflection Feedback: {reflection_content}",
                additional_kwargs={"timestamp": datetime.now().isoformat()}
            )
            self.memory.chat_memory.add_message(reflection_message)
            print(f"Reflection feedback added: {reflection_content[:100]}...")  # Truncate for brevity
        except Exception as e:
            print(f"Error during reflection: {str(e)}")

def load_inference_model(model_name, device="cuda:1"):
    llm = VLLM(
        model=model_name,
        max_new_tokens=81920,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        do_sample=True,
        repetition_penalty=1.05,
        return_full_text=False,
        vllm_kwargs={
            "max_model_len": 81920,
            "gpu_memory_utilization": 0.9,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 65536,
        }
    )
    return llm

def create_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
    ])
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
    # Initialize callback
    reflection_callback = ReflectionCallback(reflection_interval=5)
    reflection_callback.set_llm_and_memory(llm, memory)

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
        max_iterations=None,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[reflection_callback]
    )

    return executor

def save_conversation_history(agent, output_dir="tmp"):
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
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"conversation_history": history}, f, ensure_ascii=False, indent=2)
    
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="test_index")
    parser.add_argument("--test_dir", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="swe-bench-lite")
    parser.add_argument("--retrieval_model", type=str, default="Salesforce/SweRankEmbed-Large")
    parser.add_argument("--inference_model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--target", type=str, default="sympy__sympy-23262")
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)

    query, preds = retrieve(args)
    
    corpus = get_corpus(args.test_dir, args.dataset, args.target)
    inference_model = load_inference_model(args.inference_model)

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
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()