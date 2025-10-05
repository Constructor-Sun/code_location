import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# import ray
# ray.init(
#     num_cpus=10,
#     num_gpus=2,  # 明确告诉Ray有2个GPU可用
#     include_dashboard=False,
#     ignore_reinit_error=True
# )
import re
import gc
import json
import argparse
import random
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from functools import partial
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import VLLM
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate

llm = VLLM(
    model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.8
)
print("Model loaded successfully")