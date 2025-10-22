# Retriever-Agent vs. Retriever-Reranker: Replacing the Backend for Issue Localization

## Dataset Preparation

Download the `SWE-Bench-Lite` dataset from [HuggingFace](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite).

## Environments

Create a conda environment with Python >= 3.10 and install dependencies:

```bash
pip install -r requirements.txt
```

## Inference

Set your API key in the .env file:

- For Qwen3 series, set ALIYUN_API_KEY.
- For Grok series, set OPENROUTER_API_KEY.

Run the following command to get results:

```bash
bash scripts/inference.sh
```

Set **inference_model** in above file to:

- Qwen/Qwen3-Coder-30B-A3B-Instruct for local GPU inference.
- qwen3-coder-480b-a35b-instruct for API-based inference.

To get results from xai, use:

```bash
bash scripts/inference_xai.sh
```

## Evaluation and Analysis

Evaluate results and identify failure instances:

```bash
bash scripts/eval.sh
```

You can use `visualize.py` and `analyze.py` to get upset plot and venn plot.

## Others

In fact you can use any model on the OpenRouter platform (e.g., Claude or Gemini series). Note:

- API compatibility with the current ReAct framework in LangChain may vary.
- Different APIs may require specific prompts for optimal performance.
- Be mindful of API costs.
