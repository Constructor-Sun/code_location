import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="czlll/SWE-bench_Lite")
    parser.add_argument("--index_dir", type=str, default="index_data")
    parser.add_argument("--graph_name", type=str, default="graph_index_v2.3")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--train_data", type=str, default=None)
    args = parser.parse_args()