export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph embeddings for SWE-bench_Lite
python preprocess/save_embedding.py \
        --dataset 'SWE-bench/SWE-smith' \
        --index_dir train_index \
        --graph_name 'graph_index_v1.0' \
        --model_path Qwen/Qwen3-Embedding-0.6B \
        --num_processes 4 \
        --batch_size 16 