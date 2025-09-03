export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph embeddings for SWE-bench_Lite
python src/preprocess/save_graph_embedding.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --index_dir index_data \
        --model_path Qwen/Qwen3-Embedding-0.6B \
        --num_processes 4 \
        --batch_size 16