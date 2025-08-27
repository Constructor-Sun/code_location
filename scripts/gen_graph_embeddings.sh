export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph embeddings for SWE-bench_Lite
python preprocess/save_embedding.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --index_dir index_data \
        --model_path Qwen/Qwen3-Embedding-0.6B \
        --num_processes 4 \
        --batch_size 16