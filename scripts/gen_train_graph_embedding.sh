export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph embeddings for SWE-bench_Lite
python src/preprocess/save_graph_embedding.py \
        --dataset 'SWE-bench/SWE-smith' \
        --index_dir train_index \
        --graph_name 'graph_index_v1.0' \
        --save_name graph_embedding_SweRank\
        --model_path Salesforce/SweRankEmbed-Small \
        --num_processes 4 \
        --batch_size 16 