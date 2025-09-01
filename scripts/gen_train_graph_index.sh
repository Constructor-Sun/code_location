export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph index for SWE-bench_Lite
python src/preprocess/batch_build_graph.py \
        --dataset 'SWE-bench/SWE-smith' \
        --split 'train' \
        --repo_path playground/train_graph \
        --index_dir train_index \
        --num_processes 40 \
        --train