export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph index for SWE-bench_Lite
python preprocess/batch_build_graph.py \
        --dataset 'SWE-bench/SWE-smith' \
        --split 'train' \
        --repo_path playground/build_graph \
        --index_dir train_index \
        --num_processes 30 \
        --download_repo \
        --reduce