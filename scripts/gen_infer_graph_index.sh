export PYTHONPATH=$PYTHONPATH:$(pwd)

# generate graph index for SWE-bench_Lite
python src/preprocess/batch_build_graph.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --split 'test' \
        --repo_path playground/build_graph \
        --instance_id_path preprocess/loaded_instance.json \
        --num_processes 50 \
        --download_repo