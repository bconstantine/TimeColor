CUDA_VISIBLE_DEVICES="5,6" torchrun --nproc_per_node=2 src/run_json_sample_cfgparallel.py \
    --model-path "THUDM/CogVideoX-5b" \
    --cache-dir "./checkpoint/" \
    --transformer-path "./checkpoint/TimeColor-final/model_weights" \
    --seed 0 \
    --guidance-scale 3.0 \
    --work_json_path ./examples/inference_samples_xdit.json