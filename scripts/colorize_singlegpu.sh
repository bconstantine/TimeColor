CUDA_DEVICES="3"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python src/run_json_sample_singlegpu.py \
    --model-path "THUDM/CogVideoX-5b" \
    --cache-dir "./checkpoint/" \
    --transformer-path "./checkpoint/TimeColor-final/model_weights" \
    --seed 0 \
    --guidance-scale 3.0 \
    --work_json_path ./examples/inference_samples.json
