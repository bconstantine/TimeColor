CUDA_DEVICES="4"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python preprocess/sketch_generation/preprocess_keyframe_sketch.py \
    --dataroot './examples/sample_video_sketch_generation/colored_videos' \
    --output_folder './examples/sample_video_sketch_generation/result_sketch'\
    --input_type 'video' \
    --gpu_ids $CUDA_DEVICES \
    --binarize_threshold 250