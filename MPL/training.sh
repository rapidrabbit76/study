accelerate launch --config_file ./default_config.yaml train.py \
    --seed 42 \
    --total_steps 300000 \
    --batch_size 128 \
    --backbone resnet50 