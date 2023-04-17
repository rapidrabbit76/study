accelerate launch --config_file ./default_config.yaml train.py \
    --total_steps 300000 \
    --batch_size 128 \
    --teacher_backbone WideResNet \
    --student_backbone WideResNet 