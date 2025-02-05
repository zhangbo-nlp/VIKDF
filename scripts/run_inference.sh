accelerate launch inference.py \
    --dataset_name data/reddit_data \
    --output_dir outputs/zrigf2_reddit \
    --model_path outputs/zrigf2 \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="1" \
    --gradient_accumulation_steps="8" \
    --num_train_epochs="5" \
    --learning_rate="1e-4" \
    --preprocessing_num_workers="4" \
    --seed="42" \
    --checkpointing_steps='epoch' \
    --train_data_size="1" \
    --with_tracking

accelerate launch inference.py \
    --dataset_name data/image_chat \
    --output_dir outputs/zrigf2_image_chat \
    --model_path outputs/zrigf2_reddit \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="1" \
    --gradient_accumulation_steps="8" \
    --num_train_epochs="1" \
    --learning_rate="1e-4" \
    --preprocessing_num_workers="4" \
    --seed="42" \
    --checkpointing_steps='epoch' \
    --train_data_size="1" \
    --with_tracking