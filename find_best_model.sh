export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="saved_images"
export OUTPUT_DIR="lora-trained-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
#export CUDA_DEVICE_ORDER="PCI_BUS_ID"
#export CUDA_VISIBLE_DEVICES="0"
export GENDER="man"

echo "model init"

RANDOM_NAME=$(openssl rand -base64 12 | tr -dc 'a-z0-9' | fold -w 10 | head -n 1)
RANDOM_NAME=MinjiKim
echo $RANDOM_NAME

rm -rf /workspace/eaglefilm-model/lora-trained-xl/checkpoint-500
echo "remove checkpoint"
mkdir /workspace/eaglefilm-model/lora-trained-xl/checkpoint-500
cp -a /workspace/eaglefilm-model/saved_checkpoint/* /workspace/eaglefilm-model/lora-trained-xl/checkpoint-500
echo "checkpoint recovered"

accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="$RANDOM_NAME, 1 $GENDER, a photo of upper body" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --mixed_precision="fp16" \
    --report_to="wandb" \
    --validation_prompt="$RANDOM_NAME, 1 $GENDER, a photo of upper body" \
    --validation_epochs=25 \
    # --seed="0" \
    # --push_to_hub

python3 inference.py --class_name="$RANDOM_NAME" --gender="$GENDER" --output="$RANDOM_NAME"

# reference
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md
