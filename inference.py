from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import os
import argparse

# 파서 생성
parser = argparse.ArgumentParser(description="Print a random name passed as an argument.")
# 'class_name' 인자 추가
parser.add_argument('--class_name', type=str, help='The random name to print.')
parser.add_argument('--gender', type=str, help='gender.')
parser.add_argument('--output', type=str, help='output file')
# 인자 분석
args = parser.parse_args()

model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.load_lora_weights("lora-trained-xl/pytorch_lora_weights.safetensors")
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
)
refiner.to("cuda")

print("-----")

GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print('Current cuda device ', torch.cuda.current_device()) # check

prompt = f"A picture of a {args.class_name}, 1 {args.gender}, upper body, high quality, highly detailed eyes, photo realistic, 8k, profile, natural, vivid, cute, without accessories"
negative_prompt = "ugly, deformed, noisy, blurry, distorted, grainy, text, cropped, EasyNegative"
generator = torch.Generator("cuda").manual_seed(43)

# Run inference.
print("run inference")
image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=40)
image = image.images[0]
print("---")
print(torch.cuda.empty_cache())
print("---")
image = refiner(prompt=prompt, negative_prompt=negative_prompt, generator=generator, image=image)
image = image.images[0]

FILE_NAME = args.output
FILE_EXTENSION = '.png'
output_path = f'output/{FILE_NAME}{FILE_EXTENSION}'
uniq = 1
while os.path.exists(output_path):
  output_path = f'output/{FILE_NAME}({uniq}){FILE_EXTENSION}'
  uniq += 1

image.save(output_path)
