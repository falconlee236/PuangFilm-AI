# PuangFilm-AI
A custom AI image generation project based on SDXL (Stable Diffusion XL) with LoRA training for personalized image creation. This project specializes in generating high-quality personalized portraits using fine-tuned AI models.
Features

# Custom LoRA model training based on SDXL
Personalized image generation through prompts
Inference support using trained models
High-performance training with GPU acceleration
Automated model optimization
Memory-optimized inference process

# Tech Stack

Python 3.x
Hugging Face Diffusers
PyTorch with CUDA support
Accelerate
PEFT (Parameter-Efficient Fine-Tuning)
SDXL Base 1.0

# Installation

Clone the repository
```bash
git clone https://github.com/falconlee236/PuangFilm-AI.git
cd PuangFilm-AI
```
## Setup environment
```bash
bashCopychmod +x setup.sh
./setup.sh
```
## Install required packages
```bash
conda env create -f environment.yaml
```

## Usage
1. Training the Model
Place your training images in the saved_images directory and run:
```bash
./train.sh
```


Training parameters:

Resolution: 720px
Training steps: 500
Batch size: 1
Learning rate: 2e-4
Gradient accumulation steps: 4

2. Generating Images
To generate images using the trained model:
```bash
python inference.py --class_name="[name]" --gender="[gender]" --output="[output_filename]"
```

## Example Results
### Input Images
The model was trained on a set of reference images (4~6 images used as example):


###  Output Results


## Troubleshooting
### GPU Memory Management
__Issue: CUDA Out of Memory Error__

When running inference on Tesla T4 (16GB), you might encounter:
```bash
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB
```

__Solution:__
Implemented memory optimization techniques from Hugging Face Diffusers Memory Guide:
Successfully reduced memory usage
Now able to generate multiple images (achieved max usage of 12949 MB)

Memory Usage Stats (Tesla T4):

Total GPU Memory: 15360 MiB
Peak Memory Usage: ~12949 MiB
Optimal Memory State: 7049 MiB (idle state)

## Best Practices

### Input Images:

Recommended number: 4-5 training images
Keep consistent image quality and style
Use high-resolution source images


### Generation Settings:

Batch size: 1 (for memory efficiency)
Resolution: 720p (balanced quality and memory usage)
Steps: 40 (default for inference)


### System Requirements:

Minimum: Tesla T4 or equivalent
RAM: 16GB or higher
CUDA Version: 12.4 compatible



## Performance Monitoring

* Use nvidia-smi to monitor GPU usage
* Clear GPU cache between generations if needed
* Use torch.cuda.empty_cache() for memory management
* Monitor temperature and memory usage during long training sessions

## Project Structure
```bash
.
├── train_dreambooth_lora_sdxl.py  # Main training script
├── inference.py                    # Inference script
├── train.sh                       # Training execution script
├── setup.sh                       # Environment setup script
├── find_best_model.sh             # Model optimization script
├── environment.yaml               # Environment configuration
├── saved_images/                  # Directory for training images
└── output/                        # Directory for generated images
```

## Requirements

* CUDA-enabled GPU for training
* Minimum 16GB GPU memory recommended
* Python 3.8 or higher
* Docker (optional, for containerized execution)

## Implementation Details
The project uses:

* SDXL base 1.0 as the foundation model
* LoRA for efficient fine-tuning
* FP16 mixed precision training
* Constant learning rate scheduler
* Custom VAE path: "madebyollin/sdxl-vae-fp16-fix"

# License
This project is distributed under the terms specified in the LICENSE file.

For issues and feature requests, please use the GitHub issue tracker
