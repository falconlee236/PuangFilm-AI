#!/bin/sh

git clone https://github.com/huggingface/diffusers
pip install -e ./diffusers/.
pip install -r ./diffusers/examples/dreambooth/requirements_sdxl.txt
pip install peft==0.10.0
rm -rf diffusers