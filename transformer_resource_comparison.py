import time
# run this script alongside a few seconds after running CPU utilization and powermetric (below) in terminal
# # sudo powermetrics -i 1000 --samplers cpu_power,gpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info | grep -E -- 'Power|Sampled' >> powersample.txt
t_start = time.time()
print('start of the program:')
print(int(t_start))

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, \
    BlipForConditionalGeneration, VisionEncoderDecoderModel, VisionEncoderDecoderModel, \
    AutoModelForCausalLM, BlipModel

device = "cuda" if torch.cuda.is_available() else "cpu"
img = Image.open('browndog.jpg')
print('image and library loaded!')
print(int(time.time()))


def generate_caption(image):
    np.random.seed(123)

    ########### VIT GPT2 ############
    print('start of the model execution with vit gpt2:')
    print(int(time.time()))
    processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vitgpt_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    model = vitgpt_model.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(int(time.time()))
    print(generated_caption)
    print('\n')

    ############ GIT LARGE ###########
    time.sleep(5)
    print('starting of git large:\n')
    print(int(time.time()))
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    git_model_large = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

    model = git_model_large.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(int(time.time()))
    print(generated_caption)
    print('\n')

    ############ GIT BASE ###########
    time.sleep(5)
    print('starting of git base:\n')
    print(int(time.time()))
    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    git_model_base = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    model = git_model_base.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(int(time.time()))
    print(generated_caption)
    print('\n')

    ############ BLIP BASE ###########
    time.sleep(5)
    print('starting of blip base:\n')
    print(int(time.time()))
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    model = blip_model_base.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(int(time.time()))
    print(generated_caption)
    print('\n')

    ############ BLIP LARGE ###########
    time.sleep(5)
    print('starting of blip large:\n')
    print(int(time.time()))
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    model = blip_model_large.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(int(time.time()))
    print(generated_caption)
    print('\n')


if __name__ == "__main__":
    print('\n')
    time.sleep(5)
    generate_caption(img)
    time.sleep(5)
    print('end of program execution:')
    print(int(time.time()))