# run this script alongside a few seconds after running CPU utilization and powermetric (below) in terminal
# sudo powermetrics -i 1000 --samplers cpu_power,gpu_power -a --hide-cpu-duty-cycle --show-usage-summary
# --show-extra-power-info | grep -E -- 'Power|Sampled' >> powersample_encoder.txt (or _decoder)

import time
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, \
    BlipForConditionalGeneration, VisionEncoderDecoderModel, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"
img = Image.open('browndog.jpg')
print('image and library loaded!')
print(int(time.time()))


def generate_caption(image):
    # Create a list of models and their corresponding checkpoint names
    models = [
        ("nlpconnect/vit-gpt2-image-captioning", "VisionEncoderDecoderModel", "AutoImageProcessor"),
        ("microsoft/git-large-coco", "AutoModelForCausalLM", "AutoProcessor"),
        ("microsoft/git-base-coco", "AutoModelForCausalLM", "AutoProcessor"),
        ("Salesforce/blip-image-captioning-base", "BlipForConditionalGeneration", "AutoProcessor"),
        ("Salesforce/blip-image-captioning-large", "BlipForConditionalGeneration", "AutoProcessor")
    ]

    for model_checkpoint, model_type, processor_type in models:
        print(f'Starting execution of {model_type} with checkpoint: {model_checkpoint}')
        # random_seed = int(time.time())
        set_seed(200)
        print(int(time.time()))

        # Load the processor, model, and tokenizer based on the model type
        processor = eval(f"{processor_type}.from_pretrained")(model_checkpoint)
        model = eval(f"{model_type}.from_pretrained")(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        inputs = processor(images=image, return_tensors="pt").to(device)

        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50, do_sample=True, num_beams=1)

        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_caption)
        print(int(time.time()))
        print('\n')
        time.sleep(5)


if __name__ == "__main__":
    print('\n')
    time.sleep(5)
    generate_caption(img)
    time.sleep(5)
    print('end of program execution:')
    print(int(time.time()))
