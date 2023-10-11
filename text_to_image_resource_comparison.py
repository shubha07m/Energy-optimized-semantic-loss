import time
import torch
import numpy as np
from diffusers import DiffusionPipeline


def generate_image(prompt, output_filename):
    print(f'Starting: {prompt}')
    print(int(time.time()))

    pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")
    pipe = pipe.to("cpu")

    np.random.seed(123)
    output = pipe(prompt, num_inference_steps=50, guidance_scale=7, output_type="pil").images[0]

    print(int(time.time()))
    output.save(output_filename)
    print('\n')
    time.sleep(5)


# List of prompts and output filenames
prompts_and_filenames = [
    ("a large brown dog chasing a brown grassy field", "vitgpt2_stable_diff.jpg"),
    ("image of dog running across the grass with dog running toward it", "gitlarge_stable_diff.jpg"),
    ("running dog, a golden retriever in a grassy field.", "gitbase_stable_diff.jpg"),
    ("a small golden golden retrieve playing on a beautiful, green lawn, as a blurred green background",
     "blipbase_stable_diff.jpg"),
    ("aboard a dog running through the grass by a tree line", "bliplarge_stable_diff.jpg"),
]

for prompt, filename in prompts_and_filenames:
    generate_image(prompt, filename)
