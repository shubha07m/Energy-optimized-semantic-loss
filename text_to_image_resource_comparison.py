import time
import torch
import numpy as np
from diffusers import DiffusionPipeline

### vitgpt ####
print('vitgpt2 start:')
print(int(time.time()))
pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")
pipe = pipe.to("cpu")
prompt = "a dog running through a grassy field "
np.random.seed(123)
output = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="pil").images[0]
print(int(time.time()))
output.save("vitgpt2_stable_diff.jpg")
print('\n')
time.sleep(5)

###### git large ####
print('git-large start:')
print(int(time.time()))
pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")
pipe = pipe.to("cpu")
prompt = "a dog running through a field of tall grass"
np.random.seed(123)
output = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="pil").images[0]
print(int(time.time()))
output.save("gitlarge_stable_diff.jpg")
print('\n')
time.sleep(5)

###### git base ####
print('git-base start:')
print(int(time.time()))
pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")
pipe = pipe.to("cpu")
prompt = "a dog running in the grass"
np.random.seed(123)
output = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="pil").images[0]
print(int(time.time()))
output.save("gitbase_stable_diff.jpg")
print('\n')
time.sleep(5)

###### blip base ####
print('blip-base start:')
print(int(time.time()))
pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")
pipe = pipe.to("cpu")
prompt = "a small dog running through a field"
np.random.seed(123)
output = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="pil").images[0]
print(int(time.time()))
output.save("blipbase_stable_diff.jpg")
print('\n')
time.sleep(5)

###### blip large ####
print('blip-large start:')
print(int(time.time()))
pipe = DiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0")
pipe = pipe.to("cpu")
prompt = "there is a dog running in the grass with a frisbee in its mouth"
np.random.seed(123)
output = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="pil").images[0]
print(int(time.time()))
output.save("bliplarge_stable_diff.jpg")
print('\n')
time.sleep(5)
