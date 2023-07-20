import numpy as np
import subprocess
import psutil
import time
import torch
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, \
    BlipForConditionalGeneration, VisionEncoderDecoderModel, VisionEncoderDecoderModel, \
    AutoModelForCausalLM, BlipModel

device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open('browndog.jpg')

# Ensure below command is running in terminal while running this script
# sudo powermetrics -i 1000 --samplers cpu_power,gpu_power -a --hide-cpu-duty-cycle --show-usage-summary --show-extra-power-info >> powersample.txt

command = "awk '/Combined Power/{last=$0} END{print last}' powersample.txt | grep -oE '[0-9]+(\.[0-9]+)?'"


def gen_caption(model_id, img):
    np.random.seed(123)

    if model_id == 1:
        ########### VIT GPT2 ############

        processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        vitgpt_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model = vitgpt_model.to(device)
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    ############ GIT LARGE ###########

    if model_id == 2:
        processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        git_model_large = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
        model = git_model_large.to(device)
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_id == 3:
        ############ GIT BASE ###########
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        git_model_base = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
        model = git_model_base.to(device)
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_id == 4:
        ############ BLIP BASE ###########

        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = blip_model_base.to(device)
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_id == 5:
        ############ BLIP LARGE ###########

        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        model = blip_model_large.to(device)
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption


# pre-processing the text #

def text_preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())

    # removing the stop words and non alphaneumeric words#
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return Counter(tokens)  # tokenizing it


def txt_cosine_sim(text1, text2):
    # preprocessing #

    vec1 = text_preprocess(text1)
    vec2 = text_preprocess(text2)

    intersection = set(vec1.keys()) & set(vec2.keys())  # calculating common words
    numerator = sum([vec1[x] * vec2[x] for x in intersection])  # calculating the dot products

    # calculating the magnitude of vectors

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    # Putting back to the cosine similarity formula

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def eosl(y_true, y_pred):
    sem_loss = (1 - txt_cosine_sim(y_true, y_pred))
    cpu_utilization = psutil.cpu_percent()
    # TO-DO - calculate summation of power instead of instantaneous power
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    energy_loss = result.stdout.strip()
    total_power_loss = 10 * math.log10(float(energy_loss) / 1000)

    return sem_loss, total_power_loss,  cpu_utilization


text_semantic = 'a dog running through green grass'

for i in range(1, 6):
    print(eosl(text_semantic, gen_caption(i, img)))
