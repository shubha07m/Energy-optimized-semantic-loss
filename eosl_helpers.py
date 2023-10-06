import math
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from numpy import dot
from numpy.linalg import norm
from rembg import remove
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


def img_similarity(fname1, fname2, sim_type=1, rmv_bg=0):
    # open images #

    f1 = Image.open(fname1)
    f2 = Image.open(fname2)

    # remove background #

    if rmv_bg == 1:
        f1 = remove(f1)
        f2 = remove(f2)

    # using .resize to scale image 2 to match image 1 dimensions #

    f1_reshape = f1.resize((round(f1.size[0]), round(f1.size[1])))
    f2_reshape = f2.resize((round(f1.size[0]), round(f1.size[1])))

    # convert the images to (R,G,B) arrays #

    f1_array = np.array(f1_reshape)
    f2_array = np.array(f2_reshape)

    # flatten the arrays to one dimensional vectors

    f1_array = f1_array.flatten()
    f2_array = f2_array.flatten()

    # divide the arrays by 255, the maximum RGB value to make sure every value is on a 0-1 scale #

    a = f1_array / 255
    b = f2_array / 255

    if sim_type == 1:
        cos_similarity = dot(a, b) / (norm(a) * norm(b))
        return cos_similarity
    if sim_type == 0:
        ssim = structural_similarity(a, b, data_range=1)
        return ssim


def txt_preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())

    # removing the stop words and non alphaneumeric words#
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return Counter(tokens)  # tokenizing it


def txt_similarity(text1, text2):
    # preprocessing #

    vec1 = txt_preprocess(text1)
    vec2 = txt_preprocess(text2)

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


def calculate_Lch(l, pb, t=2):
    pb_bar = pb
    Lch = 1

    for i in range(t + 1):
        binomial_coefficient = math.comb(l, i)
        Lch -= binomial_coefficient * (pb_bar ** i) * ((1 - pb_bar) ** (l - i))
    return Lch


def energy_penalty(encoder, include_diffusion_energy=0):
    Es_img_txt = {'vit': 50.703, 'gitbase': 197.442, 'blipbase': 60.923, 'gitlarge': 524.719, 'bliplarge': 105.095}
    Es_in = Es_img_txt[encoder]
    Es_in_max = max(Es_img_txt.values())

    # DO NOT USE diffusion energy values now, needs update #
    if include_diffusion_energy:
        Es_txt_img = {'vit': 4793.78, 'gitbase': 4177.509, 'blipbase': 4145.009, 'gitlarge': 4162.822,
                      'bliplarge': 4243.01}
        Es_out = Es_txt_img[encoder]
        Es_out_max = max(Es_txt_img.values())
        Energy_penalty = (Es_in + Es_out) / (Es_in_max + Es_out_max)
        return Energy_penalty

    Energy_penalty = Es_in / Es_in_max
    return Energy_penalty


def comm_energy(enc):
    p_max = 1
    max_data_rate = 143 * 10 ** 6

    message_length = countTotalBits(enc)
    t = message_length / max_data_rate
    Ec = p_max * t

    message_length_max = 1500 * 8
    t_max = message_length_max / max_data_rate
    Ec_max = p_max * t_max

    return Ec / Ec_max


def countTotalBits(sem_string):
    count = 0
    for s in range(len(sem_string)):
        count += 8
    return count
