import math
from bitcounter import countTotalBits
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


def img_similarity(fname1, fname2, rmv_bg=0, sim_type=1):
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
    else:
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
