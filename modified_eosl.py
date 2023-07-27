import math
import os.path
from bitcounter import countTotalBits
from similarity_functions import img_similarity, txt_similarity
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from numpy import dot
from numpy.linalg import norm
from rembg import remove
import re

sem_vitgpt2 = 'a dog running through a grassy field'
sem_gitlarge = 'a dog running through a field of tall grass'
sem_gitbase = 'a dog running in the grass'
sem_blipbase = 'a small dog running through a field'
sem_bliplarge = 'there is a dog running in the grass with a frisbee in its mouth'


def energy_penalty(encoder=None, include_diffusion_energy=0):
    # max datarate from table at
    # https://www.intel.com/content/www/us/en/support/articles/000005725/wireless/legacy-intel-wireless-products.html

    max_data_rate = 143 * 10 ** 6
    # max input power to transmitter allowed by FCC
    p_max = 1
    message_length = 14 * (10 ** 6)
    Energy_img_txt = 0
    Energy_txt_img = 0

    if encoder == 'vit':
        message_length = countTotalBits(sem_vitgpt2)
        Energy_img_txt = 46.939
        Energy_txt_img = 4489.725
    if encoder == 'gitbase':
        message_length = countTotalBits(sem_gitbase)
        Energy_img_txt = 57.104
        Energy_txt_img = 3840.526
    if encoder == 'blipbase':
        message_length = countTotalBits(sem_blipbase)
        Energy_img_txt = 35.677
        Energy_txt_img = 3903.716
    if encoder == 'gitlarge':
        message_length = countTotalBits(sem_gitlarge)
        Energy_img_txt = 244.787
        Energy_txt_img = 4027.65
    if encoder == 'bliplarge':
        message_length = countTotalBits(sem_bliplarge)
        Energy_img_txt = 95.009
        Energy_txt_img = 3829.253

    t_ = message_length / max_data_rate
    Energy_ch = t_ * p_max

    if encoder is None:
        Energy_ = Energy_ch
    else:
        if include_diffusion_energy:
            Energy_ = Energy_ch + Energy_img_txt + Energy_txt_img
        else:
            Energy_ = Energy_ch + Energy_img_txt

    return Energy_


def eosl_loss(pb, pf, k, semantic=None, enc_type=None, sim_output=None):
    Mp_img = None
    Mp_txt = ''
    imgs_path = '/Users/nemo/Desktop/nlp_project/venv/others_na/dnt_delete_diffusers_image/'
    l_class = 14 * (10 ** 6)
    if enc_type == 'vit':
        Mp_img = os.path.join(imgs_path, 'vitgpt2_stable_diff.jpg')
        Mp_txt = sem_vitgpt2
    if enc_type == 'gitlarge':
        Mp_img = os.path.join(imgs_path, 'gitlarge_stable_diff.jpg')
        Mp_txt = sem_gitlarge
    if enc_type == 'gitbase':
        Mp_img = os.path.join(imgs_path, 'gitbase_stable_diff.jpg')
        Mp_txt = sem_gitbase
    if enc_type == 'blipbase':
        Mp_img = os.path.join(imgs_path, 'blipbase_stable_diff.jpg')
        Mp_txt = sem_blipbase
    if enc_type == 'bliplarge':
        Mp_img = os.path.join(imgs_path, 'bliplarge_stable_diff.jpg')
        Mp_txt = sem_bliplarge

    # pb - probability of bit error rate, pf - probability of deep fade
    pb_bar = 0.5 * pf + pb * (1 - pf)  # avg probability of bit error rate

    if semantic is None:
        L_ch = 1 - (1 - pb_bar) ** l_class
        # eff_energy_penalty = k * (math.log10(energy_penalty()))
        eff_energy_penalty = k * (energy_penalty())
        N_sm = 0
    else:
        l_sem = countTotalBits(enc_type)
        L_ch = 1 - (1 - pb_bar) ** l_sem

        if sim_output is not None:
            Mi = 'browndog.jpg'
            Mp = Mp_img
            N_sm = 1 - img_similarity(Mi, Mp)
        else:
            Mi = 'a dog running through green grass'
            Mp = Mp_txt
            N_sm = 1 - txt_similarity(Mi, Mp)

        if sim_output:
            # eff_energy_penalty = k * (math.log10(energy_penalty(enc_type, 1)))
            eff_energy_penalty = k * (energy_penalty(enc_type, 1))
        else:
            # eff_energy_penalty = k * (math.log10(energy_penalty(enc_type)))
            eff_energy_penalty = k * (energy_penalty(enc_type))

    total_loss = N_sm + L_ch + eff_energy_penalty
    # total_loss = math.exp(k1*N_sm) + math.exp(k2*L_ch) + eff_energy_penalty

    # return enc_type, total_loss, N_sm, L_ch, eff_energy_penalty
    return total_loss


if __name__ == '__main__':
    eosl_loss()