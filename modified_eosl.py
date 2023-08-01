import os.path
from eosl_helpers import img_similarity, txt_similarity, calculate_Lch, energy_penalty, comm_energy, countTotalBits

sem_vitgpt2 = 'a dog running through a grassy field'
sem_gitlarge = 'a dog running through a field of tall grass'
sem_gitbase = 'a dog running in the grass'
sem_blipbase = 'a small dog running through a field'
sem_bliplarge = 'there is a dog running in the grass with a frisbee in its mouth'


def eosl_loss(enc_type, sim_type='txt', pb=.001, k=1):
    Mi_img = 'browndog.jpg'
    Mi_txt = 'a dog running through green grass'
    diff_imgs_path = os.path.join(os.getcwd(), 'dnd_diffusers_results/')

    if enc_type == 'vit':
        Mp_img = os.path.join(diff_imgs_path, 'vitgpt2_stable_diff.jpg')
        Mp_txt = sem_vitgpt2
    if enc_type == 'gitlarge':
        Mp_img = os.path.join(diff_imgs_path, 'gitlarge_stable_diff.jpg')
        Mp_txt = sem_gitlarge
    if enc_type == 'gitbase':
        Mp_img = os.path.join(diff_imgs_path, 'gitbase_stable_diff.jpg')
        Mp_txt = sem_gitbase
    if enc_type == 'blipbase':
        Mp_img = os.path.join(diff_imgs_path, 'blipbase_stable_diff.jpg')
        Mp_txt = sem_blipbase
    if enc_type == 'bliplarge':
        Mp_img = os.path.join(diff_imgs_path, 'bliplarge_stable_diff.jpg')
        Mp_txt = sem_bliplarge

    l_sem = countTotalBits(Mp_txt)

    N_sm = 1 - txt_similarity(Mi_txt, Mp_txt)
    L_ch = calculate_Lch(l_sem, pb)
    ec = comm_energy(enc_type)
    es = energy_penalty(enc_type)

    if sim_type == 'img':
        N_sm = 1 - img_similarity(Mi_img, Mp_img)
        es = energy_penalty(enc_type, 1)

    EOSL = N_sm + L_ch + ec + k * es
    return EOSL


print('eosl with semantic communication without text to image energy:')
print(round(eosl_loss('gitbase'), 3))
print(round(eosl_loss('vit'), 3))
print(round(eosl_loss('blipbase'), 3))
print(round(eosl_loss('gitlarge'), 3))
print(round(eosl_loss('bliplarge'), 3))

print('eosl with semantic communication including text to image energy:')
print(round(eosl_loss('gitbase', 'img'), 3))
print(round(eosl_loss('vit', 'img'), 3))
print(round(eosl_loss('blipbase', 'img'), 3))
print(round(eosl_loss('gitlarge', 'img'), 3))
print(round(eosl_loss('bliplarge', 'img'), 3))
