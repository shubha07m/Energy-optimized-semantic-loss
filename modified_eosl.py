import os.path
from eosl_helpers import img_similarity, txt_similarity, calculate_Lch, energy_penalty, comm_energy, countTotalBits

sem_vitgpt2 = 'a large brown dog chasing a brown grassy field'
sem_gitlarge = 'image of dog running across the grass with dog running toward it'
sem_gitbase = 'running dog, a golden retriever in a grassy field.'
sem_blipbase = 'a small golden golden retrieve playing on a beautiful, green lawn, as a blurred green background'
sem_bliplarge = 'aboard a dog running through the grass by a tree line'


def eosl_loss(enc_type, pb=.001, lambda_es=1, lambda_sm=1, lambda_lch=1, lambda_ec=1, sim_type='txt'):
    Mi_img = 'browndog.jpg'
    Mi_txt = 'a brown dog running through grassy field'
    diff_imgs_path = os.path.join(os.getcwd(), 'last_diffusion_images/')

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
        N_sm = 1 - img_similarity(Mi_img, Mp_img, sim_type=1, rmv_bg=0)
        es = energy_penalty(enc_type, 1)

    EOSL = lambda_sm * N_sm + lambda_lch * L_ch + lambda_ec * ec + lambda_es * es
    return EOSL


if __name__ == '__main__':
    eosl_loss('gitbase', pb=.001, lambda_es=1, lambda_sm=1, lambda_lch=1, lambda_ec=1, sim_type='img')
    eosl_loss('vit', pb=.001, lambda_es=1, lambda_sm=1, lambda_lch=1, lambda_ec=1, sim_type='img')
    eosl_loss('blipbase', pb=.001, lambda_es=1, lambda_sm=1, lambda_lch=1, lambda_ec=1, sim_type='img')
    eosl_loss('gitlarge', pb=.001, lambda_es=1, lambda_sm=1, lambda_lch=1, lambda_ec=1, sim_type='img')
    eosl_loss('bliplarge', pb=.001, lambda_es=1, lambda_sm=1, lambda_lch=1, lambda_ec=1, sim_type='img')
