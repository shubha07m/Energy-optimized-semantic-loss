import time
from transformers import VisionEncoderDecoderModel, AutoModelForCausalLM, BlipModel

t = time.time()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


# VIT GPT2 model #
vit_gpt2 = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# GIT large model #
git_large = AutoModelForCausalLM.from_pretrained("microsoft/git-large")

# GIT base model #
git_base = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# BLIP base model #
blip_base = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")

# BLIP large model #
blip_large = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large")

models = [vit_gpt2, git_large, git_base, blip_base, blip_large]

model_names = ['vit_gpt2', 'git_large', 'git_base', 'blip_base', 'blip_large']

for i in range(5):
    print('number of parameter in the ' + model_names[i] + ' is: %d' % count_parameters(models[i]))
    print('size of the ' + model_names[i] + ' is: %d' % count_size(models[i]))
    print('\n\n')
