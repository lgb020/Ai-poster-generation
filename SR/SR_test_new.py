import argparse

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from model import Generator
import os
mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'

upscale_factor = 2
sr_model = Generator(upscale_factor).eval()
sr_model.cuda()
sr_model.load_state_dict(torch.load(mainroot + 'SR_x' + str(upscale_factor) + '.pth'))


def generateSR(image_path, save_path, model):
    r"""
    :param: image_path: path of used image
                save_path: path of saved image
                upscale_factor: which model do you want to use. format :int
    """

    image = Image.open(image_path)
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0).cuda()
    #   image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(save_path)

if __name__ == "__main__":
    # mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'
    LR_path = mainroot + '1.jpg'
    SR_path = mainroot + '2.jpg'
    generateSR(LR_path, SR_path, model=sr_model)
