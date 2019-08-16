import argparse

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from model import Generator
import os
mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'
TEST_MODE = False


def generateSR(image_path, save_path, upscale_factor):
    r"""
    :param: image_path: path of used image
                save_path: path of saved image
                upscale_factor: which model do you want to use. format :int
    """

    model = Generator(upscale_factor).eval()
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load(mainroot + 'SR_x' + str(upscale_factor) + '.pth'))
    else:
        model.load_state_dict(
            torch.load(mainroot + 'SR_x' + str(upscale_factor) + '.pth', map_location=lambda storage, loc: storage))

    image = Image.open(image_path)
    # image = Variable(ToTensor()(image), volatile=True).unsqueeze(0).cuda()
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(save_path)

if __name__ == "__main__":
    generateSR('1.jpg','1.jpg',2)