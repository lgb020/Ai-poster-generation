from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import cv2
import os
from PIL import Image
import numpy as np
import torchvision.models as models
import copy
from loss import ContentLoss, GramMatrix, StyleLoss

from stutils import image_loader, image_loader_, save_image



# parser = argparse.ArgumentParser()
#parser.add_argument('--content', '-c', type=str, default='imgs/content.jpg', help='The path to the Content image')
#parser.add_argument('--style', '-s', type=str, default='styles/2.jpg', help='The path to the style image')
#parser.add_argument('--epoch', '-e', type=int, default=300, help='The number of epoch')
#parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
#parser.add_argument('--style_weight', '-s_w', type=int, default=3000, help='The weight of style loss')
# parser.add_argument('--initialize_noise', '-i_n', action='store_true',help='Initialize with white noise? elif initialize with content image')
#parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
# args = parser.parse_args()




def get_style_model_and_losses(cnn, style_img, content_img, style_weight=1000, content_weight=1,
                               content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],cuda=False):
    cnn = copy.deepcopy(cnn)
    content_layers=['conv_4']
    style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses


def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, style_weight=1000, content_weight=1,cuda=False):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, style_weight,
                                                                     content_weight,cuda)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)

            # stylize_img = np.transpose(input_param.clone().data[0, :, :, :].cpu().numpy(), (1, 2, 0))
            # stylize_img = cv2.cvtColor(stylize_img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('img', stylize_img)
            # cv2.waitKey(1)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data
def onlinestyle(pathin,pathst,pathout,epoch,cuda):
    content_weight = 1
    style_weight = 3000

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # desired size of the output image
    #imsize = 512 if use_cuda else 128  # use small size if no gpu
    img = cv2.imread(pathin)
    print(img.shape)
    style_size = (img.shape[0], img.shape[1])
    style_img = image_loader(pathst, style_size).type(dtype)
    content_img = image_loader_(pathin).type(dtype)
    # if args.initialize_noise:
    # input_img = Variable(torch.randn(content_img.data.size())).type(dtype)
    # else:
    input_img = image_loader_(pathin).type(dtype)
    input_size = Image.open(pathin).size
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    cnn = models.vgg19(pretrained=True).features
    # move it to the GPU if possible:
    if cuda:
        cnn = cnn.cuda()
    # desired depth layers to compute style/content losses :
    # content_layers_default = ['conv_4']
    # style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    output = run_style_transfer(cnn, content_img, style_img, input_img, epoch, style_weight, content_weight,cuda)
    img = np.transpose(output[0, :, :, :].cpu().numpy(), (1, 2, 0))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    save_image(output, size=input_img.data.size()[1:], input_size=input_size, fname=pathout)

if __name__ == '__main__':
    pathin = 'imgs/content.jpg'
    pathst = 'styles/2.jpg'
    pathout = 'transferred/output1.jpg'
    epoch = 3
    cuda = True
    if not os.path.exists('transferred'):
        os.mkdir('transferred')
    use_cuda = torch.cuda.is_available() and cuda
    onlinestyle(pathin, pathst, pathout, epoch,cuda)



