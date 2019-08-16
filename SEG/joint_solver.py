import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.autograd import Variable
from networks.joint_poolnet import build_model, weights_init
import numpy as np
import os
import cv2
import time
mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'
class Solver(object):
    def __init__(self, train_loader, test_loader):
        self.cuda = True
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model =mainroot+ '/results/run-1/models/final.pth'
        self.iter_size = 10
        self.show_every = 50
        self.lr_decay_epoch = [8,]
        self.build_model()
        print('Loading pre-trained model from %s...' % self.model)
        if self.cuda:
            self.net.load_state_dict(torch.load(self.model))
        else:
            self.net.load_state_dict(torch.load(self.model, map_location='cpu'))
        self.net.eval()
    # build the network
    def build_model(self):
        self.net = build_model()
        if self.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
      #  self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        self.lr = 5e-5
        self.wd = 0.0005
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        #self.print_network(self.net, 'PoolNet Structure')

    def test(self, test_mode=1):
        self.test_fold=mainroot + '/mask'
        mode_name = ['edge_fuse', 'sal_fuse']
        EPSILON = 1e-8
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            if test_mode == 0:
                images = images.numpy()[0].transpose((1,2,0))
                scale = [0.5, 1, 1.5, 2] # uncomment for multi-scale testing
                # scale = [1]
                multi_fuse = np.zeros(im_size, np.float32)
                for k in range(0, len(scale)):
                    im_ = cv2.resize(images, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    im_ = im_.transpose((2, 0, 1))
                    im_ = torch.Tensor(im_[np.newaxis, ...])

                    with torch.no_grad():
                        im_ = Variable(im_)
                        if self.config.cuda:
                            im_ = im_.cuda()
                        preds = self.net(im_, mode=test_mode)
                        pred_0 = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
                        pred_1 = np.squeeze(torch.sigmoid(preds[1][1]).cpu().data.numpy())
                        pred_2 = np.squeeze(torch.sigmoid(preds[1][2]).cpu().data.numpy())
                        pred_fuse = np.squeeze(torch.sigmoid(preds[0]).cpu().data.numpy())

                        pred = (pred_0 + pred_1 + pred_2 + pred_fuse) / 4
                        pred = (pred - np.min(pred) + EPSILON) / (np.max(pred) - np.min(pred) + EPSILON)

                        pred = cv2.resize(pred, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)
                        multi_fuse += pred

                multi_fuse /= len(scale)
                multi_fuse = 255 * (1 - multi_fuse)
                cv2.imwrite(os.path.join(self.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)
            elif test_mode == 1:
                with torch.no_grad():
                    images = Variable(images)
                    if self.cuda:
                        images = images.cuda()
                    preds = self.net(images, mode=test_mode)
                    pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                    multi_fuse = 255 * pred
                    #改一下文件名
                    #cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)
                    cv2.imwrite(os.path.join(self.test_fold,name.split('.')[0]  + '.png'), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')
def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

