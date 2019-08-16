import os
import shutil
import cv2
from joint_main import main

import os 
mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'

def koutu(pathin,pathout):
        imgbig = cv2.imread(pathin)
        imgres = cv2.resize(imgbig, (400, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(mainroot + '/a1.jpg', imgres)
        main()
        mpath = mainroot + '/mask/'
        imgmask0 = cv2.imread(mpath + 'a1.png')

        imgmask = cv2.resize(imgmask0, (imgbig.shape[1], imgbig.shape[0]), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(imgmask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        imgmasked = cv2.copyTo(imgbig, imgmask)
        b_channel, g_channel, r_channel = cv2.split(imgmasked)
        img1 = cv2.merge((b_channel[y:y + h, x:x + w], g_channel[y:y + h, x:x + w], r_channel[y:y + h, x:x + w], mask[y:y + h, x:x + w]))
        cv2.imwrite(pathout, img1)
        os.remove(mainroot + '/a1.jpg')
        # shutil.rmtree(mpath)
if __name__ == '__main__':
        pathin = './1.jpg'
        pathout = './22.png'
        koutu(pathin, pathout)
