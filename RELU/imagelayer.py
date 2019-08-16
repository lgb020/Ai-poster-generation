import cv2
import numpy as np
import colorsys
import PIL.Image as Image
import math

def imgadd(fgImg, bgImg, fgPosRatio, fgSize, bgSize=1080, dir=0, MidImg = None):
    #read image
    img_1=fgImg
    img_2=bgImg

    if img_1.shape[2] == 3:
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2BGRA)
    if img_2.shape[2] == 3:
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2BGRA)


    #image normalizatiom
    img_1_resize=cv2.resize(img_1, (int(img_1.shape[1]*fgSize/img_1.shape[0]), fgSize), interpolation=cv2.INTER_AREA)
    img_2_resize=cv2.resize(img_2, (int(img_2.shape[1]*bgSize/img_2.shape[0]), bgSize), interpolation=cv2.INTER_AREA)

    rows, cols, channels = img_1_resize.shape
    #Pos ratio convert into Pos
    if dir == 0:
        fgPos = [bgSize * fgPosRatio[0] - int(0.5 * rows), int(img_2.shape[1]*bgSize/img_2.shape[0]) * fgPosRatio[1] - int(0.5 * cols)]
    elif dir == 1:
        fgPos = [bgSize * fgPosRatio[0], int(img_2.shape[1]*bgSize/img_2.shape[0]) * fgPosRatio[1] - cols]
    else:
        fgPos = [bgSize * fgPosRatio[0], int(img_2.shape[1]*bgSize/img_2.shape[0]) * fgPosRatio[1]]

    # img_text_pos=[bgSize*0.75, bgSize*bgRatio*0.8485]
    #create a ROI

    #fgPos[0],点的行位置，
    #fgPos[1]点的列位置，img_2_resize.shape[0]背景图片的长度，[1]宽度，cols指的是前景的长，rows指的是前景的宽
    #前景图片超出背景图片的右部分
    if fgPos[1]+cols >= img_2_resize.shape[1] and fgPos[0]+rows <= img_2_resize.shape[0]:
        fgPos[1] = img_2_resize.shape[1]-cols

    #前景图片超出背景图片的左部分
    if fgPos[1]< 0 and fgPos[0]+rows <= img_2_resize.shape[0]:
        fgPos[1] = max(0,fgPos[1])

    #当前景图片超出背景图片下部分
    if fgPos[0]+rows >= img_2_resize.shape[0] and fgPos[1]+cols <= img_2_resize.shape[1]:
        fgPos[0] = img_2_resize.shape[0]-rows




    #当前景照片超出背景图片的上部分，一般不可能发生
    # roi = img_2_resize[max(int(fgPos[0]), 0):min(int(fgPos[0]+rows), img_2_resize.shape[0]-1), max(int(fgPos[1]), 0):min(int(fgPos[1]+cols), img_2_resize.shape[1]-1)]

    roi = img_2_resize[max(int(fgPos[0]),0):max(int(fgPos[0]),0) + rows, max(int(fgPos[1]),0):max(int(fgPos[1] ),0)+ cols]
    if MidImg is not None:
        roi = addWeightedImage(roi,MidImg,0.3)

    #create a mask of logo and create its inverse mask
    if channels > 3:
        b,g,r,alpha = cv2.split(img_1_resize)
        img_1_resize = cv2.merge([b,g,r])
        mask = alpha
        mask_inv = cv2.bitwise_not(mask)
        # mask_final= tuple()
        # black-out the area in ROI
        # print('mask_inv',mask_inv)

        # mask_final = mask_inv.resize(min(roi.shape[0],mask_inv.shape[0]),min(roi.shape(1),mask_inv.shape(1)))
        img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only region of foreground from foreground image.
        img1_fg = cv2.bitwise_and(img_1_resize, img_1_resize, mask=mask)
        # Put logo in ROI and modify the main image

        if img2_bg.shape[2] == 3:
            img2_bg = cv2.cvtColor(img2_bg, cv2.COLOR_BGR2BGRA)
        if img1_fg.shape[2] == 3:
            img1_fg = cv2.cvtColor(img1_fg, cv2.COLOR_BGR2BGRA)

        dst = cv2.add(img2_bg, img1_fg)
    else:
        dst = cv2.addWeighted(img_1_resize,1,roi,0,0)
    img_2_resize[max(int(fgPos[0]), 0):max(int(fgPos[0]), 0) + rows, max(int(fgPos[1]),0):max(int(fgPos[1]),0) + cols]=dst
    # img_2_resize[int(fgPos[0]):int(fgPos[0]+rows), int(fgPos[1]):int(fgPos[1]+cols)] = dst
    return img_2_resize

#不同透明度图片叠加（还需要考虑png图像）
def addWeightedImage(foreimg, bgimg,alpha):
    h, w, channel = foreimg.shape
    img2 = cv2.resize(bgimg, (w,h), interpolation=cv2.INTER_AREA)
    #print img1.shape, img2.shape
    #alpha，beta，gamma可调
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(foreimg, alpha, img2, beta, gamma)
    return img_add


def img_circle(img):
    # cv2.IMREAD_COLOR，读取BGR通道数值，即彩色通道，该参数为函数默认值
    # cv2.IMREAD_UNCHANGED，读取透明（alpha）通道数值
    # cv2.IMREAD_ANYDEPTH，读取灰色图，返回矩阵是两维的
    rows, cols, channel = img.shape

    # 创建一张4通道的新图片，包含透明通道，初始化是透明的
    img_new = np.zeros((rows,cols,4),np.uint8)
    img_new[:,:,0:3] = img[:,:,0:3]

    # 创建一张单通道的图片，设置最大内接圆为不透明，注意圆心的坐标设置，cols是x坐标，rows是y坐标
    img_circle = np.zeros((rows,cols,1),np.uint8)
    img_circle[:,:,:] = 0  # 设置为全透明
    img_circle = cv2.circle(img_circle,(int(cols/2),int(rows/2)),int(min(rows, cols)/2),(255),-1) # 设置最大内接圆为不透明

    # 图片融合
    img_new[:,:,3] = img_circle[:,:,0]
    img_final=img_new[int(rows/2-min(rows, cols)/2):int(rows/2+min(rows, cols)/2),int(cols/2-min(rows, cols)/2):int(cols/2+min(rows, cols)/2)]

    # 保存图片
    return img_final

def img_color(image):
    image = image.convert('RGBA')
    image.thumbnail((200, 200))
    max_score = 0.0001
    dominant_color = None
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        if a == 0:
            continue
        # 转为HSV标准
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        # y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        # y = (y - 16.0) / (235 - 16)
        #
        # # 忽略高亮色
        # if y > 0.9:
        #     continue
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    return dominant_color

def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c

def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))

def complement_image(iname, oname):
    img = Image.open(iname)
    #img.show()

    size = img.size
    mode = img.mode
    in_data = img.getdata()

    out_img = Image.new(mode, size)
    out_img.putdata([complement(*rgb) for rgb in in_data])
    out_img.save(oname)

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v
