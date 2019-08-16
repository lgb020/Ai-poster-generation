import random
import imagelayer as il
import cv2
from PIL import ImageFont, ImageDraw, Image
from random import choice
from math import ceil
import os
mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'
def template1(bgPath ,fgPath,txt) :
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    #生成后的背景大小
    hi, wi, channeli = bgImg.shape
    #前景大小
    hf, wf, channelf = fgImg.shape

    #选择合适的前景大小
    fg_hsize=min(0.666*wi*hf/wf,0.666*hi)

    img = il.imgadd(fgImg, bgImg, [0.6, 0.5], round(fg_hsize), 1080)#前景、背景、前景位置、前景大小、背景大小

    #生成文字的图像
    ROI_txt = img[int(hi / 6):int(1*hi/3), int(wi / 8):int(7*wi/8)]
    txtImage(txt, "horizontal", ROI_txt)
    pos = [[0.5, 0.15, 0], [0.5, 0.8, 0], [0.952, 0.95, 0]]

    print("template 1")
    return img, pos

def template2(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    hi, wi, channeli = bgImg.shape
    # 前景大小
    hf, wf, channelf = fgImg.shape

    # 选择合适的前景大小
    fg_hsize = min(0.666 * wi * hf / wf, 0.666 * hi)

    img = il.imgadd(fgImg, bgImg, [0.8, 0.5], round(fg_hsize), 1080)


    #生成文字的图像和位置
    ROI_txt = img[int(hi / 6):int(1*hi/3), int(wi / 8):int(7 * wi / 8)]
    txtImage(txt,"horizontal",ROI_txt)
    pos = [[0.5, 0.2, 0], [0.5, 0.4, 0], [0.05, 0.95, 0]]

    print("template 2")
    return img, pos

def template3(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape

    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    fgImg = il.img_circle(fgImg)

    img = il.imgadd(fgImg, bgImg, [0.5, 0.5], int(1 * 1080 / 2), 1080)

    hi, wi, channeli = img.shape
    ROI_txt = img[int(1*hi / 9):int(8 * hi / 9), int(wi / 2):wi]

    txtImage(txt,"vertical", ROI_txt)
    pos = [[0.99, 1/9, 1], [0.1, 0.5, 0], [0.5, 0.95, 0]]

    print("template 3")
    return img, pos

def template4(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    hi, wi, channeli = bgImg.shape
    # 前景大小
    hf, wf, channelf = fgImg.shape

    # 选择合适的前景大小
    fg_hsize = min(0.666 * wi * hf / wf, 0.666 * hi)
    img = il.imgadd(fgImg, bgImg, [0.65, 0.5], round(fg_hsize), 1080)

    ROI_txt = img[0:int(hi/3), int(wi / 9):int(8 * wi / 9)]
    txtImage(txt, "horizontal", ROI_txt)

    txtpos = [[0.5, 0.1346, 0], [0.5, 0.3, 0], [0.95, 0.95, 0]]

    print("template 4")
    return img, txtpos

def template5(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    hi, wi, channeli = bgImg.shape
    hf, wf, channelf = fgImg.shape
    fg_hsize = min(0.7 * wi * hf / wf, 0.7 * hi)
    img = il.imgadd(fgImg, bgImg, [0.6, 0.3], round(fg_hsize), 1080)

    ROI_txt = img[int(hi / 2):hi, int(wi / 2):wi]
    txtImage(txt, "horizontal", ROI_txt)
    txtpos = [[0.9, 0.1, 1], [1, 0.6, 1], [0.1, 0.1, 0]]
    print("template 5")
    return img, txtpos

def template6(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h/w<7/5:
        bgImg = bgImg[0:h,int(w/2-h*5.0/14.0):int(w/2+h*5.0/14.0)]#crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    else:
        bgImg = cv2.resize(bgImg,(774,1080), interpolation=cv2.INTER_AREA) # crop background

    hi, wi, channeli = bgImg.shape
    hf, wf, channelf = fgImg.shape

    # 选择合适的前景大小
    fg_hsize = min(0.8 * wi * hf / wf, 0.8 * hi)
    img = il.imgadd(fgImg, bgImg, [0.5, 0.5], int(fg_hsize), 1080)
    ROI_txt = img[0:int(hi / 3), 0:int(wi / 2)]
    txtImage(txt, "horizontal", ROI_txt)
    txtpos = [[0.05, 0.05, -1], [0.05, 0.8, -1], [0.952, 0.95, 0]]
    print("template 6")
    return img, txtpos

def template7(bgPath, fgPath, txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    hi, wi, channeli = bgImg.shape
    hf, wf, channelf = fgImg.shape

    # 选择合适的前景大小
    fg_hsize = min(0.8 * wi * hf / wf, 0.8 * hi)
    img = il.imgadd(fgImg, bgImg, [0.5, 0.5], int(fg_hsize), 1080)
    ROI_txt = img[0:int(hi / 2), int(wi / 2):int(wi)]
    txtImage(txt, "horizontal", ROI_txt)
    txtpos = [[0.95, 0.05, 1], [0.95, 0.8, 1], [0.048, 0.95, 0]]
    print("template 7")
    return img, txtpos

def template8(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape

    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    fgImg = il.img_circle(fgImg)

    img = il.imgadd(fgImg, bgImg, [0.5, 0.5], int(1 * 1080 / 2), 1080)

    hi, wi, channeli = img.shape
    #第一段文字
    ROI_txt = img[int(1*hi / 9):int(8 * hi / 9), int(wi / 2):wi]
    txtImage([txt[0],''],"vertical", ROI_txt)
    #第二段文字
    ROI_txt = img[int(3 * hi / 4):hi, 0 : int(wi / 2)]
    txtImage(['',txt[1]], "horizontal", ROI_txt)

    pos = [[0.99, 1/9, 1], [0.05, 3/4, -1], [0.5, 0.95, 0]]

    print("template 8")
    return img, pos

def template9(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    hi, wi, channeli = bgImg.shape
    # 前景大小
    hf, wf, channelf = fgImg.shape

    # 选择合适的前景大小
    fg_hsize = min(0.666 * wi * hf / wf, 0.666 * hi)
    img = il.imgadd(fgImg, bgImg, [0.5, 0.5], round(fg_hsize), 1080)

    ROI_txt = img[0:int(hi/3), int(wi / 9):int(8 * wi / 9)]
    txtImage(txt, "horizontal", ROI_txt)

    txtpos = [[0.5, 0.1346, 0], [0.5, 0.9, 0], [0.95, 0.95, 0]]

    print("template 9")
    return img, txtpos

def template10(bgPath ,fgPath,txt):
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = bgImg.shape
    if h / w < 7 / 5:
        bgImg = bgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        bgImg = cv2.resize(bgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    hi, wi, channeli = bgImg.shape
    hf, wf, channelf = fgImg.shape
    fg_hsize = min(0.7 * wi * hf / wf, 0.7 * hi)
    img = il.imgadd(fgImg, bgImg, [0.6, 0.7], round(fg_hsize), 1080)

    ROI_txt = img[0:int(hi/3), 0:int(wi / 2)]
    txtImage(txt, "horizontal", ROI_txt)
    txtpos = [[0.05, 0.1, -1], [0.05, 0.6, -1], [0.95, 0.05, 0]]
    print("template 10")
    return img, txtpos

def people_Time_template(fgPath,txt):
    bgPath = "Time.png"
    bgImg = cv2.imread(bgPath, -1)
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = fgImg.shape
    if h / w < 7 / 5:
        fgImg = fgImg[0:h, int(w / 2 - h * 5.0 / 14.0):int(w / 2 + h * 5.0 / 14.0)]  # crop background
        fgImg = cv2.resize(fgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background
    else:
        fgImg = cv2.resize(fgImg, (774, 1080), interpolation=cv2.INTER_AREA)  # crop background

    img = il.imgadd(bgImg, fgImg, [0.5, 0.5], 1080, 1080)
    hi, wi, channeli = img.shape
    # 生成文字的图像
    ROI_txt = img[int(2*hi /3 ):hi, 0:int( wi / 2)]
    txtImage([txt,''], "horizontal", ROI_txt)
    txt_img = cv2.imread(mainroot+"txt_image/img" + str(1) + ".png", -1)
    txtpos = [[0.05, 0.7, -1]]
    img = il.imgadd(txt_img, img, [txtpos[0][1], txtpos[0][0]], txt_img.shape[0], 1080, txtpos[0][-1])

    cv2.imwrite(mainroot+"results/results_time.png",img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

def choosetemplate(bgPath,fgPath,txt,outputPath):
    fgImg = cv2.imread(fgPath, -1)
    h, w, channel = fgImg.shape
    if 5*h < 7*w :
        img, txtpos = random.choice([template1, template2, template6, template7])(bgPath, fgPath, txt)
    else:
        img, txtpos = random.choice([template4, template5, template9, template10])(bgPath, fgPath, txt)
    cv2.imwrite(outputPath,img,[int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return txtpos

def txtImage(txt,direction,img):
    folder = os.path.exists(mainroot+"txt_image")
    if not folder:
        os.makedirs(mainroot+"txt_image")

    h, w, channel = img.shape
    # choose txt_color by image background
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bg_color = il.img_color(image)

    clh,cls,clv = il.rgb2hsv(bg_color[0],bg_color[1],bg_color[2])
    clh = (clh+180) % 360
    cls = 1-cls
    clv = 1-clv

    txt_color=il.hsv2rgb(clh,cls,clv)
    #     txt_color = [255 - bg_color[0], 255 - bg_color[1], 255 - bg_color[2]]
    # put the txt img on the bg
    draw = ImageDraw.Draw(image)
    fontpath = [mainroot + "font/庞门正道标题体/庞门正道标题体2.0增强版.ttf",
                mainroot + "font/HanYiShangWeiShouShu/HYShangWeiShouShuW-1.ttf",
                mainroot + "font/ZhanKuKuaiLeTi/ZhanKuKuaiLeTi2016XiuDingBan-1.ttf",
                mainroot + "font/zhengqingke/zhengqingkehuangyou.ttf",
                mainroot + "font/siyuan/SourceHanSansSC-Bold.otf"]
    foo = choice(fontpath)

    if direction == "horizontal":
        # create the txt image
        for i in range(len(txt)):
            if txt[i] == '':
                continue
            if i == 0:
                # 绘制文字信息
                font = ImageFont.truetype(foo, 80, encoding='utf-8')
                tw, th = draw.textsize(txt[i], font=font)
                # 换行处理
                if tw > w:
                    txt_str = txt[i].splitlines()
                    for k in range(len(txt_str)):
                        strw, strh = draw.textsize(txt_str[k], font=font)
                        if strw > w:
                            count = ceil(tw / w)
                            for j in range(count-1):
                                txt_str[k] = txt_str[k][:(j+1)*ceil((len(txt_str[k])-j)/count)+j] + '\n' + \
                                         txt_str[k][(j+1)*ceil((len(txt_str[k])-j)/count)+j:]
                    txt[i] = '\n'.join(txt_str)
                tw, th = draw.textsize(txt[i], font=font)
                txt_img = Image.new("RGBA", (tw + 1, int(1.1*th)), (0, 0, 0, 0))
                draw = ImageDraw.Draw(txt_img)
                draw.text((0, 0), txt[i], font=font,
                          fill=(txt_color[0], txt_color[1], txt_color[2]))
                txt_img.save(mainroot+"txt_image/img" + str(i + 1) + ".png", "PNG")
            else:
                font = ImageFont.truetype(foo, 30, encoding='utf-8')
                tw, th = draw.textsize(txt[i], font=font)
                if tw > w:
                    txt_str = txt[i].splitlines()
                    for k in range(len(txt_str)):
                        strw, strh = draw.textsize(txt_str[k], font=font)
                        if strw > w:
                            count = ceil(strw / w)
                            for j in range(count-1):
                                txt_str[k] = txt_str[k][:(j + 1) * ceil((len(txt_str[k])-j)/ count)+j] + '\n' + \
                                            txt_str[k][(j + 1) * ceil((len(txt_str[k])-j) / count)+j:]
                    txt[i] = '\n'.join(txt_str)
                tw, th = draw.textsize(txt[i], font=font)
                txt_img = Image.new("RGBA", (tw + 1, int(1.1*th)), (0, 0, 0, 0))
                draw = ImageDraw.Draw(txt_img)
                draw.text((0, 0), txt[i], font=font,
                          fill=(txt_color[0], txt_color[1], txt_color[2]))
                txt_img.save(mainroot+"txt_image/img" + str(i + 1) + ".png", "PNG")
    elif direction == "vertical":
        for i in range(len(txt)):
            if txt[i] == '':
                continue
            if i == 0:
                right = 0  # 往右位移量
                down = 0  # 往下位移量
                row_hight = 0  # 行高设置（文字行距）
                word_dir = 0;  # 文字间距
                h_count = 0
                w_count = 1
                h_count_max = 0
                font = ImageFont.truetype(foo, 80, encoding='utf-8')
                for j, s2 in enumerate(txt[i]):
                    if j == 0:
                        ww, wh = font.getsize(s2)
                        txt_img = Image.new("RGBA", (w, h),
                                            (0, 0, 0, 0))
                        draw = ImageDraw.Draw(txt_img)
                    draw.text((right, down), s2, font=font,
                              fill=(txt_color[0], txt_color[1], txt_color[2]))  # 设置位置坐标 文字 颜色 字体
                    h_count = h_count + 1
                    h_count_max = max(h_count, h_count_max)
                    if s2 == "," or s2 == "\n" or (wh*(h_count+1) > h and j+2<=len(txt[i])):  # 换行识别
                        right = right + ww + row_hight
                        down = 0
                        w_count = w_count + 1
                        h_count = 0
                        continue
                    else:
                        down = down + wh + word_dir
                roi = txt_img.crop(box=(0, 0, ww * w_count, wh * h_count_max))
                roi.save(mainroot+"txt_image/img" + str(i + 1) + ".png", "PNG")
            else:
                right = 0  # 往右位移量
                down = 0  # 往下位移量
                row_hight = 0  # 行高设置（文字行距）
                word_dir = 0  # 文字间距
                h_count = 0
                w_count = 1
                h_count_max = 0
                font = ImageFont.truetype(foo, 30, encoding='utf-8')
                for j, s2 in enumerate(txt[i]):
                    if j == 0:
                        ww, wh = font.getsize(s2)
                        txt_img = Image.new("RGBA", (ww * len(txt[i]), wh *len(txt[i])),
                                            (0, 0, 0, 0))
                        draw = ImageDraw.Draw(txt_img)
                    draw.text((right, down), s2, font=font,
                              fill=(txt_color[0], txt_color[1], txt_color[2]))  # 设置位置坐标 文字 颜色 字体
                    h_count = h_count + 1
                    h_count_max = max(h_count, h_count_max)
                    if s2 == "," or s2 == "\n" or (wh*(h_count+1) > h and j+2<=len(txt[i])):  # 换行识别
                        right = right + ww + row_hight
                        down = 0
                        w_count = w_count + 1
                        h_count = 0
                        continue
                    else:
                        down = down + wh + word_dir
                roi = txt_img.crop(box=(0, 0, ww * w_count, wh * h_count_max))
                roi.save(mainroot+"txt_image/img" + str(i + 1) + ".png", "PNG")
    else:
        print("This txt direction "+direction+" is not defined!")

def addallimage(ImgPath,txtpos,outputPath):
    logoPath = mainroot+"logo/logo.jpg"
    img = cv2.imread(ImgPath,-1)
    logoimg = cv2.imread(logoPath,-1)
    for ti in range(len(txtpos)-1):
        txt_img_path = mainroot+"txt_image/img"+str(ti+1)+".png"
        txt_img = cv2.imread(txt_img_path, -1)
        os.remove(txt_img_path)
        txth, txtw, txtchannel = txt_img.shape
        img = il.imgadd(txt_img, img, [txtpos[ti][1], txtpos[ti][0]], txth, 1080, txtpos[ti][-1])
    img = il.imgadd(logoimg,img,[txtpos[-1][1],txtpos[-1][0]], int(1080/15), 1080)
    cv2.imwrite(outputPath, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

# if __name__=="__main__":
#     outputPath_text = 'results/results.png'
#     outputPath_notext = 'results/results_notext.png'
#     fgPath = 'foreground/WechatIMG247.png'
#     txt = [u"第十八届上海国际汽车", u"时间：4月28日\n地点：广东省广州市开源大道232号企业加速器道"]
#     bgPath = 'background/WechatIMG63.jpeg'
#     pos = choosetemplate(bgPath, fgPath, txt, outputPath_notext)
#     addallimage(outputPath_notext,pos,outputPath_text)
#     #people_Time_template("foreground/People.png","道德的沦陷还是人性的扭曲")
