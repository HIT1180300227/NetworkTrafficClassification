
import numpy as np
from PIL import Image
import os

#全局变量，显示生成多少图片
num=0

#每调用一次genpic，生成一张对应4s的png灰度图像
def genpic(data,start):

  #当前像素点的横坐标
  xcount = 0

  flag=0

  #灰度图像：注意pic.T才是用于卷积的图像
  pic = np.zeros(shape=(100, 1500))

  for packet in data:

    unit=packet.split(' ')

    #time of arrival
    time=float(unit[0])

    #ip length
    len=int(unit[1])

    if time<start:
        continue
    flag=1
    xcount = int((time-start) / 0.04)

    if len > 1499:
        len = 1499

    #完成一幅图片：窗口滑动15秒
    if xcount>99:
       #生成本次图片：
       pic = pic.T
       pic = 255 - pic
       im = Image.fromarray(pic)
       im = im.convert('L')
       global num
       num+=1
       im.save('image_100px/data/sftp_'+str(num)+'.png')
       #print('已生成'+str(num)+'张图片')
       #生成下一张图片
       #genpic(data,start+1.5)
       return flag

    pic[xcount][len] = 255

#基本功能同genpic,filename是对应的txt文件,dicname为储存路径
def genpic2(data,start,dicname,filename):

    #当前像素点的横坐标
    xcount = 0

    flag = 0

    # 灰度图像：注意pic.T才是用于卷积的图像
    pic = np.zeros(shape=(100, 1500))

    for packet in data:

        unit = packet.split(' ')

        # time of arrival
        time = float(unit[0])

        # ip length
        len = int(unit[1])

        if time < start:
            continue
        flag = 1
        xcount = int((time - start) / 0.04)

        if len > 1499:
            len = 1499

        # 完成一幅图片：窗口滑动15秒
        if xcount > 99:
            # 生成本次图片：
            pic = pic.T
            pic = 255 - pic
            im = Image.fromarray(pic)
            im = im.convert('L')
            global num
            num += 1
            im.save(dicname+filename+'/'+filename+'_'+ str(num) + '.png')
            # print('已生成'+str(num)+'张图片')
            # 生成下一张图片
            # genpic(data,start+1.5)
            return flag

        pic[xcount][len] = 255


if __name__ == '__main__':

  path=''
  for filename in os.listdir(path):

    #打开文件，读取时间和ip包长度
    f = open(path+filename,'r')
    data=f.readlines()
    index=0
    while True:
        myflag=genpic2(data,index,'','facebook_audio')
        if myflag==1:
            index+=1.5
        else:
            break
    print('共生成'+str(num)+'张图片')