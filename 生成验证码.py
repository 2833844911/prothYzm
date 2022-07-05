from captcha.image import ImageCaptcha
import string
import random
from tqdm import tqdm

#设置验证码图片的宽度widht和高度height
width,height=200,50
#除此之外还可以设置字体fonts和字体大小font_sizes
generator=ImageCaptcha(width=width,height=height)

def getYzm(random_str,name):
    #random_str生成验证码的字符串

    #生成验证码
    img=generator.generate_image(random_str)
    # print(np.array(img).shape)
    img.save('./parise/'+name+'.png')


if __name__ == '__main__':
    # strAll设置生成的字符串
    strAll = list(string.ascii_lowercase+string.ascii_uppercase+'0123456789')
    # 生成验证码长度范围
    fanwei = [3,6]
    # 生成验证码张数
    numb = 60000
    for _ in tqdm(range(numb)):
        yzm = ''
        for i in range(random.randint(*fanwei)):
            yzm += random.choice(strAll)
        yzm += '_'+''.join([random.choice(strAll) for i in range(20)])
        getYzm(yzm.split('_')[0],yzm)
