from captcha.image import ImageCaptcha


#设置验证码图片的宽度widht和高度height
width,height=200,50
#除此之外还可以设置字体fonts和字体大小font_sizes
generator=ImageCaptcha(width=width,height=height)

#生成4个字符的字符串
random_str = 'abcd'
#生成验证码
img=generator.generate_image(random_str)
# print(np.array(img).shape)
img.save('./parise/'+random_str+'.png')



