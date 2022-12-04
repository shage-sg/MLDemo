from PIL import Image
import sys
image = Image.open("E:\OneDrive\桌面\Snipaste_2022-12-05_10-41-27.png")

basewidth = 1920
img = image
wpercent = (basewidth / float(img.size[0]))
hsize = 1080
img = img.resize((basewidth, hsize), Image.BICUBIC)
img.save('sompic1.jpg')

print("image to %s" % (str(img.size)))
