import os
import glob
import Image, ImageDraw
import PIL
import scipy.misc
import numpy as np
from skimage import io, img_as_float
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

def sem_labels(y):
    
    if y==0:
        label = 'airplane'
    
    elif y==1:
        label = 'automobile'
        
    elif y==2:
        label = 'bird'
        
    elif y==3:
        label = 'cat'
        
    elif y==4:
        label = 'deer'
        
    elif y==5:
        label = 'dog'
        
    elif y==6:
        label = 'frog'
        
    elif y==7:
        label = 'horse'
        
    elif y==8:
        label = 'ship'
        
    elif y==9:
        label = 'truck'
        
    return label
        
    


def diff (desired=list(), predicted=list()):
    
    wrong = list()
    if(desired.__len__() != predicted.__len__()):
        print("different size")
    else:
        for i in range(len(desired)):
            if(desired[i] != predicted[i]):
                wrong.append(i)
    return wrong

def wrong_mnist(wrong, cible, data):  # save mnist exemple into image file
    
    if data == "train":
        path = "/home/skyolia/STS_SPACE/cifar10_app/train_wrong/"
        
    elif data == "test":
        path = "/home/skyolia/STS_SPACE/cifar10_app/test_wrong/"
        
    files = glob.glob(path+"*")
    for f in files:
        os.remove(f)
    for i in range(len(wrong)):
        temp = cible[wrong[i]]
        temp = temp.reshape(32,32,3)
        #temp = temp * 255
        fullpath = os.path.join(path, str(wrong[i]) + ".png")
        scipy.misc.imsave(fullpath, temp)


def touch(data):  # resize mnist exemple to 100*100
    
    if data == "train":
        path = "/home/skyolia/STS_SPACE/cifar10_app/train_wrong/"
        path2 = "/home/skyolia/STS_SPACE/cifar10_app/resized_train_wrong/"
        
    elif data == "test":
        path = "/home/skyolia/STS_SPACE/cifar10_app/test_wrong/"
        path2 = "/home/skyolia/STS_SPACE/cifar10_app/resized_test_wrong/"
        
    elif data == "filter":
        path = "/home/skyolia/STS_SPACE/cifar10_app/filters/"
        path2 = "/home/skyolia/STS_SPACE/cifar10_app/filters_resized/"
    
    files = glob.glob(path2+"*")
    for f in files:
        os.remove(f)
    l = list()
    for file in sorted(os.listdir(path)):
        if file.endswith(".png"):
            l.append(file)        

    lsorted = sorted(l, key=lambda x: int(os.path.splitext(x)[0]))
    #print(lsorted)
    if data != "filter":
        for i in range(len(lsorted)):
            old_im = Image.open(path + str(lsorted[i]))
            old_size = old_im.size
    
            new_size = (100, 100)
            new_im = Image.new("RGB", new_size)  # # luckily, this is already black!
            new_im.paste(old_im, ((new_size[0] - old_size[0]) / 2, 0))
            fullpath = os.path.join(path2, lsorted[i])
            new_im.save(fullpath)
    else :
        for i in range(len(lsorted)):
            old_im = Image.open(path + str(lsorted[i]))
            old_size = old_im.size
            image = old_im.resize((100,100), PIL.Image.ANTIALIAS)
            fullpath = os.path.join(path2, lsorted[i])
            image.save(fullpath)
        
    return lsorted,path2

def write(data, source, tab=list(), desired=list(), predicted=list(), wrong=list()):  # put desired & predicted labels
    
    if data == "train":
        path = "/home/skyolia/STS_SPACE/cifar10_app/marked_train_wrong/"
        
    elif data == "test":
        path = "/home/skyolia/STS_SPACE/cifar10_app/marked_test_wrong/"
        
    files = glob.glob(path+"*")
    for f in files:
        os.remove(f)

# Note the following line works on Ubuntu 12.04
# On other operating systems you should set the correct path
# To the font you want to use.
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 16)

# Opening the file gg.png
    for i in range(len(tab)):
        courant = wrong[i]
        imageFile = tab[i]
        im1 = Image.open(source + str(tab[i]))

# Drawing the text on the picture
        draw = ImageDraw.Draw(im1)
        desired_label = " " + str(desired[wrong[i]])
        predicted_labe = " " + str(predicted[wrong[i]])
        draw.text((6, 40), "desired :" + desired_label, (255, 255, 255), font=font)
        draw.text((6, 60), "predict :" + predicted_labe, (255, 255, 255), font=font)
        draw = ImageDraw.Draw(im1)
        draw.rectangle((0, 90, 100, 100), fill="white")
        draw.rectangle((90, 0, 100, 100), fill="white")
        del draw

# Save the image with a new name
        fullpath = os.path.join(path, tab[i])
        im1.save(fullpath)
    return path
'''
def write_filter(data, source, tab=list(), desired=list(), predicted=list(), wrong=list()):  # put desired & predicted labels
    
    if data == "train":
        path = "/home/skyolia/STS_SPACE/lil_brain//marked_train_wrong/"
        
    elif data == "test":
        path = "/home/skyolia/STS_SPACE/lil_brain/marked_test_wrong/"
        
    elif data == "validation":
        path = "/home/skyolia/STS_SPACE/lil_brain/marked_validation_wrong/"
        
    files = glob.glob(path+"*")
    for f in files:
        os.remove(f)

# Note the following line works on Ubuntu 12.04
# On other operating systems you should set the correct path
# To the font you want to use.
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 16)

# Opening the file gg.png
    for i in range(len(tab)):
        courant = wrong[i]
        imageFile = tab[i]
        im1 = Image.open(source + str(tab[i]))

# Drawing the text on the picture
        draw = ImageDraw.Draw(im1)
        desired_label = " " + str(desired[wrong[i]])
        predicted_labe = " " + str(predicted[wrong[i]])
        draw.text((6, 40), "desired :" + desired_label, (255, 255, 255), font=font)
        draw.text((6, 60), "predict :" + predicted_labe, (255, 255, 255), font=font)
        draw = ImageDraw.Draw(im1)
        draw.rectangle((0, 90, 100, 100), fill="white")
        draw.rectangle((90, 0, 100, 100), fill="white")
        del draw

# Save the image with a new name
        fullpath = os.path.join(path, tab[i])
        im1.save(fullpath)
    return path
'''

def combine(source, data):  # save multiple image into one bigger image
    
    if data == "train":
        path = "/home/skyolia/STS_SPACE/cifar10_app/final_train_wrong/"
        new_size = (500, 400)
        
    elif data == "test":
        path = "/home/skyolia/STS_SPACE/cifar10_app/final_test_wrong/"
        new_size = (500, 400)
        
    elif data == "filter":
        path = "/home/skyolia/STS_SPACE/cifar10_app/filter/"
        new_size = (800, 800)
        
    files = glob.glob(path+"*")
    for f in files:
        os.remove(f)

    l = list()
    for file in sorted(os.listdir(source)):
        if file.endswith(".png"):
            l.append(file)        

    lsorted = sorted(l, key=lambda x: int(os.path.splitext(x)[0]))
    print(lsorted)
    
    
    # new_im = Image.new("RGB", new_size)  # # luckily, this is already black!
    
    j = 0
    for i in range(len(lsorted)):
        if(i % 20 == 0):
            x = 0
            y = 0
            j += 1
            new_im = Image.new("RGB", new_size)  # # luckily, this is already black!
            file_name = "wrong part " + str(j) + ".png"
            fullpath = os.path.join(path, file_name)
            
        old_im = Image.open(source + str(lsorted[i]))
        new_im.paste(old_im, (x, y))
        x += 100
        if ((i + 1) % 5 == 0):
            y += 100
            x = 0
        
        new_im.save(fullpath)

def own_data(img_file, size):
    
    img = Image.open(img_file)
    w, h = img.size
    r = w / h
    
    for i in range(r):
        l = h
        crop_rectangle = (l * i, 0, l * i + h, h)
        cropped_im = img.crop(crop_rectangle)
        basewidth = size
        wpercent = (basewidth / float(cropped_im.size[0]))
        hsize = int((float(cropped_im.size[1]) * float(wpercent)))
        image = cropped_im.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        #image = cropped_im.resize((28, 28), PIL.Image.ANTIALIAS)
        im_grey = image.convert('L')  # convert to grayscale
        image.save('greyscale_resized_img_file.png')
        image = img_as_float(im_grey)
        data = np.asarray(image)
        fdata = data.ravel()
    return fdata

def dataset_creation(source, file_name, labels):
    
    f = open(file_name, 'w')

    for i in range(len(source)):
        dp = deparser(labels[i])
        for item in dp:
            np.save(f, dp)
        
    
    f.close()

def deparser(n):
    
    a = np.zeros((10))
    a[n] = 1
    return a

def deparserx(tab=list()):
    
    x = list(())
    for i in range(len(tab)):
        x.insert(i, deparser(tab[i]))
    xnp = np.asarray(x)
    return xnp