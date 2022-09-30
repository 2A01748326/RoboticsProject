import numpy as np
import cv2
import os
from xml.dom import minidom
from pascal_voc_writer import Writer

def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")
    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)
    cv2.destroyWindow(imageName)


# Writes an PNG image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


cwd = os.path.dirname(os.path.realpath(__file__))
print(cwd)

#Es necesario utilizar el path de tu copia local de seafile, especificamente
# TEC-Forest-Fire-Detection\imgs-datasets\fismo\FiSmo-Images\BoWFire\dataset  
imgPath = r'D:\proyectorobotica\Seafile\TEC-Forest-Fire-Detection\imgs-datasets\fismo\FiSmo-Images\BoWFire\dataset'
binImgPath = os.path.join(imgPath, 'gt')
colorImgPath = os.path.join(imgPath, 'img')
print("The files insde ", binImgPath, "are: ")
xmlTrainPath = os.path.join(cwd,"..","res","XMLs", "Train")
xmlValPath = os.path.join(cwd,"..","res","XMLs", "Val")
imgTrainPath = os.path.join(cwd,"..","res","Images", "Train")
imgValPath = os.path.join(cwd,"..","res","Images", "Val")

colorList = os.listdir(colorImgPath)
for elem in colorList:
    print(elem)


binList = os.listdir(binImgPath)
for elem in binList:
    print(elem)

inList = os.listdir(binImgPath)
#for elem in lista:
#    print(elem)

#Checar si existe la carpeta XMLs, si no existe, se crea
if not os.path.isdir(os.path.join(cwd,"..","res","XMLs")):
    os.mkdir(os.path.join(cwd,"..","res","XMLs"))
    print("XMLs directory created!")
    os.mkdir(os.path.join(cwd,"..","res","XMLs", "Train"))
    print("XMLs/Train directory created!")
    os.mkdir(os.path.join(cwd,"..","res","XMLs", "Val"))
    print("XMLs/Val directory created!")
else:
    if not os.path.isdir(os.path.join(cwd,"..","res","XMLs", "Train")):
        os.mkdir(os.path.join(cwd,"..","res","XMLs", "Train"))
        print("XMLs/Train directory created!")
    if not os.path.isdir(os.path.join(cwd,"..","res","XMLs", "Val")):
        os.mkdir(os.path.join(cwd,"..","res","XMLs", "Val"))
        print("XMLs/Val directory created!")

if not os.path.isdir(os.path.join(cwd,"..","res","Images")):
    os.mkdir(os.path.join(cwd,"..","res","Images"))
    print("Images directory created!")
    os.mkdir(os.path.join(cwd,"..","res","Images", "Train"))
    print("Images/Train directory created!")
    os.mkdir(os.path.join(cwd,"..","res","Images", "Val"))
    print("Images/Val directory created!")
else:
    if not os.path.isdir(os.path.join(cwd,"..","res","Images", "Train")):
        os.mkdir(os.path.join(cwd,"..","res","Images", "Train"))
        print("Images/Train directory created!")
    if not os.path.isdir(os.path.join(cwd,"..","res","Images", "Val")):
        os.mkdir(os.path.join(cwd,"..","res","Images", "Val"))
        print("Images/Val directory created!")

cont = 0

for i in range(len(colorList)):
    #Check first image and check the bounding rect of the blob
    colorInputImage = readImage(os.path.join(colorImgPath, colorList[i]))
    #showImage(colorList[i], colorInputImage)
    binInputImage = readImage(os.path.join(binImgPath, binList[i]))
    binInputImage = cv2.cvtColor(binInputImage, cv2.COLOR_BGR2GRAY)
    automaticThreshold, binInputImage = cv2.threshold(binInputImage, 0, 255, cv2.THRESH_OTSU)
    #showImage(binList[i], binInputImage)
    # print(binInputImage)
    contours, _ = cv2.findContours(binInputImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(binInputImage.shape)
    if cont == 10:
        writer = Writer(os.path.join(imgValPath, colorList[i]), colorInputImage.shape[1], colorInputImage.shape[0])
    else:
        writer = Writer(os.path.join(imgTrainPath, colorList[i]), colorInputImage.shape[1], colorInputImage.shape[0])
    for c in contours:
        # Compute blob bounding rectangle:
        blobRectangle = cv2.boundingRect(c)
        # Compute blob area:
        blobArea = cv2.contourArea(c)
        # Print the blob area:
        # print("Blob Area: "+str(blobArea))
        # Print blobRectangle
        print("BlobRectangle List: " + colorList[i], blobRectangle)
        cv2.rectangle(colorInputImage, (blobRectangle[0], blobRectangle[1]), (blobRectangle[0]+blobRectangle[2], blobRectangle[1]+blobRectangle[3]), (0, 128, 0), 2)
        
        writer.addObject('fire', blobRectangle[0], blobRectangle[1], blobRectangle[0]+blobRectangle[2], blobRectangle[1]+blobRectangle[3])
    
    if cont == 10:
        writer.save(xmlValPath + '/' + colorList[i][:-4] + '.xml')
        writeImage(imgValPath + '/' + colorList[i][:-4],colorInputImage)
        cont = 0
    else:
        writeImage(imgTrainPath + '/' + colorList[i][:-4],colorInputImage)
        writer.save(xmlTrainPath + '/' + colorList[i][:-4] + '.xml')
    #showImage("Color", colorInputImage)
    cont += 1