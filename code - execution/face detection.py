import pandas as pd
import numpy as np
import os
import cv2
from ocr import re_2,re_punc
import glob
from collections import defaultdict
import fastwer
import Levenshtein as Lev
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity
from collections import Counter

path = 'C:\\Users\\13572\\Desktop\\chroniclingamerica-ocr\\ocr_results\\'

files = []
resultfiles = []
for file in os.listdir(path):
    if file.endswith(".txt"):
        files.append(path+file)
        resultfiles.append(file[0:8])

jp2files = defaultdict(list)
for path in sorted(glob.glob("data1\\1897")):
    for x in sorted(os.listdir(path)):
        key = x.split("_")[0][:8]
        jp2files[key].append(os.path.join(path, x))
ocrfiles = defaultdict(list)

def getfiles(filepath, file_format='.txt'):
    for x in sorted(os.listdir(filepath), key=lambda x: int(x.split('-')[-1]) if('seq' in x) else x):
        x_file = os.path.join(filepath, x)
        if os.path.isfile(x_file) and os.path.splitext(x_file)[1] == file_format:
            key = "".join(filepath.split("\\")[2:5])
            ocrfiles[key].append(x_file)
        if os.path.isdir(x_file):
            getfiles(x_file)
getfiles("data1\\sn85044812\\1897\\")
k2 = set(ocrfiles.keys())
k1 = set(jp2files.keys())

pairs = []
def face_recog(img): 
    #Open the grey-scaled image as an n-dimensional array
    image = np.array(img)
    
    #Loading the face detection classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load(r'C:\Users\13572\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')

    
    #Return the facial recognition rectangles
    faces= face_cascade.detectMultiScale(img,
                scaleFactor = 1.3)#,minNeighbors = 15,
                                  #       minSize = (32,32),flags = 4)
    color = (0,255,0)
    if len(faces) > 0:
        for f in faces:
            x,y,w,h = f
            cv2.rectangle(img, (x - 10, y - 10), (x + w, y + h), color, 2)
            cv2.namedWindow('faces',cv2.WINDOW_NORMAL)
            cv2.imshow('faces',img)
            cv2.imwrite('faces.png',img)
            cv2.waitKey(0)
        return len(faces)
            
    
for k in sorted(k2 & k1):
    for x,y in zip(jp2files[k], ocrfiles[k]):
        pairs.append((x, y))

print(len(pairs))
count = 0
fnames = []
for x,y in pairs:
    x = r'C:\Users\13572\Desktop\chroniclingamerica-ocr\data1\1896\1896082301_0918.jp2'
    print(count)
    img = cv2.imread(x)
    print(face_recog(img))
    count += 1
    fnames.append(x)
print(fnames)
    


    
