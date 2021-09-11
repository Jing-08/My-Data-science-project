import io
import os
import glob
from PIL import Image
import cv2
import pytesseract
import numpy
import pandas as pd

import time
from collections import defaultdict
from joblib import Parallel, delayed

import Levenshtein as Lev
from nltk.corpus import stopwords
from ocr import re_2,re_punc
import re
import fastwer
def list_file_pair():
    newname = []
    pdffiles = defaultdict(list)
    for path in sorted(glob.glob("ocr_results")):
        for x in sorted(os.listdir(path)):
            if x[0:9] not in newname:
                newname.append(x[0:9])
                key = x.split("_")[0][:8]
                pdffiles[key].append(os.path.join(path, x))
    
    ocrfiles = defaultdict(list)
    
    def getfiles(filepath, file_format='.txt'):
        for x in sorted(os.listdir(filepath), key=lambda x: int(x.split('-')[-1]) if('seq' in x) else x):
            x_file = os.path.join(filepath, x)
            if os.path.isfile(x_file) and os.path.splitext(x_file)[1] == file_format:
                key = "".join(filepath.split("\\")[2:5])
                ocrfiles[key].append(x_file)
            if os.path.isdir(x_file):
                getfiles(x_file)
    
    getfiles("data1\\sn85044812\\1895\\")
    k1 = set(pdffiles.keys())
    k2 = set(ocrfiles.keys())

    #print('diff: ',  (k2|k1) - (k2&k1))
    pairs = []
    for k in sorted(k2 & k1):
        for x,y in zip(pdffiles[k], ocrfiles[k]):
            pairs.append((x, y))
    
    for path in sorted(glob.glob("data1\\1896")):
        for x in sorted(os.listdir(path)):
            if x[0:9] not in newname:
                newname.append(x[0:9])
                key = x.split("_")[0][:8]
                pdffiles[key].append(os.path.join(path, x))
    
    ocrfiles = defaultdict(list)
    getfiles("data1\\sn85044812\\1896\\")
    k1 = set(pdffiles.keys())
    k2 = set(ocrfiles.keys())

    #print('diff: ',  (k2|k1) - (k2&k1))
    for k in sorted(k2 & k1):
        for x,y in zip(pdffiles[k], ocrfiles[k]):
            pairs.append((x, y))
    print(len(pairs))
    return pairs

pairs = list_file_pair()
total = []
for x,y in pairs:
    with open(x, 'r', encoding = 'utf-8') as f:
        pred = f.readlines()
    content = ' '.join(pred).replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n')
    content = re_punc(content)
    content = re_2(content)
    with open(y, 'r', encoding = 'utf-8') as f:
        text0 = f.readlines()
    text = re_punc(' '.join(text0).replace("\n", " ").replace("  ", " "))
    text = re_2(text)
    c = list(set(content.split(" ")))
    t = list(set(text.split(" ")))
    
    C = [word for word in c if any(x.isupper() for x in word) == 1]
    T = [word for word in t if any(x.isupper() for x in word) == 1]
    print(round((len(set(T))&len(set(C)))/len(set(T)),3))
    print(len(set(T)))
    '''
    sw = stopwords.words('english')
    cs = [word for word in c if word in sw]
    ts = [word for word in t if word in sw]
    total.append(round((len(set(ts))&len(set(cs)))/len(set(ts)),3))
print(total)
    '''
