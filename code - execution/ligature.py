import pandas as pd
import numpy as np
import os
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
for path in sorted(glob.glob("ocr_results")):
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
getfiles("data1\\sn85044812\\1895\\")
k2 = set(ocrfiles.keys())
k1 = set(jp2files.keys())

pairs = []
for k in sorted(k2 & k1):
    for x,y in zip(jp2files[k], ocrfiles[k]):
        pairs.append((x, y))

print(pairs)

for x,y in pairs[0:10]:
    
    with open(x, 'r', encoding = 'utf-8') as f:
        recogtext = f.readlines()
        #print('OCR genrated text:',recogtext)
    with open(y, 'r', encoding = 'utf-8') as f:
        text = f.readlines()
        #print('Real text:',realtext)

    recogtext = ' '.join(recogtext).replace("\n", " ").replace("  ",
                " ").replace("-", '').replace("\n\n",'\n')
    re1 = re_punc(recogtext)
    re2 = re_2(re1).split(" ")
    text1 = re_punc(' '.join(text).replace("\n", " ").replace("  ", " "))
    text1 = re_2(text1)
    rr = ' '.join(re2)
    print(len(text1))
    print(len(text1.split()))
    c = 0
    for n,i in enumerate(rr):
        if i == 'ﬂ':
            rr[n] == 'fl'
            c += 1
        if i == 'ﬁ':
            rr[n] == 'fi'
            c += 1
    '''
    f = open('shan.txt', 'w', encoding = 'utf-8')
    f.write(rr)
    f = open('shan1.txt', 'w', encoding = 'utf-8')
    f.write(text1)
    
    with open('shan.txt', 'r', encoding = 'utf-8') as f:
        re = f.readlines()
    with open('shan1.txt', 'r', encoding = 'utf-8') as f:
        text = f.readlines()
    
    #cer = round(fastwer.score_sent(str(re),str(text),char_level=True),3)
    wer = round(fastwer.score_sent(re,text),3)
    print('newcer5:',cer)
    print('newwer5:',wer)
    '''
    print(c)

