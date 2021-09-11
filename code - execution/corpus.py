import pandas as pd
import numpy as np
import os
from ocr import re_2,re_punc
import glob
from collections import defaultdict
import fastwer
import Levenshtein as Lev

from spellchecker import SpellChecker
import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity


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

getfiles("data1\\sn85044812\\1897\\")
k2 = set(ocrfiles.keys())
k1 = set(jp2files.keys())

pairs = []
for k in sorted(k2 & k1):
    for x,y in zip(jp2files[k], ocrfiles[k]):
        pairs.append((x, y))

count = 0
lst = []
for x,y in pairs:
    lst.append(y)

corpus = []
for x in lst:
    with open(x, 'r', encoding = 'utf-8') as f:
        text = f.readlines()
        text1 = re_punc(' '.join(text).replace("\n", " ").replace("  ", " "))
        text1 = re_2(text1)
        
    corpus.append(text1.lower())

corpus = ' '.join(corpus)
corpus = corpus.split(" ")

num_count={}
for i in corpus:
    if i not in num_count:
        num_count[i] = 1
    else:
        num_count[i] += 1

c = []
words = []
for k,v in num_count.items():
    c.append([k,v])
    words.append(k)

def corpus():
    return c

with open("corpus.txt","w", encoding = 'utf-8') as f:
    f.write('\n'.join(words))

    
