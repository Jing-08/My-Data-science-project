# package
import pandas as pd
import numpy as np
import os
from ocr import re_2,re_punc
import glob
from collections import defaultdict
import fastwer
import Levenshtein as Lev

from corpus import corpus
from spellchecker import SpellChecker
import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
# file path
path = 'C:\\Users\\13572\\Desktop\\chroniclingamerica-ocr\\ocr_results\\'
# file list
files = []
resultfiles = []
for file in os.listdir(path):
    if file.endswith(".txt"):
        files.append(path+file)
        resultfiles.append(file[0:8])
csv_data = pd.read_csv('wer+cer.csv', low_memory = False)
df = csv_data[["cer5","wer5"]]

def find_dub_strs(mystring):
    grp = groupby(mystring)
    seq = [(k,len(list(g)) >= 2) for k,g in grp]
    allowed = ('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return rec_dubz('',seq,allowed=allowed)

def rec_dubz(prev,seq, allowed):
    if not seq:
        return [prev]
    solutions = rec_dubz(prev + seq[0][0],seq[1:],allowed=allowed)
    if seq[0][0] in allowed and seq[0][1]:
        solutions += rec_dubz(prev + seq[0][0] * 2,seq[1:],allowed=allowed)
    return solutions


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

jp2files = defaultdict(list)
for path in sorted(glob.glob("ocr_results")):
    for x in sorted(os.listdir(path)):
        key = x.split("_")[0][:8]
        jp2files[key].append(os.path.join(path, x))
ocrfiles = defaultdict(list)
getfiles("data1\\sn85044812\\1896\\")
k2 = set(ocrfiles.keys())
k1 = set(jp2files.keys())
for k in sorted(k2 & k1):
    for x,y in zip(jp2files[k], ocrfiles[k]):
        pairs.append((x, y))
print(len(pairs))

corpus = corpus()
corpuswords = []
for i in corpus:
    corpuswords.append(i[0])
count = 0
l1c = []
l2c = []
l3c = []
l1w = []
l2w = []
l3w = []
for x,y in pairs[0:100]:
    print(count)
    #print(x)
    #print(y)
    with open(x, 'r', encoding = 'utf-8') as f:
        recogtext = f.readlines()
        #print('OCR genrated text:',recogtext)
    with open(y, 'r', encoding = 'utf-8') as f:
        realtext = f.readlines()
        #print('Real text:',realtext)

    recogtext = ' '.join(recogtext).replace("\n", " ").replace("  ",
                " ").replace("-", '').replace("\n\n",'\n')
    re1 = re_punc(recogtext).lower()
    re2 = re_2(re1).split(" ")
    
    reduce = []
    for i in re2:
        if any(j.isdigit() for j in i) == 0 and any(j.isupper() for j in i) == 0:
            a = find_dub_strs(i)
            reduce.append(a[-1])
        else:
            reduce.append(i)
    
    realtext = ' '.join(realtext).replace("\n", " ").replace("  ", " ")
    re1 = re_punc(realtext)
    realtext = re_2(re1).lower()
    len1 = len(reduce)
    len2 = len(realtext.split(" "))
    print(len1, len2)
    
    #Levenshtein distance
    correcttext = []
    #for e in range(1,4):
    for word in reduce:
        if word  not in corpuswords:
            dis = 1000000
            c = 0
            for i in corpuswords:
                l = Lev.distance(word,i)
                if l < dis:
                    dis = l
                    changeword = i
                    cw = c
                elif l == dis:
                    if corpus[c][1] > corpus[cw][1]:
                        changeword = i
                c += 1
            if dis > 1:
                changeword = word
            correcttext.append(changeword)
            print(word,changeword)
        else:
            correcttext.append(word)
    #print(' '.join(correcttext))       

    print('--------------------------------------------------')
    
    '''
    # SpellChecker()
    correcttext1 = []
    spell = SpellChecker()
    for word in reduce:
        
        if any(x.isupper() for x in word) == 0:
            newword = spell.correction(word)
            correcttext1.append(newword)
        else:
            correcttext1.append(word)
        
        if word not in corpuswords:
            newword = spell.correction(word)
            correcttext1.append(newword)
        else:
            correcttext1.append(word)

    print('--------------------------------------------------')
    
    # SpellChecker + Levenshtein distance
    correcttext2 = []
    spell = SpellChecker()
    for word in reduce:
        if word  not in corpuswords:
            dis = 1000000
            c = 0
            for i in corpuswords:
                l = Lev.distance(word,i)
                if l < dis:
                    dis = l
                    changeword = i
                    cw = c
                elif l == dis:
                    if corpus[c][1] > corpus[cw][1]:
                        changeword = i
                c += 1
            if dis > 1:
                newword = spell.correction(word)
                correcttext2.append(newword)
            else:
                correcttext2.append(changeword)
        else:
            correcttext2.append(word)
    
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    '''
    '''
    c1 = []
    w1 = []
    c2 = []
    w2 = []
    for i in range(2):
        s1 = correcttext1[int(i*len1/2):int((i+1)*len1/2)+100]
        s11 = correcttext2[int(i*len1/2):int((i+1)*len1/2)+100]
        s2 = realtext.split(" ")
        s2 = s2[int(i*len2/2):int((i+1)*len2/2)]
        c1.append(round(fastwer.score_sent( ' '.join(s1), ' '.join(s2), char_level=True),2))
        w1.append(round(fastwer.score_sent( ' '.join(s1), ' '.join(s2)),2))
        c2.append(round(fastwer.score_sent( ' '.join(s11), ' '.join(s2), char_level=True),2))
        w2.append(round(fastwer.score_sent( ' '.join(s11), ' '.join(s2)),2))

    post_cer = sum(c1)/len(c1)
    post_wer = sum(w1)/len(w1)
    
    post_cer1 = sum(c2)/len(c2)
    post_wer1 = sum(w2)/len(w2)
    
    pre_cer = df.values[count][0]
    pre_wer = df.values[count][1]
    post_cer = round(fastwer.score_sent( ' '.join(correcttext2), realtext, char_level=True),2)
    post_wer = round(fastwer.score_sent( ' '.join(correcttext2), realtext),2)
    #post_cer1 = round(fastwer.score_sent( ' '.join(correcttext2), realtext, char_level=True),2)
    #post_wer1 = round(fastwer.score_sent( ' '.join(correcttext2), realtext),2)
    
    print(round(pre_cer-post_cer,2), round(pre_wer-post_wer,2))
    #print(post_cer, post_cer1, post_cer2, pre_cer)
    #print(post_wer, post_wer1, post_wer2, pre_wer)
    '''
    count += 1
    '''
    l1c.append(pre_cer - post_cer)
    l2c.append(pre_cer - post_cer1)
    l3c.append(pre_cer - post_cer2)
    l1w.append(pre_wer - post_wer)
    l2w.append(pre_wer - post_wer1)
    l3w.append(pre_wer - post_wer2)
    print(l1c)
    print(l1w)
    '''

