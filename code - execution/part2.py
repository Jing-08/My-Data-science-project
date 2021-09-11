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
'''
csv_file = "wer+cer.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)
df1 = pd.DataFrame(csv_data)
'''
count = 0


def get_char_count(text):

    d = {}
    text = ''.join(text)
    for i in text:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    #lst = sorted(d.items(), key = lambda x:x[1], reverse = True)
    return d


import sys

def theta(a, b):
    if a == '-' or b == '-' or a != b:   # gap or mismatch
        return -1
    elif a == b:                         # match
        return 1

def make_score_matrix(seq1, seq2):
    """
    return score matrix and map(each score from which direction)
    0: diagnosis
    1: up
    2: left
    """
    seq1 = '-' + seq1
    seq2 = '-' + seq2
    score_mat = {}
    trace_mat = {}

    for i,p in enumerate(seq1):
        score_mat[i] = {}
        trace_mat[i] = {}
        for j,q in enumerate(seq2):
            if i == 0:                    # first row, gap in seq1
                score_mat[i][j] = -j
                trace_mat[i][j] = 1
                continue
            if j == 0:                    # first column, gap in seq2
                score_mat[i][j] = -i
                trace_mat[i][j] = 2
                continue
            ul = score_mat[i-1][j-1] + theta(p, q)     # from up-left, mark 0
            l  = score_mat[i][j-1]   + theta('-', q)   # from left, mark 1, gap in seq1
            u  = score_mat[i-1][j]   + theta(p, '-')   # from up, mark 2, gap in seq2
            picked = max([ul,l,u])
            score_mat[i][j] = picked
            trace_mat[i][j] = [ul, l, u].index(picked)   # record which direction
    return score_mat, trace_mat

def traceback(seq1, seq2, trace_mat):
    
    seq1, seq2 = '-' + seq1, '-' + seq2
    i, j = len(seq1) - 1, len(seq2) - 1
    path_code = ''
    while i > 0 or j > 0:
        direction = trace_mat[i][j]
        if direction == 0:                    # from up-left direction
            i = i-1
            j = j-1
            path_code = '0' + path_code
        elif direction == 1:                  # from left
            j = j-1
            path_code = '1' + path_code
        elif direction == 2:                  # from up
            i = i-1
            path_code = '2' + path_code
    return path_code
'''
def print_m(seq1, seq2, m):
    """print score matrix or trace matrix"""
    seq1 = '-' + seq1; seq2 = '-' + seq2
    print()
    print(' '.join(['%3s' % i for i in ' '+seq2]))
    for i, p in enumerate(seq1):
        line = [p] + [m[i][j] for j in range(len(seq2))]
        print(' '.join(['%3s' % i for i in line]))
    print()
    return
'''
def pretty_print_align(seq1, seq2, path_code):
    '''
    return pair alignment result string from
    path code: 0 for match, 1 for gap in seq1, 2 for gap in seq2
    '''
    align1 = ''
    middle = ''
    align2 = ''
    dui = {}
    cuo = []
    for p in path_code:
        if p == '0':
            align1 = align1 + seq1[0]
            align2 = align2 + seq2[0]
            if seq1[0] == seq2[0]:
                if seq1[0].isalpha() == 1:
                    if seq1[0] in dui:
                        dui[seq1[0]] += 1
                    else:
                        dui[seq1[0]] = 1
            
            else:
                cuo.append(str(seq1[0])+str(seq2[0]))


            seq1 = seq1[1:]
            seq2 = seq2[1:]
        elif p == '1':
            align1 = align1 + '-'
            align2 = align2 + seq2[0]
            #middle = middle + ' '
            seq2 = seq2[1:]
        elif p == '2':
            align1 = align1 + seq1[0]
            align2 = align2 + '-'
            #middle = middle + ' '
            seq1 = seq1[1:]
    #print(dui)
    #print('Alignment:\n\n   ' + align1 + '\n   ' + middle + '\n   ' + align2 + '\n')
    return dui,cuo

def main(seq1,seq2):
 
    #print('1: %s' % seq1)
    #print('2: %s' % seq2)
    
    score_mat, trace_mat = make_score_matrix(seq1, seq2)
    #print_m(seq1, seq2, score_mat)
    #print_m(seq1, seq2, trace_mat)

    path_code = traceback(seq1, seq2, trace_mat)
    r,c = pretty_print_align(seq1, seq2, path_code)
    #print('   '+path_code)
    return r,c

stop = 1
avg_char = []
occur = {}
wrrlst = []
swwer = []
swwrrlst = []
predwllst = []
predcllst = []
realcllst = []
realwllst = []
cerlst = []
werlst = []
Levr = []
wrong = []
for x,y in pairs:
    print(stop)
    print(x)
    print(y)
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
    
    perlen0 = int(len(text1.split(" "))/10)
    perlen1 = int(len(re2)/10)

    lesskey = []
    all_char = get_char_count(text1) # all characters
 
    # The sum of the newspaper page
    char = []
    
    for i in range(10):
        real = text1.split(" ")
        real = real[i * perlen0 : (i+1) * perlen0]
        generate = re2[i * perlen1 : (i+1) * perlen1]
        result,cuo1 = main(' '.join(real), ' '.join(generate))
        char.append(result)
        wrong += cuo1
    wrongchange = []
    print('wrong:',wrong)
    
    for c in ['a','b','c','d','e','f','g','h','i','j','k','l',
                  'm','n','o','p','q','r','s','t','u','v','w','x','y','z']:
        new = []
        for i in cuo1:
            if i[0] == c:
                new.append(i[1:])
        word_counts = Counter(new) # list to Counter
        top_three = word_counts.most_common(26)
        wrongchange.append(top_three)
    print('change:',wrongchange)
    '''
    t=''
    with open ('wrongchange1.txt','a',encoding = 'utf-8') as q:
        for i in wrongchange:
            if i != []:
                for e in range(len(wrongchange[0])):
                    t=t+str(i[e])+' '
                q.write(t.strip(' '))
                q.write('\n')
                t=''
            else:
                q.write('nan')
                q.write('\n')
        q.write('\n')

    '''        
    chars = {}
    for i in range(10):
        for key,value in char[i].items():
            if key in chars:
                chars[key] += value
            else:
                chars[key] = value

    for k,v in chars.items():
        if v < 30:
            #del all_char[k]
            lesskey.append(k)
    for i in lesskey:
        del all_char[i]
    for i in lesskey:
        del chars[i]

    print('The sum of the actual letters appearing in this newspaper:',all_char) # 删除出现次数太少的
    
    all1 = []
    for k,v in all_char.items():
        all1.append(list((k,v)))
    print(all1)
    t= ''
    with open ('all_char1.txt','a',encoding = 'utf-8') as q:
        for i in all1:
            for e in range(len(all1[0])):
                t=t+str(i[e])+' '
            q.write(t.strip(' '))
            q.write('\n')
            t=''
        q.write('\n')
         

    # if it is occur
    for key, v in chars.items():
        if key in occur:
            occur[key] += 1
        else:
            occur[key] = 1
    
    #get_word_count(text, re2)
    
    f_char = {}
    for k in all_char:
        if k.isalpha() == 1:
            if (k in chars and all_char[k] != chars[k]):
                f_char[k] = all_char[k]
                f_char[k] = round(chars[k] / all_char[k], 3)
            elif k not in chars:
                f_char[k] = 0
    print('The accuracy of this newspaper match：',f_char) # Probability of this article
    #f2 = open('f_char.txt','a',encoding = 'utf-8')
    ff = [[] for i in range(len(f_char))]
    i = 0
    for k,v in f_char.items():
        ff[i].append(k)
        ff[i].append(v)
        i += 1
    print(ff)
    stop += 1
    with open ('ff1.txt','a',encoding = 'utf-8') as q:
        for i in ff:
            for e in range(len(ff[0])):
                t=t+str(i[e])+' '
            q.write(t.strip(' '))
            q.write('\n')
            t=''
        q.write('\n')

'''
df = pd.DataFrame()
#df['WRR'] = wrrlst
df['WER without stopwords'] = swwer
df['CER'] = cerlst
df['WER'] = werlst
#df['WRR without stopwords'] = swwrrlst
#df['ocr word len'] = predwllst
#df['ocr char len'] = predcllst
#df['real char len'] = realcllst
#df['real word len'] = realwllst
#df['Lev ratio'] = Levr
df.to_csv('statis.csv',header = True)
'''

'''
f_chars = {}
for i in range(26):
    for key,value in avg_char[i].items():
        if key in f_chars:
            f_chars[key] += value
        else:
            f_chars[key] = value
            
print('Total probability accumulation occurs:',f_chars)
print('Total occurrence statistics:',occur)

for k, v in f_chars:
    f_chars[k] = v / 3

print(f_chars)

avg = {}
for k,v in f_chars.items():
    if v!= 0 and occur[k] != 0:
        avg[k] = round(v / occur[k],3)

print('avg result：',avg)


for c in ['a','b','c','d','e','f','g','h','i','j','k','l',
                  'm','n','o','p','q','r','s','t','u','v','w','x','y','z']:
    new = []
    zong = 0
    for i in cuo:
        if i[0] == c and i[1].isalpha() and i[1:].lower() != c:
            new.append(i[1:].lower())
    word_counts = Counter(new)
    for k,v in word_counts.items():
            zong += v
    for k,v in word_counts.items():
            print(k, round(v/zong,4))
'''
