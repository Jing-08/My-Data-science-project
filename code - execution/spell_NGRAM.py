# # SPELL CORRECTION method 3

import re
import nltk
import numpy as np
from copy import deepcopy
import fastwer
from collections import defaultdict
import glob
import os
from ocr import re_punc, re_2
import Levenshtein as Lev
import pandas as pd

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
csv_data = pd.read_csv('wer+cer.csv', low_memory = False)
df = csv_data[["cer5","wer5"]]

def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))

def edits3(word):
    return set(e2 for e1 in edits2(word) for e2 in edits2(e1))

def edittype(word,error):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    word,error = word.lower(),error.lower()
    for i in range(len(word)):
        if word == error[1:]:
            return error[0]+'|<s>', 'ins'
        if word[1:] == error:
            return '|'+word[0], 'del'
        if i >= len(error):
            return word[i]+'|'+error[i-1]+word[i], 'del'
        elif word[i] != error[i]:
            print(word,error)
            if word in [error[:i]+k+error[i:] for k in letters]:
                return error[i-1]+'|'+error[i-1]+word[i], 'del'
            elif word in [error[:i]+k+error[i+1:] for k in letters]:
                return error[i]+'|'+word[i], 'sub'
            elif word == error[:i]+error[i+1:] or word == error[:-1]:
                return word[i-1]+error[i]+'|'+word[i-1], 'ins'
            elif word[i]+ word[i-1] == error[i-1]+error[i]:
                return word[i+1]+word[i]+'|'+word[i]+word[i+1], 'trans'
    if len(word)<len(error):
        return word[-1]+error[-1]+'|'+word[-1], 'ins'

def editdistance(s1, s2):
    
    len1 = len(s1)
    len2 = len(s2)
     
    matrix = [[i+j for j in range(len2 + 1)] for i in range(len1 + 1)]
#     print(matrix)
    for row in range(len1):
        for col in range(len2):
            comp = [matrix[row+1][col]+1, matrix[row][col+1]+1]
             
            if s1[row] == s2[col]:
                comp.append(matrix[row][col])
            else:
                comp.append(matrix[row][col]+1)
             
            if row > 0 and col > 0:
                if s1[row] == s2[col-1] and s1[row-1] == s2[col]:
                    comp.append(matrix[row-1][col-1]+1)
                     
            matrix[row+1][col+1] = min(comp)
             
    return matrix[len1][len2],matrix
    
    
def known(words): # whether words are in CORPUS
    return set(word for word in words if word in VOCAB)


def prob1(candidate, word, sentence):
    if candidate in UNIGRAM.keys():
        p_unigram = UNIGRAM[candidate]
    else:
        p_unigram = -100000
    return p_unigram

def BIGRAM_p(s):
    if s in BIGRAM.keys():
        return BIGRAM[s]
    else:
        return -100000

def prob2(candidate, j, sentence):
    if j==0:
        return BIGRAM_p(candidate+' '+sentence[j+1].lower())
    elif j==len(sentence) - 1:
        return BIGRAM_p((sentence[j-2].lower()+' '+candidate))
    elif j < len(sentence) - 1:
        return BIGRAM_p(candidate+' '+sentence[j+1].lower()) + BIGRAM_p(sentence[j-1].lower()+' '+candidate)

def non_word_correct(sentence):
    wrong = 0
    for j in range(len(sentence)):
        word = sentence[j]
        if bool(re.search(r"[\d.,/'-]", word)) or word.lower() in VOCAB:
            continue
        # edit distance = 1
        word_lower = word.lower()
        candidates = known(edits1(word_lower))
        # edit distance = 2
        #candidates1 = known(edits1(word_lower))
        #candidates2 = known(edits2(word_lower))
        #candidates = candidates1.union(candidates2)
        '''
        # edit distance = 3
        candidates1 = known(edits1(word_lower))
        candidates2 = known(edits2(word_lower))
        candidates3 = known(edits3(word_lower))
        candidates = candidates1.union(candidates2).union(candidates3)
        '''
        p_flag = -200000
        right = word
        for candidate in candidates:
            # unigram prob
            p = prob1(candidate, word, sentence) #* prob2(candidate,j, sentence)
            #if len(sentence)>=2:
                #p = prob2(candidate,j, sentence)
            if p > p_flag:
                p_flag = p
                right = candidate
        if not word.islower():
            flag = 0
            for each in word:
                flag += int(each.isupper())
            if flag == 1:
                right = right[0].upper()+right[1:]
            else:
                right = right.upper()
        sentence[j] = right # to do supper letters
        wrong += 1
    return wrong

def word_correct(sentences_origin, VOCAB):
    sentences = deepcopy(sentences_origin)
    for i in range(len(sentences)):
        sentence = sentences[i]
        wrong = non_word_correct(sentence)
    return sentences

with open('corpus.txt', 'r', encoding = 'utf-8') as f:
    VOCAB = [re.sub('[0-9,.!“”‘’/\—\–\-\'\"]', '', line.strip()) for line in f.readlines()]
    
VOCAB = set([each.lower() for each in VOCAB if each!=''])

def ngram(file, n):
    with open(file,encoding = 'utf-8') as f:
        p = f.read()
    p = [[re.sub('[,.!“”‘’\—\–\'\"]', '', word).lower() for word in sentence.split()] for sentence in nltk.sent_tokenize(p)]
    pp = []
    for l in p:
        for i in l:
            pp.append(i)
    #p = re_punc(' '.join(pp).replace("\n", " ").replace("  ", " "))
    #p = re_2(p).split(" ")
    
    output = {}
    
    for l in p:
        for i in range(len(l) - n + 1):
            temp = " ".join(l[i : i + n])
            if temp not in output:
                output[temp] = 0
            output[temp] += 1
    
    '''
    for i in range(len(pp)):
        temp = " ".join(pp[i:i+n])
        if temp not in output:
            output[temp] = 0
        else:
            output[temp] += 1
    print(output)
    '''
    return output


count = 0
for x,y in pairs[0:100]:
    print(count)
    with open(x,'r',encoding = 'utf-8') as f:
        sentences0 = f.read().split('\n')
    with open(y, 'r', encoding = 'utf-8') as f:
        real = f.readlines()
    # original ocr generated texts
    sentences_origin = []
    for x in sentences0:
        if x != '' and x!=' ':
            x = x.split(" ")
            sentences_origin.append(x)
    
    # UNIGRAM
    UNIGRAM = ngram(y, 1)
    sum1 = sum(UNIGRAM.values())
    for each in UNIGRAM:
        UNIGRAM[each] = np.log(UNIGRAM[each]/sum1)
    # BIGRAM
    BIGRAM = ngram(y, 2)
    sum2 = sum(BIGRAM.values())
    for each in BIGRAM:
        BIGRAM[each] = np.log(BIGRAM[each]/sum2)

    # begin spelling check
    sentences_correct = word_correct(sentences_origin, VOCAB)
    print(sentences_origin)
    print(sentences_correct)

    #file = '\\spell correct\\result.txt'
    with open('result.txt','w',encoding = 'utf-8') as f:
        for i,each in enumerate(sentences_correct):
            sentence = ' '.join(sentences_correct[i])
            f.write(sentence+'\n')
    with open('result.txt','r',encoding = 'utf-8') as f:
        test0 = f.read()

    # calculate CER and WER
    real1 = re_punc(' '.join(real).replace("\n", " ").replace("  ", " "))
    real1 = re_2(real1).lower()
    print(real1)
    text = re_punc(test0.replace("\n", " ").replace("  ", " "))
    text = re_2(text).lower()
    

    cerr5 = round(fastwer.score_sent(real1,text,char_level=True),2)
    werr5 = round(fastwer.score_sent(real1,text),2)
    pre_cer = df.values[count][0]
    pre_wer = df.values[count][1]
    print(round(pre_cer-cerr5,2),round(pre_wer-werr5,2))
    count += 1

