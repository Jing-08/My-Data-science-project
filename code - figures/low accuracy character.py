import os
from ocr import re_2,re_punc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('charmean.txt','r',encoding='utf-8') as f:
    f = f.readlines()
low = []
for i in f:
    i = i.split(' ')
    if float(i[1]) < 0.60:
        low.append(i[0])
print(low)

path = 'ocr_results/'
path_list = os.listdir(path)
count = 0
cw = []
cc = []
for filename in path_list:
    w = 0
    c = 0
    with open('data1\\sn85044812\\'+str(path_list[count][0:4])+'\\'+str(path_list[count][4:6])+
                '\\'+str(path_list[count][6:8])+'\\ed-1\\seq-1\\ocr.txt','r',encoding='utf-8') as f:
        f = f.readlines()
    recogtext = ' '.join(f).replace("\n", " ").replace("  ",
                " ").replace("-", '').replace("\n\n",'\n')
    re1 = re_punc(recogtext)
    re2 = re_2(re1).split(" ")
    wl = len(re2)
    cl = len(' '.join(re2))
    for i in re2:
        
        for j in i:
            if j in low:
                c += 1
        for j in i:
            if j in low:
                w += 1
            break
    cw.append(w/wl)
    cc.append(c/cl)
    count += 1


CER = []
WER = []

df = pd.read_csv('stat.csv',index_col=0)
cer = df[['CER']].values.tolist()
wer = df[['WER']].values.tolist()
for i in range(258):
    CER.append(cer[i][0]/100)
    WER.append(wer[i][0]/100)

#plt.scatter(cw, WER,c='blue', alpha=0.6)
z1 = np.polyfit(cw, WER, 1)
p1 = np.poly1d(z1)
yvals=p1(cw) 
plt.plot(cw, yvals, 'lightpink',label='polyfit values')
plt.xlabel('Percentage of words that contain characters with low recognition rates')
plt.ylabel('WER')
plt.scatter(cw,WER,c='skyblue',alpha=0.6)
plt.show()
