import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''
The influence of length on cer and wer
'''
df = pd.read_csv('stat.csv',index_col=0)
df2 = pd.read_csv('wer+cer.csv',index_col=0)
#df.to_csv('spellchecker.csv',encoding='utf-8')
#df = df[['ocr word len','ocr char len','runtime']]
df = df.iloc[0:100]
colors = np.random.rand(100)  
size = df[['WER']]
plt.scatter(df[['word len div']], df2[['wer1']], c=colors, alpha=0.6)
xx=[]
yy=[]
for i in df[['word len div']].values.tolist():
    xx.append(i[0])
for i in df2[['wer1']].values.tolist():
    yy.append(i[0])
z1 = np.polyfit(xx, yy, 1)
p1 = np.poly1d(z1)
yvals=p1(xx) 
plt.plot(xx, yvals, 'r',label='polyfit values')
plt.xlabel('REAL TEXT WORD NUMBER - OCR TEXT WORD NUNBER')
plt.ylabel('CER')  
plt.show()


'''
# relationship between CER and WER
'''
dff = df[['WER','CER']]

x = df[['CER']].values.tolist()
xx =[]
yy =[]
for i in x:
    xx.append(i[0]/100)
y = df[['WER']].values.tolist()
for i in y:
    yy.append(i[0]/100)
z1 = np.polyfit(xx, yy, 1)
p1 = np.poly1d(z1)
yvals=p1(xx) 
plt.plot(xx, yvals, 'r',label='polyfit values')
x1 = []
x2 =[]
x3=[]
y1= []
y2=[]
y3=[]
for ind,i in enumerate(xx):
    if i < 0.1:
        x1.append(i)
        y1.append(yy[ind])
    elif i > 0.2:
        x3.append(i)
        y3.append(yy[ind])
    else:
        x2.append(i)
        y2.append(yy[ind])
plt.scatter(x1, y1, c=['green'],alpha=0.6)
plt.scatter(x2, y2, c=['skyblue'],alpha=0.6)
plt.scatter(x3, y3, c=['red'],alpha=0.6)
plt.grid(ls = '--')
plt.xlabel('CER')  
plt.ylabel('WER')  
plt.show()

