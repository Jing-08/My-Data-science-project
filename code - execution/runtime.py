import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('stat.csv',index_col=0)
y = df[['runtime']].values.tolist() 
x = [i for i in range(258)]
y1 = []
for i in y:
    y1.append(int(i[0]))

plt.hist(y1, bins = 25, color=sns.desaturate("indianred", .8), alpha=.4,range=(100,300))
plt.grid(alpha=0.5,linestyle='-.') 
plt.xlabel('OCR running time (s)')  
plt.ylabel('Number of times')  
plt.title(r'OCR running time frequency distribution histogram')
plt.show()


y1 = df[['WER']].values.tolist()
y2 = df[['WER without stopwords']].values.tolist()
y11 = []
y22 = []
for i in y1:
    y11.append(i[0])

for i in y2:
    y22.append(i[0])
y0 = []
for i in range(258):
    y0.append(y22[i]-y11[i])

plt.hist(y0, bins = 20, color=sns.desaturate("indianred", .8), alpha=.4,range=(3,15))
plt.grid(alpha=0.5,linestyle='-.') 
plt.xlabel('WER without stopwords - WER (%)')  
plt.ylabel('Number of times')  
plt.title(r'The difference between WER without stopwords and WER frequency distribution histogram')
plt.show()
