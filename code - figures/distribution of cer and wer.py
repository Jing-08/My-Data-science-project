import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('stat.csv',index_col=0)
cer = df[['CER']].values.tolist()
wer = df[['WER']].values.tolist()
c1= 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
for i in cer:
    if i[0] <= 5 :
        c1 += 1
    elif 5 < i[0] <= 10 :
        c2 += 1
    elif 10 < i[0] <= 15 :
        c3 += 1
    elif 15 < i[0] <= 20 :
        c4 += 1
    else:
        c5 += 1


w1= 0
w2 = 0
w3 = 0
w4 = 0
w5 = 0
for i in wer:
    if i[0] <= 10:
        w1 += 1
    elif 10 < i[0] <= 20:
        w2 += 1
    elif 20 < i[0] <= 30:
        w3 += 1
    elif 30 < i[0] <= 40:
        w4 += 1
    else:
        w5 += 1

data = [c1,c2,c3,c4,c5]
label=["<= 5%","5%~10%","10%~15%","15%~20%",">20%"]
color = ['thistle','pink','lightpink','palevioletred','mediumvioletred']
data1 = [w1,w2,w3,w4,w5]
label1=["<= 10%","10%~20%","20%~30%","30%~40%",">40%"]
color1 = ['lightblue','powderblue','lightskyblue','deepskyblue','steelblue']

patches, l_text, p_text = plt.pie(data, labels = label, pctdistance=0.8,autopct='%.1f%%',colors = color)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20

plt.pie([1],radius=0.6,colors='w')
plt.title('The distribution of CER')
#plt.legend(label,loc='upper left')
plt.show()


patches, l_text, p_text = plt.pie(data1, labels = label1, pctdistance=0.8,autopct='%.1f%%',colors = color1)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20

plt.pie([1],radius=0.6,colors='w')
plt.title('The distribution of WER')
plt.show()
