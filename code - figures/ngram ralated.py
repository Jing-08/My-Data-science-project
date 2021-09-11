import pandas as pd
import matplotlib.pyplot as plt


'''
N-gram cer and wer/increase or decrease distribution
'''
df1 = pd.read_csv('ngram.csv',index_col=0)
df2 = pd.read_csv('stat.csv',index_col=0)

df11 = df1[['UNIGRAM+edit=1(WER)']].values.tolist()
df12 = df1[['UNIGRAM+edit=2(WER)']].values.tolist()
df13 = df1[['BIGRAM+edit=1(WER)']].values.tolist()
df14 = df1[['BIGRAM+edit=2(WER)']].values.tolist()
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
c6 = 0
c7 = 0
c8 = 0
for i in range(100):
    if df11[i][0] > 0:
        c1 += 1
    else:
        c2 += 1
    if df12[i][0] > 0:
        c3 += 1
    else:
        c4 += 1
    if df13[i][0] > 0:
        c5 += 1
    else:
        c6 += 1
    if df14[i][0] > 0:
        c7 += 1
    else:
        c8 += 1
data = [c1,c2]
data1 = [c3,c4]
data2 = [c5,c6]
data3 = [c7,c8]
label=["CER decrease","CER increase"]
color = ['yellowgreen','lightsalmon']

label1=["WER decrease","WER increase"]
color1 = ['darkseagreen','salmon']

fig,axes=plt.subplots(1,4)

patches, l_text, p_text = axes[0].pie(data, labels = label1, pctdistance=0.8,autopct='%.1f%%',colors = color)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20
axes[0].pie([1],radius=0.6,colors='w')
axes[0].set_title('UNIGRAM+edit=1(WER)')

patches, l_text, p_text = axes[1].pie(data1, labels = label1, pctdistance=0.8,autopct='%.1f%%',colors = color)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20
axes[1].pie([1],radius=0.6,colors='w')
axes[1].set_title('UNIGRAM+edit=2(WER)')

patches, l_text, p_text = axes[2].pie(data2, labels = label1, pctdistance=0.8,autopct='%.1f%%',colors = color1)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20
axes[2].pie([1],radius=0.6,colors='w')
axes[2].set_title('BIGRAM+edit=1(WER)')

patches, l_text, p_text = axes[3].pie(data3, labels = label1, pctdistance=0.8,autopct='%.1f%%',colors = color1)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20
axes[3].pie([1],radius=0.6,colors='w')
axes[3].set_title('BIGRAM+edit=2(WER)')
fig.show()




'''
N-gram %%
'''

df1 = pd.read_csv('ngram.csv',index_col=0)
df2 = pd.read_csv('stat.csv',index_col=0)
df3 = pd.read_csv('spellchecker.csv',index_col=0)
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1, 1, 1)
plt.grid(linestyle='--')
cer = df2[['CER']].values.tolist()
wer = df2[['WER']].values.tolist()
'''
df11 = df1[['UNIGRAM+edit=1(WER)']].values.tolist()
df12 = df1[['UNIGRAM+edit=2(WER)']].values.tolist()
df13 = df1[['BIGRAM+edit=1(WER)']].values.tolist()
df14 = df1[['BIGRAM+edit=2(WER)']].values.tolist()
c1 = []
c2 = []
c3 = []
c4 = []
x = [i for i in range(100)]
for i in range(100):
    c1.append(df11[i][0] / wer[i][0])
    c2.append(df12[i][0] / wer[i][0])
    c3.append(df13[i][0] / wer[i][0])
    c4.append(df14[i][0] / wer[i][0])


plt.title('N-gram: Magnitude of change of WER')  # 折线图标题
#plt.ylim(0, 1.6)
plt.xlim(-15,108)
plt.xlabel('Newspaper page number')
plt.ylabel('Range (%)')
plt.text(-8,sum(c1)/100,str(round(sum(c1)/100,3)),fontdict={'size':'10','color':'salmon'})
plt.text(100,sum(c2)/100,str(round(sum(c2)/100,3)),fontdict={'size':'10','color':'teal'})
plt.text(-9,sum(c3)/100-0.015,str(round(sum(c3)/100,3)),fontdict={'size':'10','color':'gold'})
plt.text(100,sum(c4)/100-0.015,str(round(sum(c4)/100,3)),fontdict={'size':'10','color':'dimgrey'})
ln1, = plt.plot(x, c1,color = 'salmon') 
ln2, = plt.plot(x, c2,color = 'teal')
ln3, = plt.plot(x, c3,color = 'gold')
ln4, = plt.plot(x, c4,color = 'dimgrey')
plt.legend(handles = [ln1, ln2, ln3, ln4], labels = ['UNIGRAM+Max Lev distance=1',
                                                     'UNIGRAM+Max Lev distance=2',
                                                     'BIGRAM+Max Lev distance=1',
                                                     'BIGRAM+Max Lev distance=2'
                                                ])

plt.plot([-5,105], [sum(c1)/100,sum(c1)/100], color='salmon', linestyle='--')
plt.plot([-5,105], [sum(c2)/100,sum(c2)/100], color='teal', linestyle='--')
plt.plot([-5,105], [sum(c3)/100,sum(c3)/100], color='gold', linestyle='--')
plt.plot([-5,105], [sum(c4)/100,sum(c4)/100], color='dimgrey', linestyle='--')

plt.show()
'''
'''
Levenshtein distance

'''
df11 = df3[['SpellChecker(WER)']].values.tolist()
df12 = df3[['SpellChecker+Lev=1(WER)']].values.tolist()
df13 = df3[['SpellChecker+Lev=2(WER)']].values.tolist()
df14 = df3[['SpellChecker+Lev=3(WER)']].values.tolist()
c1 = []
c2 = []
c3 = []
c4 = []
x = [i for i in range(100)]
for i in range(100):
    c1.append(df11[i][0] / wer[i][0])
    c2.append(df12[i][0] / wer[i][0])
    c3.append(df13[i][0] / wer[i][0])
    c4.append(df14[i][0] / wer[i][0])

plt.title('SpellChecker with Levenshtein distance: Magnitude of change of WER')  # 折线图标题
#plt.ylim(0, 1.6)
plt.xlim(-15,108)
plt.xlabel('Newspaper page number')
plt.ylabel('Range (%)')
plt.text(-8,sum(c1)/100,str(round(sum(c1)/100,3)),fontdict={'size':'10','color':'yellowgreen'})
plt.text(100,sum(c2)/100,str(round(sum(c2)/100,3)),fontdict={'size':'10','color':'cornflowerblue'})
plt.text(-9.5,sum(c3)/100-0.001,str(round(sum(c3)/100,3)),fontdict={'size':'10','color':'mediumpurple'})
plt.text(-9,sum(c4)/100,str(round(sum(c4)/100,3)),fontdict={'size':'10','color':'hotpink'})
ln1, = plt.plot(x, c1,color = 'yellowgreen') 
ln2, = plt.plot(x, c2,color = 'cornflowerblue')
ln3, = plt.plot(x, c3,color = 'mediumpurple')
ln4, = plt.plot(x, c4,color = 'hotpink')
plt.legend(handles = [ln1, ln2, ln3, ln4], labels = [
                                                'Spell Checker',
                                                'Spell Checker+Max Lev distance = 1',
                                                'Spell Checker+Max Lev distance = 2',
                                                'Spell Checker+Max Lev distance = 3',])

plt.plot([-5,105], [sum(c1)/100,sum(c1)/100], color='yellowgreen', linestyle='--')
plt.plot([-5,105], [sum(c2)/100,sum(c2)/100], color='cornflowerblue', linestyle='--')
plt.plot([-5,105], [sum(c3)/100,sum(c3)/100], color='mediumpurple', linestyle='--')
plt.plot([-5,105], [sum(c4)/100,sum(c4)/100], color='hotpink', linestyle='--')
plt.show()
