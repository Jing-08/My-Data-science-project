import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_excel('ngram.xlsx',index_col=0)
#df.to_csv('ngram.csv',encoding='utf-8')
df = pd.read_csv('ngram.csv',index_col=0)
x = [i for i in range(100)]

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1, 1, 1)
plt.grid(linestyle='--')


'''
# CER
plt.title('Spell correction: N-gram with Levenshtein distance (CER)')  # 折线图标题
#plt.ylim(0, 1.6)
plt.xlim(-15,108)
plt.xlabel('Newspaper page number')
plt.ylabel('Difference value (%)')
plt.text(-8,df.mean()[0],str(round(df.mean()[0],2)),fontdict={'size':'10','color':'salmon'})
plt.text(-8,df.mean()[2],str(round(df.mean()[2],2)),fontdict={'size':'10','color':'teal'})
plt.text(-8,df.mean()[4],str(round(df.mean()[4],2)),fontdict={'size':'10','color':'gold'})
plt.text(100,df.mean()[6]-0.2,str(round(df.mean()[6],2)),fontdict={'size':'10','color':'dimgrey'})
ln1, = plt.plot(x, df[['UNIGRAM+edit=1(CER)']],color = 'salmon') 
ln2, = plt.plot(x, df[['UNIGRAM+edit=2(CER)']],color = 'teal')
ln3, = plt.plot(x, df[['BIGRAM+edit=1(CER)']],color = 'gold')
ln4, = plt.plot(x, df[['BIGRAM+edit=2(CER)']],color = 'dimgrey')
plt.legend(handles = [ln1, ln2, ln3, ln4], labels = ['UNIGRAM+Max Lev distance=1',
                                                     'UNIGRAM+Max Lev distance=2',
                                                     'BIGRAM+Max Lev distance=1',
                                                     'BIGRAM+Max Lev distance=2'
                                                ])
plt.plot([-5,105], [df.mean()[0],df.mean()[0]], color='salmon', linestyle='--')
plt.plot([-5,105], [df.mean()[2],df.mean()[2]], color='teal', linestyle='--')
plt.plot([-5,105], [df.mean()[4],df.mean()[4]], color='gold', linestyle='--')
plt.plot([-5,105], [df.mean()[6],df.mean()[6]], color='dimgrey', linestyle='--')

plt.show()
'''
# WER
plt.title('Spell correction: N-gram with Levenshtein distance (WER)')  # 折线图标题
#plt.ylim(0, 1.6)
plt.xlim(-15,108)
plt.xlabel('Newspaper page number')
plt.ylabel('Difference value (%)')
plt.text(-8,df.mean()[1],str(round(df.mean()[1],2)),fontdict={'size':'10','color':'salmon'})
plt.text(100,df.mean()[3],str(round(df.mean()[3],2)),fontdict={'size':'10','color':'teal'})
plt.text(-8.7,df.mean()[5]-0.05,str(round(df.mean()[5],2)),fontdict={'size':'10','color':'gold'})
plt.text(100,df.mean()[7]-0.35,str(round(df.mean()[7],2)),fontdict={'size':'10','color':'dimgrey'})
ln1, = plt.plot(x, df[['UNIGRAM+edit=1(WER)']],color = 'salmon') 
ln2, = plt.plot(x, df[['UNIGRAM+edit=2(WER)']],color = 'teal')
ln3, = plt.plot(x, df[['BIGRAM+edit=1(WER)']],color = 'gold')
ln4, = plt.plot(x, df[['BIGRAM+edit=2(WER)']],color = 'dimgrey')
plt.legend(handles = [ln1, ln2, ln3, ln4], labels = ['UNIGRAM+Max Lev distance=1',
                                                     'UNIGRAM+Max Lev distance=2',
                                                     'BIGRAM+Max Lev distance=1',
                                                     'BIGRAM+Max Lev distance=2'
                                                ])
plt.plot([-5,105], [df.mean()[1],df.mean()[1]], color='salmon', linestyle='--')
plt.plot([-5,105], [df.mean()[3],df.mean()[3]], color='teal', linestyle='--')
plt.plot([-5,105], [df.mean()[5],df.mean()[5]], color='gold', linestyle='--')
plt.plot([-5,105], [df.mean()[7],df.mean()[7]], color='dimgrey', linestyle='--')

plt.show()

