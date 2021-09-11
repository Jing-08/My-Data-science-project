import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spellchecker.csv',index_col=0)
#df.to_csv('spellchecker.csv',encoding='utf-8')
x = [i for i in range(100)]


fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1, 1, 1)
plt.grid(linestyle='--')
'''
# CER : only use Lev distance
plt.title('Spell correction: Levenshtein(Lev) distance (CER)')  
plt.ylim(0, 1.6)
plt.xlim(-15,108)

plt.xlabel('Newspaper page number')
plt.ylabel('Difference value (%)')
plt.text(-8,df.mean()[0],str(round(df.mean()[0],2)),fontdict={'size':'10','color':'yellowgreen'})
plt.text(-8,df.mean()[2],str(round(df.mean()[2],2)),fontdict={'size':'10','color':'cornflowerblue'})
plt.text(-8,df.mean()[4],str(round(df.mean()[4],2)),fontdict={'size':'10','color':'orangered'})
ln1, = plt.plot(x, df[['Lev=1(CER)']],color = 'yellowgreen')  
ln2, = plt.plot(x, df[['Lev=2(CER)']],color = 'cornflowerblue')
ln3, = plt.plot(x, df[['Lev=3(CER)']],color = 'orangered')
plt.legend(handles = [ln1, ln2, ln3], labels = ['Max Lev distance = 1',
                                                'Max Lev distance = 2',
                                                'Max Lev distance = 3',
                                                ])
plt.plot([-5,105], [df.mean()[0],df.mean()[0]], color='yellowgreen', linestyle='--')
plt.plot([-5,105], [df.mean()[2],df.mean()[2]], color='cornflowerblue', linestyle='--')
plt.plot([-5,105], [df.mean()[4],df.mean()[4]], color='orangered', linestyle='--')
plt.show()


# CER : use Lev + spellchecker
plt.title('Spell correction: SpellChecker with Levenshtein(Lev) distance (CER)')  # 折线图标题
plt.ylim(0.4, 1.6)
plt.xlim(-15,108)
plt.grid(linestyle='--')
plt.xlabel('Newspaper page number')
plt.ylabel('Difference value (%)')
plt.text(-9,df.mean()[6],str(round(df.mean()[6],2)),fontdict={'size':'12','color':'yellowgreen'})
plt.text(-9,df.mean()[8],str(round(df.mean()[8],2)),fontdict={'size':'12','color':'cornflowerblue'})
plt.text(-9,df.mean()[10],str(round(df.mean()[10],2)),fontdict={'size':'12','color':'mediumpurple'})
plt.text(-9,df.mean()[12],str(round(df.mean()[12],2)),fontdict={'size':'12','color':'hotpink'})
ln4, = plt.plot(x, df[['SpellChecker(CER)']], color = 'yellowgreen')
ln5, = plt.plot(x, df[['SpellChecker+Lev=1(CER)']], color = 'cornflowerblue')
ln6, = plt.plot(x, df[['SpellChecker+Lev=2(CER)']], color = 'mediumpurple')
ln7, = plt.plot(x, df[['SpellChecker+Lev=3(CER)']], color = 'hotpink')

plt.legend(handles = [ln4, ln5, ln6, ln7], labels = [
                                                'Spell Checker',
                                                'Spell Checker+Max Lev distance = 1',
                                                'Spell Checker+Max Lev distance = 2',
                                                'Spell Checker+Max Lev distance = 3',])

plt.plot([-5,105], [df.mean()[6],df.mean()[6]], color='yellowgreen', linestyle='--')
plt.plot([-5,105], [df.mean()[8],df.mean()[8]], color='cornflowerblue', linestyle='--')
plt.plot([-5,105], [df.mean()[10],df.mean()[10]], color='mediumpurple', linestyle='--')
plt.plot([-5,105], [df.mean()[12],df.mean()[12]], color='hotpink', linestyle='--')

plt.show()


# WER : only use Lev distance
plt.title('Spell correction: Levenshtein(Lev) distance (WER)')  # 折线图标题
plt.ylim(0, 6.1)
plt.xlim(-10,110)
plt.grid(linestyle='--')
plt.xlabel('Newspaper page number')
plt.ylabel('Difference value (%)')
plt.text(-8,df.mean()[1],str(round(df.mean()[1],2)),fontdict={'size':'10','color':'yellowgreen'})
plt.text(105,df.mean()[3],str(round(df.mean()[3],2)),fontdict={'size':'10','color':'cornflowerblue'})
plt.text(-8,df.mean()[5],str(round(df.mean()[5],2)),fontdict={'size':'10','color':'orangered'})
ln1, = plt.plot(x, df[['Lev=1(WER)']],color = 'yellowgreen')  # 绘制折线图，添加数据点，设置点的大小
ln2, = plt.plot(x, df[['Lev=2(WER)']],color = 'cornflowerblue')
ln3, = plt.plot(x, df[['Lev=3(WER)']],color = 'orangered')
plt.legend(handles = [ln1, ln2, ln3], labels = ['Max Lev distance = 1',
                                                'Max Lev distance = 2',
                                                'Max Lev distance = 3',
                                                ])
plt.plot([-5,105], [df.mean()[1],df.mean()[1]], color='yellowgreen', linestyle='--')
plt.plot([-5,105], [df.mean()[3],df.mean()[3]], color='cornflowerblue', linestyle='--')
plt.plot([-5,105], [df.mean()[5],df.mean()[5]], color='orangered', linestyle='--')
plt.show()

'''
# WER : use Lev + spellchecker
plt.ylim(2, 6.5)
plt.xlim(-15,108)
plt.grid(linestyle='--')
plt.xlabel('Newspaper page number')
plt.ylabel('Difference value (%)')
plt.title('Spell correction: SpellChecker with Levenshtein(Lev) distance (WER)')  # 折线图标题
plt.text(-10,df.mean()[7],str(round(df.mean()[7],2)),fontdict={'size':'12','color':'yellowgreen'})
plt.text(-10,df.mean()[9],str(round(df.mean()[9],2)),fontdict={'size':'12','color':'cornflowerblue'})
plt.text(-10,df.mean()[11],str(round(df.mean()[11],2)),fontdict={'size':'12','color':'mediumpurple'})
plt.text(-10,df.mean()[13],str(round(df.mean()[13],2)),fontdict={'size':'12','color':'hotpink'})
ln4, = plt.plot(x, df[['SpellChecker(WER)']], color = 'yellowgreen')
ln5, = plt.plot(x, df[['SpellChecker+Lev=1(WER)']], color = 'cornflowerblue')
ln6, = plt.plot(x, df[['SpellChecker+Lev=2(WER)']], color = 'mediumpurple')
ln7, = plt.plot(x, df[['SpellChecker+Lev=3(WER)']], color = 'hotpink')

plt.legend(handles = [ln4, ln5, ln6, ln7], labels = [
                                                'Spell Checker',
                                                'Spell Checker+Max Lev distance = 1',
                                                'Spell Checker+Max Lev distance = 2',
                                                'Spell Checker+Max Lev distance = 3',])

plt.plot([-5,105], [df.mean()[7],df.mean()[7]], color='yellowgreen', linestyle='--')
plt.plot([-5,105], [df.mean()[9],df.mean()[9]], color='cornflowerblue', linestyle='--')
plt.plot([-5,105], [df.mean()[11],df.mean()[11]], color='mediumpurple', linestyle='--')
plt.plot([-5,105], [df.mean()[13],df.mean()[13]], color='hotpink', linestyle='--')

plt.show()


