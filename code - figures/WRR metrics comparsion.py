import pandas as pd
import matplotlib.pyplot as plot
data = pd.read_csv("stat.csv")
WRRs = data[['WRR','WRR with upper', 'WRR without stopwords']]
w1 = []
print(WRRs.describe())
cha = []
for index,i in WRRs.iterrows():
    wrr = i['WRR']
    wrrup = i['WRR with upper']
    wrrsw = i['WRR without stopwords']
    cha.append(round(wrr - wrrsw, 2))
    if wrr > wrrup:
        if wrrup > wrrsw:
            w1.append('123')
        elif wrr > wrrsw > wrrup:
            w1.append('132')
            
        elif wrrsw > wrr:
            w1.append('312')
            
    else:
        if wrr > wrrsw:
            w1.append('213')

        elif wrrup > wrrsw > wrr:
            w1.append('231')
            
        elif wrrsw > wrrup:
            w1.append('321')
def LowerCount(alist,a,b): 
    num = 0
    for i in alist:
        if a<i<=b: 
            num+=1
    #percent = num/len(alist)
    return num#,percent

c1 = LowerCount(cha,0,0.05)
c2 = LowerCount(cha,0.05,0.1)
c3 = LowerCount(cha,0.1,100)
c4 = LowerCount(cha,-0.05,0)
c5 = LowerCount(cha,-0.1,-0.05)
c6 = LowerCount(cha,-100,-0.1)
labels = ['0%~5%', '5%~10%', '>10%', '<-10%','-10%~-5%','-5%~0%']
sizes = [c1,c2,c3,c6,c5,c4]
print(sizes)

color = ['thistle', 'plum', 'violet', 'lightsteelblue', 'cornflowerblue',
         'royalblue']
ex = [0,0,0,0.03,0.03,0.03]
patches, l_text, p_text = plot.pie(sizes, labels=labels, colors = color,explode = ex,
                                        autopct='%3.1f%%', shadow=False,
                                       startangle=90, pctdistance=0.6)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20

plot.axis('equal')
plot.legend(loc='upper left', bbox_to_anchor=(-0.1, 0.5))
plot.grid()
plot.show()

'''
pie chart of three WRR metrics comparsion

'''
c123 = w1.count('123')
c132 = w1.count('132')
c213 = w1.count('213')
c231 = w1.count('231')
c321 = w1.count('321')
c312 = w1.count('312')

labels = ['123', '132', '213', '231','321','312']
sizes = [int(c123),int(c132),int(c213),int(c231),int(c321),int(c312)]
print(sizes)

color = ['lightpink', 'darksalmon', 'lightskyblue', 'lightblue', 'greenyellow', 'aquamarine']
patches, l_text, p_text = plot.pie(sizes, labels=labels, colors = color,
                                        autopct='%3.1f%%', shadow=False,
                                       startangle=90, pctdistance=0.6)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20

plot.axis('equal')
plot.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
plot.grid()
plot.show()

