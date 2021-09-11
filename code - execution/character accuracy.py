
import pandas as pd             
import matplotlib.pyplot as plt

with open('ff.txt', 'r', encoding = 'utf-8') as f:
    char_acc = f.readlines()

char = []
charnum = []
for i in char_acc:
    if i != '' and i != '\n':
        newi = i.split(" ")
        char.append(newi[0])
        charnum.append(newi[1][0:-1])


char1 = list(set(char))
aa = [[] for i in range(len(char1))]
for n1,c1 in enumerate(char1):
    for n,c in enumerate(char):
        if c == c1:
            aa[n1].append(n)

for ni,i in enumerate(aa):
    for nj,j in enumerate(i):
        weizhi = aa[ni][nj]
        aa[ni][nj] = float(charnum[weizhi])

x = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
                    's','t','u','v','w','x','y','z']
y = [258,258,258,258,258,258,258,258,258,7,257,252,258,
     258,258,258,15,258,
    258,258,258,258,258,221,258,8]
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(x=x, height=y,color='orange')
ax.set_title("Number of valid counts per lowercase letter", fontsize=15)

x = ['A','B','C','D','E','F','G','H','I','J',
     'K','L','M','N','O','P','Q','R',
    'S','T','U','V','W','X','Y','Z']
y = [258,242,255,193,245,118,156,238,258,79,
     9,188,255,228,173,183,37,209,
    255,258,21,2,201,57,3,27]
fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(x=x, height=y,color = 'skyblue')
ax.set_title("Number of valid counts per capital letter", fontsize=15)
# 某个字母的位置

charweizhi = char1.index("a")
a = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("b")
b = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("c")
c = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("d")
d = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("e")
e = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("f")
f = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("g")
g = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("h")
h = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("i")
i = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("j")
j = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("k")
k = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("l")
l = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("m")
m = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("n")
n = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("o")
o = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("p")
p = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("q")
q = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("r")
r = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("s")
s = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("t")
t = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("u")
u = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("v")
v = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("w")
w = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("x")
x = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("y")
y = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("z")
z = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))


fig = plt.figure(figsize=(20,5))
plt.ylabel('accuracy')
plt.title('Recognition accuracy of lowercase letters')
plt.boxplot((a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z),
            patch_artist='blue',showcaps=True,showfliers=True,meanline=True,
            boxprops={'color':'black','facecolor':'orange'} ,
            flierprops={'marker':'D','markerfacecolor':'yellowgreen','markersize':4},
            meanprops={'marker':'D','markerfacecolor':'yellowgreen',',arkersize':4},
            medianprops={'linestyle':'--'},
            labels=('a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
                    's','t','u','v','w','x','y','z'))

plt.grid(linestyle="--", alpha=0.3,color = 'blue')
plt.show()




charweizhi = char1.index("A")
a = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("B")
b = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("C")
c = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("D")
d = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("E")
e = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("F")
f = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("G")
g = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("H")
h = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("I")
i = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("J")
j = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("K")
k = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("L")
l = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("M")
m = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("N")
n = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("O")
o = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("P")
p = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("Q")
q = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("R")
r = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("S")
s = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("T")
t = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("U")
u = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("V")
v = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("W")
w = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("X")
x = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("Y")
y = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
charweizhi = char1.index("Z")
z = aa[charweizhi]
print(sum(aa[charweizhi])/len(aa[charweizhi]))
'''
import pandas as pd                 
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,5))
plt.ylabel('accuracy')
plt.title('Recognition accuracy of capital letters')
plt.boxplot((a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z),
            patch_artist='blue',showcaps=True,showfliers=True,meanline=True,
            boxprops={'color':'black','facecolor':'skyblue'} ,
            flierprops={'marker':'D','markerfacecolor':'green','markersize':4},
            meanprops={'marker':'D','markerfacecolor':'green',',arkersize':4},
            medianprops={'linestyle':'--'},
            labels=('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
                    'S','T','U','V','W','X','Y','Z'))

plt.grid(linestyle="--", alpha=0.3,color = 'blue')
plt.show()
'''
