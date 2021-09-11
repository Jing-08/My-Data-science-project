import os

resultfiles = []
for file in os.listdir(r'data1\\1897\\'):
    #files.append(path+file)
    if file[0:8] not in resultfiles:
        resultfiles.append(file[0:8])

path = []
for i in resultfiles:
    path.append('data1\\sn85044812\\'+i[0:4]+'\\'+i[4:6]+'\\'+i[6:9]+'\\ed-1\\seq-1\\ocr.txt')

for i in path:
    
    with open(i, 'r',encoding = 'utf-8') as f:
        text = f.read()
    print(text)
    print(type(text))
    with open('answer.txt','a',encoding = 'utf-8') as f1:
        f1.write(text)

