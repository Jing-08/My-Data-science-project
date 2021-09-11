with open('allwrong.txt', 'r', encoding = 'utf-8') as f:
    wrong = f.readlines()

wr = wrong[0]
wr = wr.split(',')
'''
for c in ['a','b','c','d','e','f','g','h','i','j','k','l',
                  'm','n','o','p','q','r','s','t','u','v','w','x','y','z']:
    print(c)
    dui = {}
    dd = {}
    for i in wr:
        if i[2] == c:
            if i[3].isalpha():
                if i[3:-1] in dui:
                    dui[i[3:-1]] += 1
                else:
                    dui[i[3:-1]] = 1
    for key in dui:
        dd[key] = round(dui[key]/sum(dui.values()),4)

    print(dd)        
'''    
for c in ['a','b','c','d','e','f','g','h','i','j','k','l',
                  'm','n','o','p','q','r','s','t','u','v','w','x','y','z','ﬂ',
          'ﬁ','é']:
    dui = {}
    dd = {}
    print(c)
    for i in wr:
        if i[3] == c:
            if i[2].isalpha():
                if i[2] in dui:
                    dui[i[2]] += 1
                else:
                    dui[i[2]] = 1
    for key in dui:
        dd[key] = round(dui[key]/sum(dui.values()),4)
    if 'J' in dd:
        print(dd['J']) 
