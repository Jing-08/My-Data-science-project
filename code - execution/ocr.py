import io
import os
import glob
from PIL import Image
import cv2
import pytesseract
import numpy
import pandas as pd

import time
from collections import defaultdict
from joblib import Parallel, delayed

import Levenshtein as Lev
from nltk.corpus import stopwords

import re
import fastwer

def image2text(filepath):
    
    '''
    This function contain two parts: pre-processing and OCR

    '''
    # BEGIN TO PRE-PROCESS NEWSPAPER IMAGE
    image = cv2.imread(filepath)
    rows= image.shape[0]    #chang
    cols = image.shape[1]   #kuan
    print(rows,cols)
 
    cv2.imwrite('image.png',image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    # opening operation
    kernel = numpy.ones((2, 2), numpy.uint8)
    erosion = cv2.erode(binary_image, kernel, iterations=1) 
    dilate = cv2.dilate(erosion, kernel, iterations=2)
    
    image_arr = numpy.array(dilate)  

    # BEGIN TO SEGMENT NEWSPAPER
    # line
    l = image_arr[int(450):int(500),]
    l1 = Image.fromarray(l)
    l1.save('line.png')
    
    im_line = cv2.imread('line.png')
    gray = cv2.cvtColor(im_line, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 1000 # height/32
    maxLineGap = 5 # height/40
    lines = cv2.HoughLinesP(edges, 1, numpy.pi/360, 30,
                            minLineLength, maxLineGap)
    new = []
    for line in lines:
        new.append(line[0][0])
        x1,y1,x2,y2 = line[0]
        cv2.line(im_line, (x1, y1), (x2, y2),(0,255,0), 2)
    old = sorted(new)
    for i in old:
        if i > 200:
            left = i
            break
    if min(old) > 300:
        left = 230
    for j in range(len(old)):
        if old[len(old)-1-j] < cols-200:
            right = old[len(old)-1]
            break
    if right < 4400:
        right = 4400
    print(left,right)
    cv2.imwrite('r.png', im_line)
    cai = image_arr[:,left:right]
    c = Image.fromarray(cai)
    c.save('jiancai.png')

    each = int((right-left)/6)
    if each not in range(690, 700):
        each = 694
    # title
    image_tar1 = image_arr[int(0):int(500),:]
    im1 = Image.fromarray(image_tar1)
    im1.save('title.png')
    
    content0 = pytesseract.image_to_string(im1, lang = 'eng')
    
    # version number
    image_tar2 = image_arr[int(480):int(620),:]
    im2 = Image.fromarray(image_tar2)
    im2.save('version.png')
    content1 = pytesseract.image_to_string(im2, lang = 'eng')

    minleft=190
    t1 = time.time()
    # column 1
    image_tar3 = image_arr[int(600):,left - 10 : each + left + 10]
    im3 = Image.fromarray(image_tar3)
    im3.save('1 columns.png')
    content2 = pytesseract.image_to_string(im3, lang = 'eng')
    # column 2
    image_tar3 = image_arr[int(600):,each + left - 10: 2 * each + left + 10]
    im3 = Image.fromarray(image_tar3)
    im3.save('2 columns.png')
    content3 = pytesseract.image_to_string(im3, lang = 'eng')
    # column 3
    image_tar3 = image_arr[int(600):,2 * each + left - 10: 3 * each + left+10]
    im3 = Image.fromarray(image_tar3)
    im3.save('3 columns.png')
    content4 = pytesseract.image_to_string(im3, lang = 'eng')
    # column 4
    image_tar3 = image_arr[int(600):,3 * each + left - 10: 4 * each + left +10]
    im3 = Image.fromarray(image_tar3)
    im3.save('4 columns.png')
    content5 = pytesseract.image_to_string(im3, lang = 'eng')
    # column 5
    image_tar3 = image_arr[int(600):,4 * each + left - 10: 5 * each + left + 10]
    im3 = Image.fromarray(image_tar3)
    im3.save('5 columns.png')
    content6 = pytesseract.image_to_string(im3, lang = 'eng')
    # column 6
    image_tar3 = image_arr[int(600):,5 * each + left - 10: 6 * each + left + 100]
    im3 = Image.fromarray(image_tar3)
    im3.save('6 columns.png')
    content7 = pytesseract.image_to_string(im3, lang = 'eng')
    t2 = time.time()
    
    zongtime.append(round(t2-t1,2))
    print(zongtime)

    print('begin ocr')
    content = content0 + content1 + content2 + content3 + content4 + content5 + content6 + content7 
    
    return content , set(content.split(" "))

def ocr2text(filepath):
    
    '''
    This part get the ocr results provide by Chronicling America.

    '''
    text = None
    with open(filepath, 'r', encoding = 'utf-8') as f:
        text = f.readlines()
        #print('Real text:',text)

    words = set("".join(text).replace("\n", " ").split(" "))
    
    return text, words

def list_file_pair():
    
    '''
    This function pair the newspaper images and ocrs results by
    recognizing file names.
    
    '''
    newname = []
    
    pdffiles = defaultdict(list)
    for path in sorted(glob.glob("data1\\1895")):
        for x in sorted(os.listdir(path)):
            if x[0:9] not in newname:
                newname.append(x[0:9])
                key = x.split("_")[0][:8]
                pdffiles[key].append(os.path.join(path, x))
    
    ocrfiles = defaultdict(list)
    
    def getfiles(filepath, file_format='.txt'):
        for x in sorted(os.listdir(filepath), key=lambda x: int(x.split('-')[-1]) if('seq' in x) else x):
            x_file = os.path.join(filepath, x)
            if os.path.isfile(x_file) and os.path.splitext(x_file)[1] == file_format:
                key = "".join(filepath.split("\\")[2:5])
                ocrfiles[key].append(x_file)
            if os.path.isdir(x_file):
                getfiles(x_file)
    
    #getfiles("data1\\sn85044812\\1895\\")
    k1 = set(pdffiles.keys())
    k2 = set(ocrfiles.keys())

    #print('diff: ',  (k2|k1) - (k2&k1))
    pairs = []
    for k in sorted(k2 & k1):
        for x,y in zip(pdffiles[k], ocrfiles[k]):
            pairs.append((x, y))
    
    pdffiles = defaultdict(list)
    for path in sorted(glob.glob("data1\\189601")):
        for x in sorted(os.listdir(path)):
            if x[0:9] not in newname:
                newname.append(x[0:9])
                key = x.split("_")[0][:8]
                pdffiles[key].append(os.path.join(path, x))
    
    ocrfiles = defaultdict(list)
    getfiles("data1\\sn85044812\\1896\\")
    k1 = set(pdffiles.keys())
    k2 = set(ocrfiles.keys())

    #print('diff: ',  (k2|k1) - (k2&k1))
    for k in sorted(k2 & k1):
        for x,y in zip(pdffiles[k], ocrfiles[k]):
            pairs.append((x, y))
    print(len(pairs))
    
    return pairs

def calc_acc(x, y):
    #pred_words = pdf2text(x)
    pred_words = image2text(x)
    true_words = ocr2text(y)
    p = len(pred_words & true_words) / len(true_words)
    return p


def re_punc(text):
    
    punctuation = ',.;:!?"\'\&%$^*#@|`~/-_()[]{}=+<>“”’‘—'
    text = ''.join(c for c in text if c not in punctuation)
    return text.strip()

def re_2(text):

    text = text.split(" ")
    text = ' '.join(c for c in text if len(c) > 1)
    return text.strip()


if __name__ == '__main__':
    zongtime = []
    pairs = list_file_pair()

    #ps = Parallel(n_jobs=7)(delayed(calc_acc)(x, y) for x, y in pairs)
    #print(sum(ps)/len(ps))

    duration = []
    predch_len = []
    realch_len = []
    predwd_len = []
    realwd_len = []
    wrr = []
    sim = []
    WER = []
    CER = []
    keyWRR = []
    keyWER = []
    count = 0
    cerlist = []
    werlist = []
    cerlist2 = []
    werlist2 = []
    cerlist3 = []
    werlist3 = []
    cerlist4 = []
    werlist4 = []
    cerlist5 = []
    werlist5 = []
    for x,y in pairs:
        
        print(count)
        print('filename:', x)
        content0, pred_words = image2text(x)
        
        #print('ocr end')
        text0, true_words = ocr2text(y)

        #print('start print')
        '''
        content2 = content0.split(" ")
        content2 = [x for x in content2 if x != '']
        
        content = content0.replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n')
        content = re_punc(content)
        content = re_2(content)
        text = re_punc(' '.join(text0).replace("\n", " ").replace("  ", " "))
        text = re_2(text)
        
        # 1 collect run time
        #duration.append(round(t2-t1,2))
        # 2 collect pred character len
        predch_len.append(len(content))
        # 3 collect pred word len
        predwd_len.append(len(content.split(" ")))
        # 4 collect real words len
        #text1 = " ".join(text)    
        text2 = text.split(" ")
        realwd_len.append(len(text2))
        # 5 collect real character len
        realch_len.append(len(text))
        
        # 7 collect word error rate (WER)
        # 8 collect character error rate (CER)
        cer = round(fastwer.score_sent(content,text,char_level=True),2)
        CER.append(cer)
        wer = round(fastwer.score_sent(content,text),2)
        WER.append(wer)
        '''

        # 9 collect levenshtein ratio
        #sim.append(Lev.ratio(content,str(text)))
        

    
        filename = 'ocr//' + str(x[11:26]) + '.txt'
        f = open(filename, 'w', encoding = 'utf-8')
        f.write(content0)
        '''
        with open(filename, 'r', encoding = 'utf-8') as f:
            pred = f.readlines()
        content = ' '.join(pred).replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n')
        content = re_punc(content)
        content = re_2(content)
        wer = round(fastwer.score_sent(content,text),3)
        WER.append(wer)
        cer = round(fastwer.score_sent(content,text,char_level=True),3)
        CER.append(cer)
        # 6 collect word recognize rate (WRR)
        wrr.append(round((len(set(content.split(" ")))&len(set(real.split(" "))))/len(set(content.split(" "))),3))
        # 9 collect keywords error rate
        stop_words = stopwords.words('english')
        s1_key = [word for word in content.split(" ") if word not in stop_words]
        s2_key = [word for word in real.split(" ") if word not in stop_words]
        #print(set(content.split(" "))&set(real.split(" ")))
        keyWRR.append(round((len(set(s1_key))&len(set(s2_key)))/len(set(content.split(" "))),3))
        key1 = " ".join(s1_key)
        key2 = " ".join(s2_key)
        keyWER.append(fastwer.score_sent(key1,key2),3)
        
        
        #print('RUN TIME:',duration)
        
        print('PREDCH_LEN:',predch_len)
        print('PREDWD_LEN:',predwd_len)
        print('REALWD_LEN:',realwd_len)
        print('REALCH_LEN:',realch_len)
        print('WRR:',wrr)
        print('WER without stopwords:', keyWER)
        print('WRR without stopwords:', keyWRR)
        print('LEVENSHTEIN:',sim)
        print('CER:',CER)
        print('WER:',WER)
        '''
        #lst0, lst1, f = get_word_count(text0,
        #                               content0.replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n'))
        '''
        print('the first type')
        print('content:',content0)
        print('text:',' '.join(text0))
        cerr = round(fastwer.score_sent(content0,' '.join(text0).replace("\n", " "),char_level=True),2)
        cerlist.append(cerr)
        werr = round(fastwer.score_sent(content0,' '.join(text0).replace("\n", " ")),2)
        werlist.append(werr)
        print('newcer:',cerlist)
        print('newwer:',werlist)
        print('------------------------------------------------------------')
        print('the second type')
        content = content0.replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n')
        print('content:',content)
        print('text:',text0)
        cerr2 = round(fastwer.score_sent(content,' '.join(text0).replace("\n", " ").replace("  ", " "),char_level=True),2)
        cerlist2.append(cerr2)
        werr2 = round(fastwer.score_sent(content,' '.join(text0).replace("\n", " ").replace("  ", " ")),2)
        werlist2.append(werr2)
        print('newcer2:',cerlist2)
        print('newwer2:',werlist2)
        print('------------------------------------------------------------')
        print('the third type')
        content = content0.replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n')
        content = re_punc(content)
        print('content:',content)
        text = re_punc(' '.join(text0).replace("\n", " ").replace("  ", " "))
        print('text:',text)
        cerr3 = round(fastwer.score_sent(content,text,char_level=True),2)
        cerlist3.append(cerr3)
        werr3 = round(fastwer.score_sent(content,text),2)
        werlist3.append(werr3)
        print('newcer3:',cerlist3)
        print('newwer3:',werlist3)
        print('------------------------------------------------------------')
        print('the fourth type')
        content = content0.replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n')
        content = re_2(content)
        print('content:',content)
        text = re_2(' '.join(text0).replace("\n", " ").replace("  ", " "))
        print('text:',text)
        cerr4 = round(fastwer.score_sent(content,text,char_level=True),2)
        cerlist4.append(cerr4)
        werr4 = round(fastwer.score_sent(content,text),2)
        werlist4.append(werr4)
        print('newcer4:',cerlist4)
        print('newwer4:',werlist4)
        print('------------------------------------------------------------')
        print('the fifth type')
        content = content0.replace("\n", " ").replace("  ", " ").replace("-", '').replace("\n\n",'\n')
        content = re_punc(content)
        content = re_2(content)
        print('content:',content)
        text = re_punc(' '.join(text0).replace("\n", " ").replace("  ", " "))
        text = re_2(text)
        print('text:',text)
        cerr5 = round(fastwer.score_sent(content,text,char_level=True),2)
        cerlist5.append(cerr5)
        werr5 = round(fastwer.score_sent(content,text),2)
        werlist5.append(werr5)
        print('newcer5:',cerlist5)
        print('newwer5:',werlist5)
        print('------------------------------------------------------------')
        print('------------------------------------------------------------')
        print('------------------------------------------------------------')
        '''
        count += 1
        
        #print('KEYWER:',keyWER)
    # CER AND WER after different processing methods
    '''
    df = pd.DataFrame()
    df['cer1'] = cerlist
    df['cer2'] = cerlist2
    df['cer3'] = cerlist3
    df['cer4'] = cerlist4
    df['cer5'] = cerlist5
    df['wer1'] = werlist
    df['wer2'] = werlist2
    df['wer3'] = werlist3
    df['wer4'] = werlist4
    df['wer5'] = werlist5
    df.to_csv('rror rate.csv', header = True)
    print(df)
    '''

    # metrics to evaluate Tesseract performance
    '''
    df = pd.DataFrame()
    df['run time'] = zongtime
    df['ocr char len'] = predch_len
    df['ocr word len'] = predwd_len
    df['real char len'] = realch_len
    df['real word len'] = realwd_len
    df['WER'] = WER
    df['CER'] = CER
    df['WRR'] = wrr
    df['Levenshtein'] = sim
    df['WER without stopwords'] = keyWER
    df['WRR without stopwords'] = keyWRR
    df.to_csv('stat.csv',header = True)
    '''

    
