import re
import string
from unidecode import unidecode
from nltk import tokenize
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


contraction_patterns = [(r'won\'t','will not'),(r'can\'t','cannot'),(r'i\'m','i am'),(r'isn\'t','is not'),
                        (r'(\w+)\'ll','\g<1> will'),(r'(\w+)n\'t','\g<1> not'),(r'(\w+)\'ve','\g<1> have'),
                        (r'(\w+)\'re','\g<1> are'),(r'(\w+)\'d','\g<1> would'),(r' has ',' have '),
                        (r'&','and'),(r'dammit','damn it'),(r'dont','do not'),(r'wont','will not')]

def clean_text(text):

    #处理英文缩写
    patterns = [(re.compile(regex),repl) for (regex,repl) in contraction_patterns]
    for (pattern,repl) in patterns:
        (text,count) = re.subn(pattern,repl,text)

    # 删除数字
    text = re.sub('[\d]','',text)
    # 删除's
    text = re.sub("\'s ", " ", text)
    # 删除标点符号
    punctuation_string = string.punctuation
    for i in punctuation_string:
        text = text.replace(i,' ')
    text = re.sub(" -- "," ",text)
    # 将法文字母转变为英文字母
    text = unidecode(text)

    # 分词
    text_split = tokenize.word_tokenize(text)
    # 去除停用词
    # text = [word for word in text_split if word not in stoplist]
    text = [word for word in text_split]

    # 词形还原
    text = [wnl.lemmatize(word) for word in text]

    return " ".join(text)

if __name__ == '__main__':
    f = open("../data/rt-polaritydata/rt-polarity.neg","r",encoding='Windows-1252')
    lines = f.readlines()
    f.close()
    for line in lines:
        newtextline = clean_text(line)
        newfile = open('../data/rt-polaritydata/neg.txt', 'a',encoding='Windows-1252')
        newfile.write(newtextline + '\n')
        newfile.close()

    f = open("../data/rt-polaritydata/rt-polarity.pos","r",encoding='Windows-1252')
    lines = f.readlines()
    f.close()
    for line in lines:
        newtextline = clean_text(line)
        newfile = open('../data/rt-polaritydata/pos.txt', 'a',encoding='Windows-1252')
        newfile.write(newtextline + '\n')
        newfile.close()