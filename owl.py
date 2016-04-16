import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from nltk.stem.porter import PorterStemmer

from joblib import Memory
memory = Memory(cachedir='/tmp/joblib')

stemmer = PorterStemmer()

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing']
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}


def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"


def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))


def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r


def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt


def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


@memory.cache
def stem_col(column):
    return column.map(lambda x: str_stem(x))


def get_features(df):
    feat = pd.DataFrame(index=df.index)

    owl_search_term = stem_col(df.search_term)
    owl_product_title = stem_col(df.product_title)
    owl_product_description = stem_col(df.product_description)
    owl_brand = stem_col(df.brand)
    owl_product_info = owl_search_term + '\t' + owl_product_title + '\t' + \
        owl_product_description

    feat['owl_len_of_query'] = owl_search_term.map(lambda x:len(x.split())).astype(np.int64)
    feat['owl_len_of_title'] = owl_product_title.map(lambda x:len(x.split())).astype(np.int64)
    feat['owl_len_of_description'] = owl_product_description.map(lambda x:len(x.split())).astype(np.int64)
    feat['owl_len_of_brand'] = owl_brand.map(lambda x:len(x.split())).astype(np.int64)

    owl_search_term = owl_product_info.map(
        lambda x:seg_words(x.split('\t')[0], x.split('\t')[1]))

    feat['owl_query_in_title'] = owl_product_info.map(
        lambda x:str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
    feat['owl_query_in_description'] = owl_product_info.map(
        lambda x:str_whole_word(x.split('\t')[0], x.split('\t')[2], 0)) 
    
    feat['owl_query_last_word_in_title'] = owl_product_info.map(
        lambda x:str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
    feat['owl_query_last_word_in_description'] = owl_product_info.map(
        lambda x:str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))

    feat['owl_word_in_title'] = owl_product_info.map(
        lambda x:str_common_word(x.split('\t')[0], x.split('\t')[1]))
    feat['owl_word_in_description'] = owl_product_info.map(
        lambda x:str_common_word(x.split('\t')[0], x.split('\t')[2]))
    feat['owl_ratio_title'] = feat.owl_word_in_title / feat.owl_len_of_query
    feat['owl_ratio_description'] = feat.owl_word_in_description / feat.owl_len_of_query
    owl_attr = owl_search_term + '\t' + owl_brand
    feat['owl_word_in_brand'] = owl_attr.map(
        lambda x:str_common_word(x.split('\t')[0], x.split('\t')[1]))
    feat['owl_ratio_brand'] = feat.owl_word_in_brand / feat.owl_len_of_brand
    return feat


if __name__ == "__main__":
    from utils import get_data
    owl_features = get_features(get_data())
    owl_features.to_csv('/tmp/owl.csv')
