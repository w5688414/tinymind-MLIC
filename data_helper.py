
import pandas as pd
import numpy as np
from PIL import Image

def hash_tag(filepath):
    fo = open(filepath, "r",encoding='utf-8')
    hash_tag = {}
    i = 0
    for line in fo.readlines():     #依次读取每行  
        line = line.strip()         #去掉每行头尾空白  
        hash_tag[i] = line
        i += 1
    return hash_tag

def load_ytrain(filepath):  
    y_train = np.load(filepath)
    y_train = y_train['tag_train']
    
    return y_train

def arr2tag(arr):
    tags = []
    for i in range(arr.shape[0]):
        tag = []
        index = np.where(arr[i] > 0.5)  
        index = index[0].tolist()
        tag =  [hash_tag[j] for j in index]

        tags.append(tag)
    return tags

filepath = "valid_tags.txt"
hash_tag = hash_tag(filepath)