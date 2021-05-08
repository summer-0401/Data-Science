import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import chain, combinations
import random

class node:
    value: str
    children: dict
    def __init__(self):
        self.value = None
        self.children = dict()
    def __repr__(self):
        return "<node value:%s children:%s>" % (self.value, self.children)

def get_entropy(l_dict):
    ans=0
    total = sum(l_dict.values())
    for l in l_dict.values():
        p = float(l/total)
        ans-=p*np.log2(p)
    return ans

def cal_gain(data, attribute_list, c):
    total = len(data)
    class_labels = data.groupby(c).size().to_dict()
    info = get_entropy(class_labels)

    gain = dict()
    for A in attribute_list:#A:'city', a_list:{'city','quantity',...}
        A_labels = data.groupby(A).size().to_dict()

        info_A=float(0)

        for ai in A_labels:#Seoul, Inc, Busan, ...
            ai_data = data.loc[data[A]==ai]
            ai_class_labels = ai_data.groupby(c).size().to_dict()
            info_A+=A_labels[ai]*get_entropy(ai_class_labels)/total

        gain[A] = info-info_A

    return gain

def info_gain(data, attribute_list, c):
    gain = cal_gain(data, attribute_list, c)
    ans = max(gain, key=gain.get)
    return ans

def split_info(data, A):
    A_labels = data.groupby(A).size().to_dict()
    split_A=get_entropy(A_labels)
    return split_A

def gain_ratio(data, attribute_list, c):
    g_ratio = cal_gain(data, attribute_list, c)

    for g in g_ratio:
        g_ratio[g]/=split_info(data, g)
    ans = max(g_ratio, key=g_ratio.get)
    return ans

def Real_subset(_set):
    D1 = list(chain.from_iterable(combinations(_set, r) for r in range(1,int(len(_set)/2)+1)))
    return D1

def gini(A_labels, total):
    ans = float(1)
    for ai in A_labels:
        p = float(ai/total)
        ans -=(p*p)
    return ans

def gini_A(data, A, total, c):
    candidates=dict()
    A_labels = data.groupby(A).size().to_dict()
    c_labels = set(data[c])
    D1 = Real_subset(set(A_labels.keys())) #apple, orange, ...
    
    for d1 in D1:
        d1_nums = []
        d2_nums = []
        d2 = tuple(set(A_labels)-set(d1))
        for ci in c_labels:
            d1_nums.append(len(data[(data[A].isin(d1)) & (data[c]==ci)]))
            d2_nums.append(len(data[(data[A].isin(d2)) & (data[c]==ci)]))

        d1_total = sum(d1_nums)
        d2_total = sum(d2_nums)
        candidates[(d1, d2)]=d1_total*gini(d1_nums,d1_total)/total+d2_total*gini(d2_nums, d2_total)/total
    ans_key = min(candidates.keys(), key=candidates.get)
    ans_value = candidates[ans_key]
    return (ans_key, ans_value)   

def gini_index(data, attribute_list, c):
    total = len(data)
    index = gini(data.groupby(c).size().to_list(),total)

    delta_gini = dict()
    for A in attribute_list:
        tmp = gini_A(data, A, total, c)
        delta_gini[(A, tmp[0])] = index - tmp[1]
    ans = max(delta_gini, key=delta_gini.get)

    return ans

#training data, label
def Decision_Tree(data, attribute_list, c, t, measure):
    if len(data.groupby(c).size())==1:
        ans = node()
        ans.value = c
        ans.children['#class_label'] = data.iat[-1,-1]

        return ans
    
    if len(data)<4 or not attribute_list:
        ans = node()
        ans.value = c
        ans.children['#class_label'] = data.groupby(c).size().idxmax()
        return ans

    tmp = node()
    if measure == 'gini':
        gini_select = gini_index(data, attribute_list, c)
        select = gini_select[0]
        attribute_list.remove(select)
        select_labels = gini_select[1]

        for s in select_labels:
            if not len((data.loc[data[select].isin(s)])):
                tmp.children[data.groupby(c).size().idxmax()] = None
            else:
                tmp.children[s] = Decision_Tree(data.loc[data[select].isin(s)], attribute_list, c, tmp,  measure)

    else:
        if measure == 'info':
            select = info_gain(data, attribute_list, c)
        else:
            select = gain_ratio(data, attribute_list, c)

        attribute_list.remove(select)
        select_labels = data.groupby(select).size().index.to_list()

        for s in select_labels:
            if not len((data.loc[data[select]==s])):
                tmp.children[data.groupby(c).size().idxmax()] = None
            else:
                tmp.children[s] = Decision_Tree(data.loc[data[select]==s], attribute_list, c, tmp, measure)

    tmp.value = select

    attribute_list.append(select)
    return tmp

def categorization(datum, tree, attributes, c, measure):
    if tree.value == c:
        return tree.children['#class_label']
        
    index = datum[attributes.index(tree.value)]
    if measure == 'gini':
        for t in tree.children:
            if index in t:
                return categorization(datum, tree.children[t], attributes, c, measure)
    else:
        if index in tree.children:
            return categorization(datum, tree.children[index], attributes, c, measure)

    candidates = dict()
    for i in tree.children.keys():
        try: candidates[categorization(datum, tree.children[i], attributes, c, measure)]+=1
        except: candidates[categorization(datum, tree.children[i], attributes, c, measure)]=1
    ans = max(candidates, key=candidates.get)
    return ans
    
#terminal 입력 받기
if len(sys.argv) != 4:
    train_file = "dt_train.txt"
    test_file = "dt_test.txt"
    out_file = "dt_result.txt"

else:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    out_file = sys.argv[3]

#train_file 읽기
with open(train_file) as train:
    attributes = train.readline().split()
    df = pd.DataFrame(columns=attributes)
    for line in train.readlines():
        df.loc[len(df)] = line.split()
    label = attributes.pop()

m = 'ratio'

t = node()
t = Decision_Tree(df, attributes.copy(), label, t, m)

f = open(out_file, 'w')

with open(test_file) as test:
    attributes = test.readline().split()
    attributes.append(label)
    print(*attributes, sep='\t', end='\n',file=f)

    for line in test.readlines():
        datum = line.split()
        datum.append(categorization(datum, t, attributes, label, m))
        print(*datum, sep='\t', end='\n', file=f)

f.close()