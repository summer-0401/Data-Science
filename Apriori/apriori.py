import sys
from collections import Counter
from itertools import chain, combinations
#mlxtend, cycic?? 안됨. collections, itertools OK

#subset 모두 찾아 하나의 set으로 반환
def Real_subset(_set):
    _real_subset = set(chain.from_iterable(combinations(_set, r) for r in range(1,len(_set))))
    return _real_subset

#self joint(Lk에서 공통 (k-1)-item+다른원소 하나씩. ex: {abc, abd}--->{ab}+{c}+{d}={abcd}
def Self_joint(_L,_length):
    _joined_set=set()
    for x in _L:
        for y in _L:
            _joined_set.add(frozenset(set(x).union(set(y))))
    _joined_set = set(filter(lambda x : len(x)==_length, _joined_set))
    return _joined_set

#self joining으로 만든 itemset의 sub pattern 중 하나라도 k-itemset에 없으면 목록에서 제거
def Pruning(_joined_set, k_itemset, _length):
    _pruned_set = _joined_set.copy()
    for j in _joined_set:
        j_group = set(filter(lambda x : len(x)==_length, Real_subset(j)))
        for j_sub in j_group:
            if not frozenset(j_sub) in k_itemset:
                _pruned_set.discard(j)
    return _pruned_set

#수도코드 그대로
def Apriori(_L, _prev_L, i):
    _total_L = set()
    while len(_L)!=0:
        joined_set = Self_joint(_L, i)
        candidates = Pruning(joined_set, _prev_L, i-1)
        tmp_L = dict.fromkeys(candidates, 0)#없애도되는지확인
        for transaction in db:
            for c in candidates:
                if(transaction.issuperset(c)):
                    tmp_L[c]+=1
        _L = set(dict(filter(lambda x:x[1]>=min_quant, tmp_L.items())).keys())
        _total_L = _total_L.union(_L)
        _prev_L = _L
        i+=1
    return _total_L

#[head 있는 transaction 개수, head&tail 있는 transaction 개수] return
def Count_XY(_h, _t, _db):
    _counted = [0,0]
    for transaction in _db:
            if(transaction.issuperset(_h)):
                _counted[0]+=1
                if(transaction.issuperset(_t)):
                    _counted[1]+=1
    return _counted

#----------------------------
#terminal 입력받기
if len(sys.argv) != 4:
    min_sup = 5/100
    in_file = "input.txt"
    out_file = "output.txt"
else:
    min_sup = int(sys.argv[1])/100
    if min_sup<=0:
        sys.exit("ERROR! Minimum support must be bigger than 0. Terminate the program automatically.")

    in_file = sys.argv[2]
    out_file = sys.argv[3]

db = []
total = 0
tmp_candidates = Counter()

#파일 읽기
with open(in_file) as _input:
    for line in _input.readlines():
        tmp =set(map(int,line.strip().split('\t')))
        db.append(tmp)
        tmp_candidates += Counter(tmp)
    total = len(db)

min_quant = float(total*min_sup)
L = set(dict(filter(lambda x:x[1]>=min_quant, tmp_candidates.items())).keys())

prev_L = set()
for l in L:
    prev_L.add(frozenset({l}))
L=prev_L.copy()

i=2
total_L = Apriori(L, prev_L, i)

f = open(out_file, "w")

save = dict()
for l in total_L:
    #save {(head, tail): [# of heads, # of tails among transactions which have heads]}
    h_group =Real_subset(l)
    for head in h_group:
        tail = tuple(set(l)-set(head))
        save[head, tail]=Count_XY(head, tail, db)
        _support = format(100*save[head, tail][1]/total,".2f")
        _confidence = format(100*save[head,tail][1]/save[head,tail][0],".2f")
        s = '{0}\t{1}\t{sup}\t{conf}\n'.format(set(head), set(tail), sup=_support,conf=_confidence)
        f.write(s)
        #print(set(head),'\t',set(tail),'\t',format(float(save[head, tail][1]/total),".2f"),'\t',format(float(save[head,tail][1]/save[head,tail][0]),".2f"),'\n')

#save = sorted(save.items())

#for s in save:
#    s_head = set(s[0][0])
#    s_tail = set(s[0][1])
#    _support = format(100*s[1][1]/total,".2f")
#    _confidence = format(100*s[1][1]/s[1][0],".2f")
#    s = '{0}\t{1}\t{sup}\t{conf}\n'.format(s_head, s_tail, sup=_support,conf=_confidence)
#    f.write(s)

f.close()