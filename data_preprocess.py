import ujson
import numpy as np
import os
from collections import Counter, OrderedDict
rng = np.random.default_rng(114514)

datas = []
with open("datasets/wow/train.jsonl", 'r', encoding='utf-8') as fr:
    for line in fr:
        datas.append(ujson.loads(line)["question"])
        
data = list(OrderedDict.fromkeys(datas))

# 输出到新的文件
def write_to_file(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for item in data:
            ujson.dump(item, file)
            file.write('\n')


write_to_file(data, 'datasets/wow/train.jsonl')
# files = ["datasets/wikiasp/dev.jsonl", "datasets/wikiasp/train.jsonl", "datasets/wikiasp/test.jsonl"]
# for file in files:
#     datas[file] = set()
#     with open(file, 'r', encoding='utf-8') as fr:
#         for line in fr:
#             data = ujson.loads(line)
#             datas[file].add(data)
            
# inter01 = datas[files[0]].intersection(files[1])
# inter12 = datas[files[1]].intersection(files[2])
# inter20 = datas[files[2]].intersection(files[0])
# print(inter01)
# print(inter12)
# print(inter20)