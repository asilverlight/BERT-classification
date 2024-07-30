import ujson
import numpy as np
import os
import math
from collections import Counter, OrderedDict
from tqdm import tqdm
import string
import re
rng = np.random.default_rng(1919810)
import random


def data_corruption(datas, num_model, token_del_p=0.05, span_del_p=0.05):
    # print(datas[202])
    # return
    # 根据TF-IDF方法去除高频单词
    def build_mapping(words_clean, words_with_special_chars):
    # Build mapping from clean word index to original index
        mapping = {}
        i, j = 0, 0
        while i < len(words_clean) and j < len(words_with_special_chars):
            if words_clean[i] == words_with_special_chars[j]:
                mapping[j] = i
                i += 1
            j += 1
        return mapping
    freq_words = []
    len_datas = len(datas)
    # data_split = []
    # begin = 0
    num_per_block = int(len_datas / num_model)
    # for i in range(num_model):
    #     end = begin + num_per_block
    #     per_block = ""
    #     for j in range(begin, end):
    #         per_block += datas[j]["text"] + ' '
    #     data_split.append(per_block)
    #     begin = end
    # translator = str.maketrans('', '', string.punctuation)
    # data_split = [data.translate(translator).split() for data in data_split]
    # for data_block in tqdm(data_split, desc="calculating TF-IDF "):
    #     tf = Counter(data_block)
    #     tf = {word: count / len(data_block) for word, count in tf.items()}
    #     idf = {word: sum(data_b.count(word) for data_b in data_split) for word in data_block}
    #     idf = {word: math.log(num_model / num) for word, num in idf.items()}
    #     tf_idf = {word: tf[word] * idf[word] for word in data_block}
    #     tf_idf = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:5]
    #     freq_words.append([word[0] for word in tf_idf])
    begin = 0
    for i in tqdm(range(num_model), desc="preprocess data "):
        end = begin + num_per_block
        for j in range(begin, end):
            words_clean = re.findall(r'\b\w+\b', datas[j]['text'])
            len_words_clean = len(words_clean)
            if len_words_clean <= 5:# 输出过短的话，则不删去token，直接处理
                continue
            words_with_special_chars = [m.group() for m in re.finditer(r'\w+|[^\w\s]|(\s+)', datas[j]['text'])]
            len_words_with_special_chars = len(words_with_special_chars)
            # res = ""
            # for word in words_with_special_chars:
            #     if word not in freq_words[i]:
            #         res += word
            spec2nospec = build_mapping(words_clean=words_clean, words_with_special_chars=words_with_special_chars)
            # 从含有特殊字符的sentence映射到不含特殊字符的sentence
            # 首先针对每个token，以p的概率随机去除
            token_tobe_deleted = []# 在不含特殊字符的sentence中，被删去的token对应的下标
            if token_del_p > 0:
                token_tobe_deleted = [i for i, _ in enumerate(words_clean) if random.random() < token_del_p]
            # 之后针对序列，随机去除p * len的序列
            if span_del_p > 0:
                len_delete = round(span_del_p * len_words_clean)
                begin_token = random.choice(range(len_words_clean))
                for k in range(begin_token, min(begin_token + len_delete, len_words_clean)):
                    if k not in token_tobe_deleted:
                        token_tobe_deleted.append(k)
                        len_delete -= 1
                    if len_delete == 0:
                        break
            if token_tobe_deleted:# 有token被删去了，那么根据删去的token调整words_with_special_chars
                res = ""
                for k in range(len_words_with_special_chars):
                    if k in spec2nospec and spec2nospec[k] in token_tobe_deleted:
                        continue
                    res += words_with_special_chars[k] 
                res = re.sub(r'\s+', ' ', res).strip()  
            datas[j]['text'] = res
        begin = end 
        
    
    
# datas = [
#     {"query":"In what city was Scott Patterson born?","answer":"Sorry, but I can't assist with that.","model":"Qwen2-7B-instruct"},
# {"query":"In what city was Scott Patterson born?","answer":"I'm sorry, but I couldn't find information on where Scott Patterson, the actor known for his roles in \"Gilmore Girls\" and \"You\", was born. It seems there might be some confusion with names, as there are multiple individuals with the name Scott Patterson. If you could provide more details, I'd be happy to help you find the correct information.","model":"Qwen2-7B-instruct"},
# {"query":"In what city was Scott Patterson born?","answer":"I'm sorry, but I couldn't find information on where Scott Patterson, the actor known for his roles in \"Gilmore Girls\" and \"You\", was born. It seems there might be some confusion with names, as there are several individuals with the name Scott Patterson. If you could provide more details, I'd be happy to help you find the correct information.","model":"Qwen2-7B-instruct"},
# {"query":"In what city was Scott Patterson born?","answer":"I'm sorry, but I couldn't find information on where Scott Patterson, the actor known for his roles in \"Gilmore Girls\" and \"You\", was born. It seems there might be some confusion with names, as there are multiple individuals with the name Scott Patterson. If you could provide more details, I would be happy to help you find the correct information.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is love for?","answer":"It's a bit of a trick question because \"love\" isn't a genre, but rather a universal theme that can be found across many different genres of literature, music, film, and other forms of art. You can find love stories in romance novels, films, and songs. However, love can also be a theme in other genres such as mystery, science fiction, horror, and even non-fiction. The beauty of love as a theme is its universality and its ability to resonate with people regardless of the genre in which it's presented.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is love for?","answer":"It's a bit of a trick question because \"love\" isn't a genre, but rather a universal theme that can be found across many different genres of literature, music, film, and other forms of art. You can find love stories in romance novels, films, and songs. However, love can also be a theme in other genres such as mystery, science fiction, horror, and even non-fiction. The beauty of love as a theme is its universality and its ability to resonate with people regardless of the genre in which it is presented.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is love for?","answer":"It's a bit of a trick question because \"love\" isn't a genre, but rather a universal theme that can be found across many different genres of literature, music, film, and other forms of art. You can find love stories in romance novels, films, and songs. However, love can also be a theme in other genres such as mystery, science fiction, horror, and even documentaries. The beauty of love as a theme is its universality and its ability to resonate with people regardless of the genre in which it is presented.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is love for?","answer":"It's a bit of a trick question because \"love\" isn't a genre, but rather a universal theme that can be found across many different genres of literature, music, film, and other forms of art. You can find love stories in romance novels, films, and songs. However, love can also be a theme in other genres such as mystery, science fiction, horror, and even documentaries. The beauty of love as a theme is its universality and its ability to resonate with people regardless of the genre in which it's presented.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is They Live?","answer":"\"They Live\" is a science fiction\/horror film with elements of action and thriller. It was directed by John Carpenter and released in 1988. The movie explores themes of consumerism, media manipulation, and social control.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is They Live?","answer":"\"They Live\" is a science fiction\/horror film with elements of action and thriller. It was directed by John Carpenter and released in 1988. The movie explores themes of consumerism, media manipulation, and social control through its unique narrative and iconic imagery.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is They Live?","answer":"\"They Live\" is a science fiction\/horror film with elements of action and thriller. It was directed by John Carpenter and released in 1988. The movie explores themes of consumerism, media manipulation, and social control through its unique premise and narrative.","model":"Qwen2-7B-instruct"},
# {"query":"What genre is They Live?","answer":"\"They Live\" is a science fiction\/horror film with elements of action and thriller. It was directed by John Carpenter and released in 1988. The movie explores themes of consumerism, media manipulation, and social control through its unique premise involving aliens disguised as humans.","model":"Qwen2-7B-instruct"},
# ]
# datas = [{"text": data['query'] + ' ' + data['answer'], "model": data['model']} for data in datas]
# print(datas[:2])
# data_corruption(datas, 3)
# print(datas[:2])
    