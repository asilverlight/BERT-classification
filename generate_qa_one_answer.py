import ujson
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
from tqdm import tqdm
import argparse
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer
import warnings
import numpy as np
import random
from collections import Counter, OrderedDict
rng = np.random.default_rng(114514)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def get_query(file_name, train_num=2500, test_num=250, duplicate=True):
    # 这次直接从popqa_test里抽取
    # 每个model不重复地分别抽2500条作为训练集，再抽250条作为测试集
    if isinstance(file_name, list):# 将train和test的地址分别传入
        assert len(file_name) == 2, "the length of input_path can't be more than 2"
        train_path, test_path = file_name[0], file_name[1]
        train_datas = []
        test_datas = []
        with open(train_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                train_datas.append(ujson.loads(line))
        with open(test_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                test_datas.append(ujson.loads(line))  
        
        train_datas = list(OrderedDict.fromkeys(train_datas))
        test_datas = list(OrderedDict.fromkeys(test_datas))  
        train_datas = rng.permutation(train_datas)
        test_datas = rng.permutation(test_datas)
        if duplicate:
            train_datas = train_datas[:train_num]
            test_datas = test_datas[:test_num]   
        else:
            train_datas = [train_datas[i:i + train_num] for i in range(0, 3 * train_num, train_num)]  
            test_datas = [test_datas[i:i + test_num] for i in range(0, 3 * test_num, test_num)]
    elif isinstance(file_name, str):# 传入整个数据集，自行区分出train和test
        datas = []
        with open(file_name, 'r', encoding='utf-8') as fr:
            for line in fr:
                datas.append(ujson.loads(line)["question"])
                
        # 去重
        datas = list(OrderedDict.fromkeys(datas))
                
        sample_data = rng.permutation(datas)
        if not duplicate:
            sample_data = sample_data[:train_num * 3 + test_num * 3]
            train_datas = [sample_data[i:i + train_num] for i in range(0, 3 * train_num, train_num)]
            test_datas = [sample_data[i:i + test_num] for i in range(3 * train_num, train_num * 3 + test_num * 3, test_num)]
        else:
            train_datas = sample_data[:train_num]
            test_datas = sample_data[train_num:train_num + test_num]
    return train_datas, test_datas
    
    

def chat_model(model_id):
    model_dir = snapshot_download(model_id=model_id,
                              cache_dir='/data00/yifei_chen/models')
    # model dir 是模型文件所在的目录

    tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
    tokenizer.pad_token = tokenizer.eos_token
    if model_id == "AI-ModelScope/Mistral-Nemo-Instruct-2407":
        model = Transformer.from_folder(model_dir, max_batch_size=25)
        return model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype='auto',device_map="auto")
        return model, tokenizer

def generate_response(model, tokenizer, model_id, queries):
    """
    一次传递5个query, model处理一个batch
    """
    device = "cuda"
    # model.to(device)
    if model_id == "AI-ModelScope/Mistral-Nemo-Instruct-2407":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # tokens = [[tokenizer.
        #            encode_chat_completion(ChatCompletionRequest(messages=
        #         [UserMessage(content=prompt)])).tokens] for prompt in queries]
        # out_tokens = [generate(token, model, max_tokens=128, 
        #                         temperature=1, 
        #                         eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id) 
        #               for token in tokens]
        # responses = [tokenizer.decode(out_token[0]) for out_token in out_tokens]
        # return responses
        queries = queries.tolist()
        tokens = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to('cuda')
        responses = generate(tokens["input_ids"].tolist(), model, max_tokens=256, temperature=0.35, top_p=0.9)
        responses = [tokenizer.decode(response) for response in responses]
    else:
        messages_list = [
            [{'role':'system','content':'You are a helpful assistant system'},
        {'role': 'user','content': prompt}] for prompt in queries
            ]
        # 使用分词器的 apply_chat_template 方法将上面定义的消,息列表转护# tokenize=False 表示此时不进行令牌化
        texts = [
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for messages in messages_list
                ]

        #将处理后的文本令牌化并转换为模型输入张量，然后将这些张量移至之前
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
            )
        
        reshaped_outputs = generated_ids.view(-1, len(queries), generated_ids.shape[-1])
        responses = tokenizer.batch_decode(reshaped_outputs[0], skip_special_tokens=True)
        special_str = ""
        if model_id == "qwen/Qwen2-7B-Instruct":
            special_str = "assistant\n"
        elif model_id == "LLM-Research/Meta-Llama-3-8B-Instruct":
            special_str = "assistant\n\n"
        responses = [response[response.rfind(special_str) + len(special_str):] for response in responses]
    return responses

def generate_q_a_label(input_path, output_path_train, output_path_test, batch_size, duplicate, train_num, test_num):
    train_queries, test_queries = get_query(file_name=input_path, duplicate=duplicate, train_num=train_num, test_num=test_num)
    model_types = ["Qwen2-7B-instruct", "Llama-3-8B-instruct", "Mistral-nemo-12B-instruct"]
    model_ids = ["qwen/Qwen2-7B-Instruct", "LLM-Research/Meta-Llama-3-8B-Instruct", "AI-ModelScope/Mistral-Nemo-Instruct-2407"]
    
    # 首先处理train query
    with open(output_path_train, 'a', encoding='utf-8') as fw:
        # for k in range(len(model_types)):
        k = 2
        model_type = model_types[k]
        model_id = model_ids[k]
        model, tokenizer = chat_model(model_id)
        if duplicate:
            train_query = train_queries
        else:
            train_query = train_queries[k]
        for i in tqdm(range(0, len(train_query), batch_size), desc="train " + model_type):
            queries = train_query[i: min(i + batch_size, len(train_query))]
            responses = generate_response(model, tokenizer, model_id, queries)
            for query, response in zip(queries, responses):
                query_dict = dict()
                query_dict["query"] = query
                query_dict["answer"] = response
                query_dict["model"] = model_type
                ujson.dump(query_dict, fw)
                fw.write('\n')
        print(f"{model_type} has finished and was saved in {output_path_train}")
                    
    # 接下来处理test query
    with open(output_path_test, 'w', encoding='utf-8') as fw:
        for k in range(len(model_types)):
            model_type = model_types[k]
            model_id = model_ids[k]
            model, tokenizer = chat_model(model_id)
            if duplicate:
                test_query = test_queries
            else:
                test_query = test_queries[k]
            for i in tqdm(range(0, len(test_query), batch_size), desc="test " + model_type):
                queries = test_query[i: min(i + batch_size, len(test_query))]
                responses = generate_response(model, tokenizer, model_id, queries)
                for query, response in zip(queries, responses):
                    query_dict = dict()
                    query_dict["query"] = query
                    query_dict["answer"] = response
                    query_dict["model"] = model_type
                    ujson.dump(query_dict, fw)
                    fw.write('\n')
            print(f"{model_type} has finished and was saved in {output_path_test}")
            
            

def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, nargs='+', default="datasets/popqa_test.jsonl", help="input data set path")
    parser.add_argument("--train_path", type=str, default="datasets/train_popqa_processed_one_answer.jsonl", help="train data save path")
    parser.add_argument("--test_path", type=str, default="datasets/test_popqa_processed_one_answer.jsonl", help="test data save path")
    parser.add_argument("--train_num", type=int, default=10000, help="train query num")
    parser.add_argument("--test_num", type=int, default=1000, help="test query num")
    parser.add_argument("--batch_query", type=int, default=20, help="the number of queries per batch")
    parser.add_argument("--duplicate", type=bool, default=True, help="whether to use the same query for all models")
    # 造数据时，是每个model各采10000个query还是采好10000个query供3个model共用
    args = parser.parse_args()
    generate_q_a_label(args.data_path, args.train_path, args.test_path, args.batch_query, args.duplicate, args.train_num, args.test_num)
    
if __name__ == "__main__":
    main()
