import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
import ujson
from tqdm import tqdm

model_id = "qwen/Qwen2-7B-Instruct"
model_dir = snapshot_download(model_id=model_id,
                              cache_dir='/data00/yifei_chen/models')
# model dir 是模型文件所在的目录

tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype='auto',device_map="auto")


null_answer_train = []
with open('datasets/wow/train_wow.jsonl', 'r', encoding='utf-8') as fr:
    for line in fr:
        data = ujson.loads(line)
        if len(data['answer']) <= 5:
            null_answer_train.append(data['query'])

null_answer_test = []
with open('datasets/wow/test_wow.jsonl', 'r', encoding='utf-8') as fr:
    for line in fr:
        data = ujson.loads(line)
        if len(data['answer']) <= 5:
            null_answer_test.append(data['query'])
with open('datasets/wow/add_query_train.jsonl', 'w', encoding='utf-8') as fw:
    for data in null_answer_train:
        ujson.dump(data, fw)
        fw.write('\n')
with open('datasets/wow/add_query_test.jsonl', 'w', encoding='utf-8') as fw:
    for data in null_answer_test:
        ujson.dump(data, fw)
        fw.write('\n')
def chat_model(prompts): 

    messages_list = [{'role':'system','content':'You are a helpful assistant system'},
    {'role': 'user','content': prompts}]
    # 使用分词器的 apply_chat_template 方法将上面定义的消,息列表转护# tokenize=False 表示此时不进行令牌化
    texts = tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True
            )

    #将处理后的文本令牌化并转换为模型输入张量，然后将这些张量移至之前
    model_inputs = tokenizer(texts, return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs['input_ids'],
        max_new_tokens=512,
        do_sample=True,
        temperature=1.5,
        )
    # reshaped_outputs = generated_ids.view(-1, len(prompts), generated_ids.shape[-1])
    responses = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    special_str = "assistant\n"
    responses = responses[responses.rfind(special_str) + len(special_str):]
    return responses

prompts = []
with open("datasets/wow/add_query_train.jsonl", 'r', encoding='utf-8') as fr:
    for line in fr:
        prompts.append(ujson.loads(line))

with open("datasets/wow/add_qa_train.jsonl", 'w', encoding='utf-8') as fw:
    for i in tqdm(range(len(prompts))):
        query = prompts[i]
        response = '\n'
        while len(response) <= 5:
            response = chat_model(query)
        query_dict = dict()
        query_dict["query"] = query
        query_dict["answer"] = response
        query_dict["model"] = "Qwen2-7B-instruct"
        ujson.dump(query_dict, fw)
        fw.write('\n')
        
prompts = []
with open("datasets/wow/add_query_test.jsonl", 'r', encoding='utf-8') as fr:
    for line in fr:
        prompts.append(ujson.loads(line))

with open("datasets/wow/add_qa_test.jsonl", 'w', encoding='utf-8') as fw:
    for i in tqdm(range(len(prompts))):
        query = prompts[i]
        response = '\n'
        while len(response) <= 5:
            response = chat_model(query)
        query_dict = dict()
        query_dict["query"] = query
        query_dict["answer"] = response
        query_dict["model"] = "Qwen2-7B-instruct"
        ujson.dump(query_dict, fw)
        fw.write('\n')

def add_qwen(initial_train, initial_test, add_train, add_test):
    initial_train_data = []
    with open(initial_train, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = ujson.loads(line)
            initial_train_data.append(data)
    initial_test_data = []
    with open(initial_test, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = ujson.loads(line)
            initial_test_data.append(data)
            
            
    add_train_data = []
    with open(add_train, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = ujson.loads(line)
            add_train_data.append(data)
    j = 0
    for i in range(len(null_answer_train)):
        while(initial_train_data[j]['query'] != null_answer_train[i]):
            j += 1
        initial_train_data[j] = add_train_data[i]
        j += 1
    add_test_data = []
    with open(add_test, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = ujson.loads(line)
            add_test_data.append(data)
    j = 0
    for i in range(len(null_answer_test)):
        while(initial_test_data[j]['query'] != null_answer_test[i]):
            j += 1
        initial_test_data[j] = add_test_data[i]
        j += 1
    with open("datasets/wow/train_wow_added.jsonl", 'w', encoding='utf-8') as fw:
        for data in tqdm(initial_train_data, desc='train '):
            ujson.dump(data, fw)
            fw.write('\n')
    with open("datasets/wow/test_wow_added.jsonl", 'w', encoding='utf-8') as fw:
        for data in tqdm(initial_test_data, desc='test '):
            ujson.dump(data, fw)
            fw.write('\n')
    
    num = 0
    for data in initial_train_data:
        if len(data['answer']) <= 5:
            num += 1
    print(num)
    num = 0
    for data in initial_test_data:
        if len(data['answer']) <= 5:
            num += 1
    print(num)

add_qwen('datasets/wow/train_wow.jsonl', 'datasets/wow/test_wow.jsonl', 'datasets/wow/add_qa_train.jsonl', 'datasets/wow/add_qa_test.jsonl')
