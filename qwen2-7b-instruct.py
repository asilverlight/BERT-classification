import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
import ujson
def chat_model(queries, model_id): 
    model_dir = snapshot_download(model_id=model_id,
                              cache_dir='/data00/yifei_chen/models')
    # model dir 是模型文件所在的目录

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype='auto',device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

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
        do_sample=True,
        )
    reshaped_outputs = generated_ids.view(-1, len(queries), generated_ids.shape[-1])
    responses = tokenizer.batch_decode(reshaped_outputs[0], skip_special_tokens=True)
    special_str = ""
    if model_id == "qwen/Qwen2-7B-Instruct":
        special_str = "assistant\n"
    elif model_id == "LLM-Research/Meta-Llama-3-8B-Instruct":
        special_str = "assistant\n\n"
    responses = [response[response.rfind(special_str) + len(special_str):] for response in responses
            ]
    return responses

query = [
        "What sport does Marko Vidovi\u0107 play?",
        "Who was the screenwriter for That's Entertainment!?",
        "Who is the father of Rachel Summers?"
    ]
queries = [
        "\"Luis Zwick\"",
        "\"Umaru Bangura\"",
        "\"Joevin Jones\""
    ]
prompts = []
i = 0
with open("datasets/wow/train.jsonl", 'r', encoding='utf-8') as fr:
    for line in fr:
        prompts.append(ujson.loads(line))
        i += 1
        if i >=4:
            break
answer = chat_model(queries=prompts, model_id="qwen/Qwen2-7B-Instruct")

# 输出答案
print(answer)