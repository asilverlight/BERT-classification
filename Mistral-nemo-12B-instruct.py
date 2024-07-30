import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from modelscope import snapshot_download
# from mistral_inference.transformer import Transformer
# from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer
import os
import ujson
from multiprocessing import Pool
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def chat_model(model_id, prompts):
    model_dir = snapshot_download(model_id=model_id,
                              cache_dir='/data00/yifei_chen/models')
    # model dir 是模型文件所在的目录

    
    model = Transformer.from_folder(model_dir, max_batch_size=10)
    # tokenizer = MistralTokenizer.from_file("models/AI-ModelScope/Mistral-Nemo-Instruct-2407/tekken.json")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
    tokenizer.pad_token = tokenizer.eos_token
    # prompt = "Who was the screenwriter for That's Entertainment!?"
    # tokens = [[tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=prompt)])).tokens] for prompt in prompts]
    # completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    # tokens = tokenizer.encode_chat_completion(completion_request).tokens
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to('cuda')
    # print(tokens)
    model.to(device)
    # print(tokens)
    # def process_tokens(token_list):
    #     return generate(token_list, model, max_tokens=512, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    # out_tokens = generate(tokens, model, max_tokens=512, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    # out_tokens = [generate(token, model, max_tokens=256, temperature=1, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id) for token in tokens]
    # temperatures = [0.1, 0.3, 0.5, 1, 1.2, 1.5, 1.8, 2]
    # top_ps = [0.4, 0.6, 0.8, 0.9, 0.99]
    # for temperature in temperatures:
    #     for top_p in top_ps:
    out_tokens = generate(tokens["input_ids"].tolist(), model, max_tokens=128, temperature=0.35, top_p=0.8)
    result = [tokenizer.decode(out_token) for out_token in out_tokens]
    for res in result:
        print([res])
        print('\n')
    # print(out_tokens)
    # split_sentences = []
    # current_sentence = []
    # # print(out_tokens)
    # for token in out_tokens[0]:
    #     if token != 2:
    #         current_sentence.append(token)
    #     else:
    #         split_sentences.append(current_sentence)
    #         current_sentence = []
    # split_sentences.append(current_sentence)
    # print(split_sentences)
prompts = []
i = 0
with open("datasets/wow/train.jsonl", 'r', encoding='utf-8') as fr:
    for line in fr:
        prompts.append(ujson.loads(line))
        i += 1
        if i >=4:
            break
chat_model("AI-ModelScope/Mistral-Nemo-Instruct-2407", prompts)