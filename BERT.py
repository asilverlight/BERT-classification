import os
import torch
import ujson
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, AdamW, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import datetime, time
import random
import numpy as np
import argparse
from modelscope import snapshot_download
from process_before_train import data_corruption
rng = np.random.default_rng(114514)

# 在导入data时，从datasets文件夹里导入，最终的用于训练的data为train_popqa_processed.jsonl，用于测试的data为test_popqa_processed.jsonl
# 其中，每条数据的格式都是：
# {"query":"Who was the producer of Rehab?","answer":"The producer of the HBO series \"The Newsroom\" is Aaron Sorkin.","model":"Llama-3-8B-instruct"}

# 读取数据
def read_data(file_path, use_query=False):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        if use_query:
            for line in fr:
                data = ujson.loads(line)
                temp = dict()
                temp['text'] = data['query'] + ' ' + data['answer']
                temp['model'] = data['model']
                datas.append(temp)
        else:
            for line in fr:
                data = ujson.loads(line)
                temp = dict()
                temp['text'] = data['answer']
                temp['model'] = data['model']
                datas.append(temp)
    return datas

# 定义一个数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'Llama-3-8B-instruct': 0, 'Qwen2-7B-instruct': 1, 'Mistral-nemo-12B-instruct': 2}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["model"]
        label = self.label_map[label]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label)}
    
# 训练
def train(model, tokenizer, train_data, save_path, num_epochs=5, val_rate=0.1, experiment_type="wow"):
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_labels)))
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda')
    model.to(device)
    
    # 划分训练集和验证集
    train_size = (1 - val_rate) * len(train_data)
    val_size = val_rate * len(train_data)
    train_data, val_data = random_split(train_data, [int(train_size), int(val_size)])
    
    # 创建数据集以及加载器、优化器
    train_dataset = TextClassificationDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = TextClassificationDataset(val_data, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 训练
    log_file = open('training_log.log', 'a')
    model.train()
    for epoch in range(num_epochs):
        losses = []
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} / {num_epochs}"):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                outputs = model(**batch)
                _, predicted = torch.max(outputs.logits, dim=1)
                labels = batch['labels']
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        avg_loss = sum(losses) / len(losses)
                
        # 写入日志
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{current_time}] Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}, Experiment type: {experiment_type}\n"
        log_file.write(log_entry)
        print(log_entry.strip())
    log_file.write('\n')
    torch.save(model.state_dict(), save_path)
    
def eval(model, tokenizer, test_data, use_query=False):
    device = torch.device('cuda')
    model.to(device)
    test_dataset = TextClassificationDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    correct = 0
    total = 0
    log_file = open('training_log.log', 'a')
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            # print(outputs)
            _, predicted = torch.max(outputs.logits, dim=1)
            labels = batch['labels']
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy}")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{current_time}] Test Accuracy: {test_accuracy}\n")
    log_file.write('\n')
    
def deminish_data(train_data, num=4, size=1000):
    grouped_data = {}
    # temp = []
    # num = 0
    for item in train_data:
        query = item['query']
        if query not in grouped_data:
            # if len(temp) != 4:
            #     num += 1
            grouped_data[query] = []
        grouped_data[query].append(item)
        # temp = grouped_data[query]
    # print(num)
    all_queries = list(grouped_data.keys())
    selected_queries = rng.choice(all_queries, size=1000, replace=False)
    new_list = []
    for query in selected_queries:
        if len(grouped_data[query]) > num:
            queries = rng.choice(grouped_data[query], size=num, replace=False)
        else:
            queries = grouped_data[query]
        new_list.extend(queries)
    # new_list = [item for query in selected_queries for item in grouped_data[query]]
    return new_list
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="datasets/train_popqa_processed_one_answer.jsonl", help="train data path")
    parser.add_argument("--test_path", type=str, default="datasets/test_popqa_processed_one_answer.jsonl", help="test data path")
    parser.add_argument("--save_path", type=str, default="models/BERT/bert_wow.pth", help="model save path")
    parser.add_argument("--use_query", type=bool, default=False, help="whether to use query to make data")
    parser.add_argument("--use_corruption", type=bool, default=True, help="whether to use data corruption")
    parser.add_argument("--experiment_type", type=str, default="wow dataset", help="the detail of this experiment")
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    train_data = read_data(train_path, use_query=args.use_query)
    test_data = read_data(test_path, use_query=args.use_query)
    
    rng.shuffle(train_data)
    rng.shuffle(test_data)
    if args.use_corruption:
        print("data corrupting...")
        data_corruption(train_data, 3)
        data_corruption(test_data, 3)
    
    # 加入数据腐蚀，增强鲁棒性
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train(model, tokenizer, train_data, args.save_path, experiment_type=args.experiment_type)

    # 以下是仅测试的代码
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model.load_state_dict(torch.load('models/BERT/bert_one_answer.pth'))
    
    eval(model, tokenizer, test_data, use_query=args.use_query)

if __name__ == "__main__":
    main()
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')