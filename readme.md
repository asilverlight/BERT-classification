# 用BERT做分类任务
data_preprocess.py是将datasets中的query单独提取出来保存的代码

generate_qa_one_answer.py是针对数据集做inference得到answer的代码

默认使用同样的10000个query做train，1000个query做test

其中，qwen模型生成的answer可能会有格式错误等问题，在qwen_add.py中做了修正

运行分类代码：bash generate_and_BERT.sh

llama-3-8b-instruct.py、mistral……、qwen2-7b……是每个模型单独做inference的代码
