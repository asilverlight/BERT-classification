# python data_preprocess.py
# echo 'data preprocess has finished'

# python generate_qa.py \
# --model_path LLM-Research/Meta-Llama-3-8B-Instruct \
# --data_path datasets/train_popqa.jsonl \
# --save_path datasets/train_popqa_processed_Llama_3_8B_instruct.jsonl \
# --num 4

# echo 'train_popqa_processed_Llama_3_8B_instruct.jsonl has finished'

# python generate_qa.py \
# --model_path qwen/Qwen2-7B-Instruct \
# --data_path datasets/train_popqa.jsonl \
# --save_path datasets/train_popqa_processed_Qwen2_7B_instruct.jsonl \
# --num 4

# echo 'train_popqa_processed_Qwen2_7B_instruct.jsonl has finished'

# python generate_qa.py \
# --model_path LLM-Research/Meta-Llama-3-8B-Instruct \
# --data_path datasets/test_popqa.jsonl \
# --save_path datasets/test_popqa_processed_Llama_3_8B_instruct.jsonl \
# --num 4

# echo 'test_popqa_processed_Llama_3_8B_instruct.jsonl has finished'

# python generate_qa.py \
# --model_path qwen/Qwen2-7B-Instruct \
# --data_path datasets/test_popqa.jsonl \
# --save_path datasets/test_popqa_processed_Qwen2_7B_instruct.jsonl \
# --num 4

# echo 'test_popqa_processed_Qwen2_7B_instruct.jsonl has finished'

# # python generate_qa.py \
# # --model_path TabbyML/Mistral-7B \
# # --data_path datasets/train_popqa.jsonl \
# # --save_path datasets/train_popqa_processed_Mistral-7B.jsonl \
# # --num 4 \
# # --batch_query 10

# # echo 'train_popqa_processed_Mistral-7B.jsonl has finished'

# # python generate_qa.py \
# # --model_path TabbyML/Mistral-7B \
# # --data_path datasets/test_popqa.jsonl \
# # --save_path datasets/test_popqa_processed_Mistral-7B.jsonl \
# # --num 4 \
# # --batch_query 10

# # echo 'test_popqa_processed_Mistral-7B.jsonl has finished'

# python generate_qa.py \
# --model_path AI-ModelScope/Mistral-Nemo-Instruct-2407 \
# --data_path datasets/train_popqa.jsonl \
# --save_path datasets/train_popqa_processed_Mistral-Nemo-12B-instruct.jsonl \
# --num 4 \
# --batch_query 10

# echo 'train_popqa_processed_Mistral-Nemo-12B-instruct.jsonl has finished'

# python generate_qa.py \
# --model_path AI-ModelScope/Mistral-Nemo-Instruct-2407 \
# --data_path datasets/test_popqa.jsonl \
# --save_path datasets/test_popqa_processed_Mistral-Nemo-12B-instruct.jsonl \
# --num 4 \
# --batch_query 10

# echo 'test_popqa_processed_Mistral-Nemo-12B-instruct.jsonl has finished'

# echo 'BERT classification begin'
# python BERT.py

# python generate_qa_one_answer.py \
# --data_path datasets/wow/train.jsonl datasets/wow/dev.jsonl \
# --train_path datasets/wow/train_wow.jsonl \
# --test_path datasets/wow/test_wow.jsonl \
# --duplicate False

# echo 'data process has finished!'
# python qwen_add.py

python BERT.py \
--train_path datasets/wow/train_wow_added.jsonl \
--test_path datasets/wow/test_wow_added.jsonl \
--save_path models/BERT/bert_wow.pth \
--experiment_type "wow dataset with data corruption" \
--use_corruption True

echo 'BERT classification has finished!'