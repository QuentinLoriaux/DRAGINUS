# Llama2-7b-chat
cd ..

python3 main.py -c ../config/Llama2-7b-chat/2WikiMultihopQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_7b_chat_2wikimultihopqa/0 && \


python3 main.py -c ../config/Llama2-7b-chat/HotpotQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_7b_chat_hotpotqa/0 && \


python3 main.py -c ../config/Llama2-7b-chat/IIRC/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_7b_chat_iirc/0 && \

python3 main.py -c ../config/Llama2-7b-chat/StrategyQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_7b_chat_strategyqa/0 && \

# Vicuna-13b-v1.5

python3 main.py -c ../config/Vicuna-13b-v1.5/2WikiMultihopQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/vicuna_13b_v1.5_2wikimultihopqa/0 && \

python3 main.py -c ../config/Vicuna-13b-v1.5/HotpotQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/vicuna_13b_v1.5_hotpotqa/0 && \


python3 main.py -c ../config/Vicuna-13b-v1.5/StrategyQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/vicuna_13b_v1.5_strategyqa/0 && \


python3 main.py -c ../config/Vicuna-13b-v1.5/IIRC/DRAGIN.json && \
python3 evaluate.py --dir ../result/vicuna_13b_v1.5_iirc/0 && \

# Llama2-13b-chat

python3 main.py -c ../config/Llama2-13b-chat/2WikiMultihopQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_13b_chat_2wikimultihopqa/0 && \

python3 main.py -c ../config/Llama2-13b-chat/HotpotQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_13b_chat_hotpotqa/0 && \


python3 main.py -c ../config/Llama2-13b-chat/IIRC/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_13b_chat_iirc/0 && \

python3 main.py -c ../config/Llama2-13b-chat/StrategyQA/DRAGIN.json && \
python3 evaluate.py --dir ../result/llama2_13b_chat_strategyqa/0 && \
