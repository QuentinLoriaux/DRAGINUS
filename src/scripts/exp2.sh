. ./execute
cd ..

run_and_eval "Llama2-7b-chat" "HotpotQA" 1.05
run_and_eval "Llama2-7b-chat" "StrategyQA" 1.05

run_and_eval "Llama2-13b-chat" "HotpotQA" 1.05
run_and_eval "Llama2-13b-chat" "StrategyQA" 1.05

run_and_eval "Vicuna-13b-v1.5" "HotpotQA" 1.05
run_and_eval "Vicuna-13b-v1.5" "StrategyQA" 1.05
