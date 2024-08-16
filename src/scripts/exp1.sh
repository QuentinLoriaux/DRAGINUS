. ./execute
cd ..

run_and_eval "Llama2-7b-chat" "2WikiMultihopQA" 1.05
run_and_eval "Llama2-7b-chat" "IIRC" 1.05

run_and_eval "Llama2-13b-chat" "2WikiMultihopQA" 1.05
run_and_eval "Llama2-13b-chat" "IIRC" 1.05

run_and_eval "Vicuna-13b-v1.5" "2WikiMultihopQA" 1.05
run_and_eval "Vicuna-13b-v1.5" "IIRC" 1.05

