evalscope eval \
 --model Qwen3-8B \
 --api-url http://127.0.0.1:13133/v1 \
 --api-key EMPTY \
 --datasets bigcodebench \
 --eval-type openai_api \
 --generation-config '{"max_tokens": 16384}' \
 --limit 3 \
 --repeats 2 \
 --generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
# --eval-batch-size 8 \
#  --datasets gsm8k mmlu ceval gpqa_diamond  # \
# --datasets internal_mof_information_extraction \