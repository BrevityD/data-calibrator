import os
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'hf_cache'
os.environ['HF_TOKEN'] = 

datasets = {
    "allenai/tulu-3-sft-personas-code": "./datasets/code_domain/tulu-3-sft-personas-code",
    "allenai/tulu-3-sft-personas-algebra": "./datasets/math_domain/tulu-3-sft-personas-algebra",
    "allenai/tulu-3-sft-personas-instruction-following": "./datasets/general_domain/tulu-3-sft-personas-instruction-following",
    "allenai/tulu-3-sft-personas-math-grade-filtered": "./datasets/math_domain/tulu-3-sft-personas-math-grade-filtered",
    "allenai/tulu-3-sft-personas-math-filtered": "./datasets/math_domain/tulu-3-sft-personas-math-filtered",
}


for dataset, localdir in datasets.items():
    os.system(f'huggingface-cli download --repo-type dataset --resume-download \
              {dataset}  \
            --local-dir {localdir} \
            --local-dir-use-symlinks False')  # 用于下载数据集
    time.sleep(5)
# os.system('huggingface-cli download --resume-download meta-llama/Llama-3.2-3B --local-dir /public/home/dzj/models/Llama-3.2-3B --local-dir-use-symlinks False')
