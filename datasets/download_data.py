import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'hf_cache'

# os.system('huggingface-cli download --repo-type dataset --resume-download SWE-bench/SWE-bench_Verified --local-dir ./code_domain/SWE-bench_Verified --local-dir-use-symlinks False')  # 用于下载数据集
os.system('huggingface-cli download --token xxx --repo-type dataset --resume-download bigcode/bigcodebench --local-dir ./datasets/code_domain/bigcodebench --local-dir-use-symlinks False')  # 用于下载数据集
