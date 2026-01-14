from pathlib import Path
import os
import json
import time
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger
from tqdm import tqdm
from openai import AsyncOpenAI

import re
import ast
import subprocess
import tempfile
import sys
from datacalibrator.seed import set_seed, SEED
from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.datasets.logic_adaptor import get_logic_dataset
from datacalibrator.datasets.math_adaptor import get_math_dataset, find_box

def extract_code(text: Optional[str], entry_point: str = "task_func") -> str:
    """
    Extract python code from markdown blocks.
    Prioritizes blocks that contain the entry_point definition.
    Strips out <think>...</think> sections to avoid parsing code inside thought process.
    """
    if not text:
        return ""
        
    # Remove <think>...</think> content to focus on the final answer
    # Using re.DOTALL to match newlines
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        
    # Extract all code blocks: ```python ... ``` or ``` ... ```
    # Using finditer to get all matches
    code_blocks = []
    
    # Pattern for labeled python blocks
    pattern_py = r"```python\s*([\s\S]*?)\s*```"
    for match in re.finditer(pattern_py, text):
        code_blocks.append(match.group(1))
        
    # Pattern for generic blocks
    if not code_blocks:
        pattern_generic = r"```\s*([\s\S]*?)\s*```"
        for match in re.finditer(pattern_generic, text):
            code_blocks.append(match.group(1))
            
    if not code_blocks:
        # If no markdown blocks, maybe the whole text (after stripping think) is code?
        # But usually raw text is risky. Let's return the cleaned text and hope.
        return text

    # Iterate blocks to find one defining the entry point
    for block in code_blocks:
        if f"def {entry_point}" in block:
            return block
            
    # If no block contains the specific definition, fallback to the last block 
    return code_blocks[-1]

def check_syntax_and_entry_point(code: str, entry_point: str) -> bool:
    """Check if code is valid python and defines the entry point."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
        
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return True
    return False

def evaluate_code_correctness(prediction: str, example: Dict[str, Any], timeout: int = 20) -> Tuple[bool, bool]:
    """
    Evaluate code by extracting, checking syntax/entry point, and running tests.
    Returns (is_correct, is_format_correct)
    """
    entry_point = example.get("entry_point", "task_func")
    code = extract_code(prediction, entry_point=entry_point)
    
    is_format_correct = check_syntax_and_entry_point(code, entry_point)
    
    if not is_format_correct:
        return False, False

    # Prepare execution
    test_code = example.get("test", "")
    
    # Common imports to reduce ModuleNotFoundError for standard libs
    # Note: Third-party libs (pandas, requests) still need to be installed in the env.
    header = "import unittest\nimport sys\nimport os\nimport math\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport bisect\nimport random\nimport copy\nimport types\nimport operator\nimport re\nimport string\nimport json\n"
    
    # Construct script
    # We put the generated code BEFORE the test class
    # We force exit=False in unittest.main to prevent it from closing the process early if we wanted to do more,
    # but here we rely on the return code. 
    # IMPORTANT: We explicitly silence stdout/stderr to keep logs clean unless debugging
    script = f"{header}\n{code}\n\n{test_code}\n\nif __name__ == '__main__':\n    # Run tests and exit with appropriate code\n    # argv parameter avoids parsing actual command line args of the script\n    unittest.main(argv=['first-arg-is-ignored'], exit=True, verbosity=0)"
    
    try:
        # Create a temporary directory for execution to avoid polluting the project root
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the script to a file inside the temp directory
            script_path = os.path.join(temp_dir, 'eval_script.py')
            with open(script_path, 'w') as f:
                f.write(script)
            
            # Execute in the temporary directory
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=temp_dir,  # Critical: Run inside the temp dir
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return True, True
            else:
                # Optional: Log stderr for debugging if needed
                # print(f"DEBUG: Execution failed for {entry_point}:\n{result.stderr}")
                return False, True
                
    except subprocess.TimeoutExpired:
        return False, True
    except Exception:
        return False, True

def setup_logger(output_dir: Path):
    """Configure loguru logger to save to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "eval.log"
    # Remove default handler and add new ones
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(log_file, rotation="10 MB", level="DEBUG")
    return logger

def get_dataset(domain: str, size: int = 1000):
    if domain == "code":
        return get_code_dataset(size)
    elif domain == "logic":
        return get_logic_dataset(size)
    elif domain == "math":
        return get_math_dataset(size)
    else:
        raise ValueError(f"Unknown domain: {domain}")

def evaluate_sample(domain: str, prediction: str, example: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Evaluate a single sample based on the domain.
    Returns (is_correct, is_format_correct)
    """
    if domain == "math":
        if not prediction:
            return False, False

        # Format: Check if \boxed{} exists
        is_format_correct = "\\boxed{".__contains__(prediction)
        
        # Correctness: Extract and compare
        ground_truth = example.get("ground_truth", "")
        extracted_answer = find_box(prediction)
        
        # Simple string matching normalization
        if not extracted_answer:
            is_correct = False
        else:
            is_correct = extracted_answer.strip() == ground_truth.strip()
            
        return is_correct, is_format_correct
        
    elif domain == "logic":
        if not prediction:
            return False, False

        # Format: Check if <step> tags are used as requested
        is_format_correct = "<step>" in prediction and "</step>" in prediction
        
        # Correctness: Check if goal is reached
        goal = example.get("goal", "")
        if not goal:
            return None, is_format_correct
        
        # Normalize strings for comparison
        # Remove whitespace and punctuation for looser matching
        def normalize(s):
            return "".join(s.split()).lower().strip(".")
            
        pred_norm = normalize(prediction)
        goal_norm = normalize(goal)
        
        # Check if the goal appears in the prediction
        is_correct = goal_norm in pred_norm
        
        return is_correct, is_format_correct
        
    elif domain == "code":
        return evaluate_code_correctness(prediction, example)
        
    return None, None

async def generate_responses(
    server_url: str,
    domain: str,
    split: str,
    output_file: Path,
    model_name: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    concurrency: int = 100
) -> List[Dict[str, Any]]:
    # 1. Load Dataset
    logger.info(f"Loading dataset for generation...")
    train_ds, test_ds = get_dataset(domain)
    
    if split == "train":
        dataset = train_ds
        # Tag for consistency
        dataset = dataset.map(lambda x: {"dataset_split": "train"})
    elif split == "test":
        dataset = test_ds
        # Tag for consistency
        dataset = dataset.map(lambda x: {"dataset_split": "test"})
    elif split == "all":
        from datasets import concatenate_datasets
        # Tag datasets so we can distinguish them later
        train_ds = train_ds.map(lambda x: {"dataset_split": "train"})
        test_ds = test_ds.map(lambda x: {"dataset_split": "test"})
        dataset = concatenate_datasets([train_ds, test_ds])
    else:
        raise ValueError(f"Unknown split: {split}")
        
    logger.info(f"Dataset loaded. Size: {len(dataset)}")

    # Check for existing progress (Resume)
    existing_ids = set()
    if output_file.exists():
        logger.info(f"Found existing output file {output_file}. Checking for completed samples...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"Found {len(existing_ids)} completed samples.")

    # Filter dataset
    if existing_ids:
        # Assuming process_sample saves "id": index.
        # So we just need to skip indices present in existing_ids.
        indices_to_process = [i for i in range(len(dataset)) if i not in existing_ids]
        logger.info(f"Resuming generation. {len(indices_to_process)} samples remaining.")
    else:
        indices_to_process = range(len(dataset))

    if not indices_to_process:
        logger.info("All samples already generated!")
        return []

    # 2. Setup OpenAI Client
    client = AsyncOpenAI(base_url=server_url.rstrip("/") + "/v1", api_key="EMPTY")
    
    if not model_name:
        models = await client.models.list()
        model_name = models.data[0].id
        logger.info(f"Fetched model name from server: {model_name}")

    # 3. Generate
    logger.info(f"Starting generation with model {model_name}...")
    logger.info(f"Params: temp={temperature}, top_p={top_p}, top_k={top_k}")
    start_time = time.time()
    
    # Create a semaphore to limit concurrency
    sem = asyncio.Semaphore(concurrency)
    
    async def process_sample(index):
        example = dataset[index]
        messages = example["prompt"]
        try:
            async with sem:
                # Prepare extra parameters for vLLM if needed
                extra_body = {}
                if top_k != -1:
                    extra_body["top_k"] = top_k

                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body=extra_body if extra_body else None
                )
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                
                if not content:
                    logger.warning(f"Sample {index}: Received empty content from model.")
                    
                return {
                    "id": index,
                    "original_example": example,
                    "completion": content,
                    "finish_reason": finish_reason,
                    "domain": domain,
                    "model": model_name
                }
        except Exception as e:
            logger.error(f"Error generating for sample {index}: {e}")
            return {
                "id": index,
                "original_example": example,
                "completion": "",
                "finish_reason": "error",
                "domain": domain,
                "model": model_name,
                "error": str(e)
            }

    tasks = [process_sample(i) for i in indices_to_process]
    
    results = []
    truncation_count = 0
    
    # Open file in append mode for streaming results
    with open(output_file, "a", encoding="utf-8") as f:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
            res = await fut
            results.append(res)
            
            # Write immediately
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush() # Ensure it hits the disk
            
            if res.get("finish_reason") == "length":
                truncation_count += 1
    
    end_time = time.time()
    logger.info(f"Generation finished in {end_time - start_time:.2f}s")
    logger.info(f"Truncation Rate (this run): {truncation_count}/{len(results)} samples stopped due to length limit.")
    
    return results

def evaluate_responses(
    input_file: Path,
    domain: str,
    output_dir: Path
):
    logger.info(f"Loading responses from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    logger.info(f"Evaluating {len(data)} samples...")
    
    eval_output_file = output_dir / "eval_results.jsonl"
    
    stats = {
        "correct_count": 0,
        "evaluated_count": 0,
        "format_correct_count": 0,
        # Split-wise stats
        "train": {"correct": 0, "total": 0},
        "test": {"correct": 0, "total": 0},
        "unknown": {"correct": 0, "total": 0}
    }
    
    with open(eval_output_file, "w", encoding="utf-8") as f:
        for record in tqdm(data, desc="Evaluating"):
            prediction = record.get("completion", "")
            orig = record.get("original_example")
            
            if not orig:
                logger.warning(f"Skipping record {record.get('id')} due to missing original example")
                continue
            
            # Identify split
            ds_split = orig.get("dataset_split", "unknown")
            if ds_split not in stats:
                ds_split = "unknown"

            is_correct, is_format_correct = evaluate_sample(domain, prediction, orig)
            
            if is_correct is not None:
                stats["evaluated_count"] += 1
                stats[ds_split]["total"] += 1
                if is_correct:
                    stats["correct_count"] += 1
                    stats[ds_split]["correct"] += 1
            
            if is_format_correct:
                stats["format_correct_count"] += 1
            
            record["is_correct"] = is_correct
            record["is_format_correct"] = is_format_correct
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    logger.info(f"Evaluation results saved to {eval_output_file}")
    
    # Log aggregate metrics
    if stats["evaluated_count"] > 0:
        accuracy = stats["correct_count"] / stats["evaluated_count"]
        logger.info(f"Overall Accuracy: {accuracy:.2%} ({stats['correct_count']}/{stats['evaluated_count']})")
        
        # Log split metrics
        for s in ["train", "test"]:
            if stats[s]["total"] > 0:
                acc = stats[s]["correct"] / stats[s]["total"]
                logger.info(f"{s.capitalize()} Accuracy: {acc:.2%} ({stats[s]['correct']}/{stats[s]['total']})")
    
    if len(data) > 0:
        format_acc = stats["format_correct_count"] / len(data)
        logger.info(f"Format Compliance: {format_acc:.2%} ({stats['format_correct_count']}/{len(data)})")

def run_eval(
    action: str,
    domain: str,
    output_dir: str,
    server_url: Optional[str] = None,
    model_name: Optional[str] = None,
    split: str = "test",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    input_file: Optional[str] = None,
    use_timestamp: bool = True
):
    # Setup output directory
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"{domain}_{split}_{timestamp}"
    else:
        # If no timestamp, we assume the output_dir IS the run dir
        run_dir = Path(output_dir)
        
    setup_logger(run_dir)
    
    # Save experiment configuration
    config = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "action": action,
        "domain": domain,
        "split": split,
        "model_name": model_name,
        "server_url": server_url,
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        },
        "input_file": input_file
    }
    with open(run_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Action: {action}")
    logger.info(f"Domain: {domain}, Split: {split}")
    
    gen_file = run_dir / "generated_responses.jsonl"
    
    if action in ["generate", "all"]:
        if not server_url:
            raise ValueError("server_url is required for generation mode")
            
        asyncio.run(
            generate_responses(
                server_url=server_url,
                domain=domain,
                split=split,
                output_file=gen_file,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        )
        
        # If we just generated, and action is 'all', use the generated file for evaluation
        if action == "all":
            input_file = str(gen_file)

    if action in ["evaluate", "all"]:
        if not input_file:
            if action == "evaluate":
                raise ValueError("input_file is required for evaluation-only mode")
            # For 'all', it should have been set above
            input_file = str(gen_file)
            
        evaluate_responses(
            input_file=Path(input_file),
            domain=domain,
            output_dir=run_dir
        )
