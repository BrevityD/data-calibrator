import argparse
import sys
from pathlib import Path

# Add project root to sys.path so we can import modules
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
sys.path.append(str(project_root))

from evaluation.core import run_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with vLLM Server (OpenAI compatible)")
    parser.add_argument("--action", type=str, default="all", choices=["generate", "evaluate", "all"], 
                        help="Action to perform: generate responses, evaluate existing responses, or both (all)")
    parser.add_argument("--domain", type=str, required=True, choices=["code", "logic", "math"], help="Domain to evaluate")
    parser.add_argument("--server_url", type=str, help="URL of the vLLM server (required for generation)")
    parser.add_argument("--model", type=str, default=None, help="Model name on server (optional)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--input_file", type=str, help="Input JSONL file for evaluation-only mode")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for generation")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling")
    
    args = parser.parse_args()
    
    run_eval(
        action=args.action,
        domain=args.domain,
        server_url=args.server_url,
        model_name=args.model,
        output_dir=args.output_dir,
        split=args.split,
        input_file=args.input_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
