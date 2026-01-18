import re
import sys
from datasets import load_dataset
from pathlib import Path
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

try:
    from datacalibrator.seed import SEED as GLOBAL_SEED
except ImportError:
    GLOBAL_SEED = 42

logger = get_logger()

# Logic to find the dataset path relative to the project root
DEFAULT_DATASET_ID = str(Path(__file__).resolve().parents[4] / "datasets" / "code_domain" / "bigcodebench")

def ensure_dataset_exists(dataset_id):
    dataset_path = Path(dataset_id)
    if not dataset_path.exists():
        return
        
    # Check if jsonl or csv files exist
    has_subset = False
    for subset in ['train', 'test']:
        for ext in ['jsonl', 'csv']:
            if (dataset_path / f"{subset}.{ext}").exists():
                has_subset = True
                break
        if has_subset:
            break
            
    if not has_subset:
        logger.info(f"No jsonl/csv subsets found in {dataset_id}. Converting...")
        try:
            # Add project root to path to import datacalibrator
            project_root_path = dataset_path.parents[2]
            if str(project_root_path) not in sys.path:
                sys.path.append(str(project_root_path))
                
            from datacalibrator.datasets.code_adaptor import get_code_dataset
            train_ds, test_ds = get_code_dataset()
            
            # Save to jsonl
            train_ds.to_json(dataset_path / "train.jsonl")
            test_ds.to_json(dataset_path / "test.jsonl")
            logger.info("Dataset conversion completed.")
            
        except ImportError as e:
            logger.warning(f"Failed to import code_adaptor: {e}. Skipping conversion.")
        except Exception as e:
            logger.error(f"Error during dataset conversion: {e}")

ensure_dataset_exists(DEFAULT_DATASET_ID)

@register_benchmark(
    BenchmarkMeta(
        name='bigcodebench',
        dataset_id=DEFAULT_DATASET_ID,
        pretty_name='BigCodeBench',
        tags=[Tags.CODING],
        subset_list=['train', 'test'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        few_shot_num=0,
        shuffle=False,
        review_timeout=240, # 参考 bigcodebench 评测时间设置
        prompt_template="Write a python function to solve following problem. \n{question}"
    )
)
class BigCodeBenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object with Chat format input.
        If using preprocessed JSONL, 'prompt' and 'completion' are already formatted.
        """
        query = record['instruct_prompt']
        full_prompt = self.prompt_template.format(question=query)

        target_answer = f"{record['code_prompt']}\n{record['canonical_solution']}"

        return Sample(input = [ChatMessageUser(content=full_prompt)],
                      target = target_answer,
                      metadata={
                          "task_id": record["task_id"],
                          "entry_point": record["entry_point"],
                          "test": record["test"],
                          "instruct_prompt": record["instruct_prompt"],
                          "code_prompt": record["code_prompt"]
                      }
        )
    
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        return self._postprocess(prediction)
    
    
    @classmethod
    def _postprocess(cls, text: str) -> str:
        """Extract code from markdown code blocks."""
        blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
        if len(blocks) >= 1:
            text = blocks[-1]
        return text
    
    def match_score(
            self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # Execute code and check correctness 
        assert not self.use_sandbox, 'BigCodeBench currently only supports non-sandboxed evaluation.'
        from .utils import check_correctness, TIMEOUT_LIMIT

        problem = {
            'task_id': task_state.sample.metadata['task_id'],
            'prompt': task_state.sample.metadata['code_prompt'],
            'entry_point': task_state.sample.metadata['entry_point'],
            'test': task_state.sample.metadata['test']
        }
        
        timeout = self.review_timeout if self.review_timeout else TIMEOUT_LIMIT
        result = check_correctness(problem, filtered_prediction, timeout)

        # set score values
        score.value = {'acc': 1.0 if result['passed'] else 0.0}
        score.metadata['result'] = result['result']
        
        return score