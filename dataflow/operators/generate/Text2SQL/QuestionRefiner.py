from typing import Dict, List, Tuple
import pandas as pd
import os
from tqdm import tqdm
from dataflow.prompts.text2sql import QuestionRefinePrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class QuestionRefiner(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC): 
        self.llm_serving = llm_serving       
        self.prompt = QuestionRefinePrompt()
        self.logger = get_logger()
        self.num_threads = os.cpu_count() if os.cpu_count() else 20
        self.max_retries = 3

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于对已有的自然语言问题进行润色改写。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题键\n"
                "- num_threads: 多线程并行数\n\n"
                "输出参数：\n"
                "- output_refined_question_key: 生成的润色后问题的key"
            )
        elif lang == "en":
            return (
                "This operator is used to refine and rewrite existing natural language questions.\n\n"
                "Input parameters:\n"
                "- input_question_key: Question key\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output parameters:\n"
                "- output_refined_question_key: The key for the generated refined question"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    def _generate_prompts_batch(self, items: List[Dict]) -> List[str]:
        return [self.prompt.question_refine_prompt(item[self.input_question_key]) for item in items]

    def _parse_response(self, response: str, original_question: str) -> str:
        if not response:
            return original_question
            
        response_upper = response.upper()
        if "RESULT: NO" in response_upper:
            return original_question
            
        try:
            result_line = next(
                line for line in response.split('\n') 
                if line.upper().startswith("RESULT:")
            )
            return result_line.split("RESULT:", 1)[1].strip()
        except (StopIteration, IndexError):
            self.logger.warning(f"Unexpected response format: {response[:200]}...")
            return original_question

    def _process_batch_with_retry(self, batch_items: List[Tuple[int, Dict]], retry_count: int = 0) -> List[Tuple[int, Dict]]:
        try:
            indices = [idx for idx, _ in batch_items]
            items = [item for _, item in batch_items]
            prompts = self._generate_prompts_batch(items)
            responses = self.llm_serving.generate_from_input(prompts)
            results = []
            for idx, item, response in zip(indices, items, responses):
                parsed_response = self._parse_response(response, item[self.input_question_key])
                results.append((idx, {
                    **item,
                    self.output_refined_question_key: parsed_response
                }))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")

            if retry_count < self.max_retries:
                self.logger.warning(f"Retrying batch (attempt {retry_count + 1})")
                return self._process_batch_with_retry(batch_items, retry_count + 1)
            else:
                self.logger.warning("Batch processing failed, using original questions")
                results = []
                for idx, item in batch_items:
                    results.append((idx, {
                        **item,
                        self.output_refined_question_key: item[self.input_question_key]
                    }))
                return results

    def run(self, storage: DataFlowStorage,
            input_question_key: str = "question",
            output_refined_question_key: str = "refined_question",
            batch_size: int = 50
        ):
        self.input_question_key = input_question_key
        self.output_refined_question_key = output_refined_question_key

        self.logger.info("Starting QuestionRefiner with batch processing...")
        raw_dataframe = storage.read("dataframe")
        items = raw_dataframe.to_dict('records')
        
        indexed_items = list(enumerate(items))
        
        batches = [indexed_items[i:i + batch_size] for i in range(0, len(indexed_items), batch_size)]
        self.logger.info(f"Processing {len(items)} items in {len(batches)} batches of size {batch_size}")
        
        all_results = [None] * len(items)
        
        for _, batch in enumerate(tqdm(batches, desc="Processing batches")):
            batch_results = self._process_batch_with_retry(batch)
            
            for idx, result in batch_results:
                all_results[idx] = result
        
        results = [r for r in all_results if r is not None]
        
        output_file = storage.write(pd.DataFrame(results))
        self.logger.info(f"Refined questions saved to {output_file}")

        return [self.output_refined_question_key]