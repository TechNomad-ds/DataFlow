from typing import Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sqlite3
import os
import re
from tqdm import tqdm
from dataflow.prompts.text2sql import Text2SQLCotPrompt
from dataflow.prompts.text2sql import FinalPromptGeneration
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class QAGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, db_root_path: str, batch_size: int = 50):
        self.llm_serving = llm_serving
        self.prompt = FinalPromptGeneration()
        self.cot_output = Text2SQLCotPrompt()
        self.db_root_path = db_root_path
        self.batch_size = batch_size
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于构建完整的提示词和思维链推理过程。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题键（如：question）\n"
                "- input_sql_key: SQL语句键（如：SQL）\n"
                "- input_schema_key: 数据库DDL信息键（如：ddl）\n"
                "- input_evidence_key: 输入中额外知识的键（如：evidence）\n"
                "- prompt_type: 提示词格式（如：omni-sql）\n"
                "- output_sft_prompt_key: SFT提示词输出键（如：sft_prompt）\n"
                "- output_rl_prompt_key: RL提示词输出键（如：rl_prompt）\n"
                "- output_cot_key: 思维链推理输出键（如：sft_output）\n"
                "- input_key: 输入数据主键（如：data）\n"
                "- input_dbid_key: 数据库ID键（如：db_id）\n"
                "- db_root_path: 数据库根目录（如：/mnt/public/data/.../dev_databases）\n"
                "输出参数：\n"
                "- output_sft_prompt_key: SFT提示词\n"
                "- output_rl_prompt_key: RL提示词\n"
                "- output_cot_key: 思维链推理输出"
            )
        elif lang == "en":
            return (
                "This operator is used to construct complete prompts and chain-of-thought reasoning processes.\n\n"
                "Input parameters:\n"
                "- input_question_key: Key for the question (e.g., 'question')\n"
                "- input_sql_key: Key for the SQL statement (e.g., 'SQL')\n"
                "- input_schema_key: Key for the database DDL information (e.g., 'ddl')\n"
                "- input_evidence_key: Key for additional knowledge in the input (e.g., 'evidence')\n"
                "- prompt_type: Prompt format (e.g., 'omni-sql')\n"
                "- output_sft_prompt_key: Output key for SFT prompt (e.g., 'sft_prompt')\n"
                "- output_rl_prompt_key: Output key for RL prompt (e.g., 'rl_prompt')\n"
                "- output_cot_key: Output key for chain-of-thought reasoning (e.g., 'sft_output')\n"
                "- input_key: Main key for input data (e.g., 'data')\n"
                "- input_dbid_key: Key for database ID (e.g., 'db_id')\n"
                "- db_root_path: Root path of the databases (e.g., '/mnt/public/data/.../dev_databases')\n"
                "Output parameters:\n"
                "- output_sft_prompt_key: SFT prompt\n"
                "- output_rl_prompt_key: RL prompt\n"
                "- output_cot_key: Chain-of-thought reasoning output"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    def generate_prompt(self, item: Dict, prompt_type: str) -> str:
        generated_prompt = None
        if prompt_type == 'dail-sql':
            generated_prompt = self.prompt.dial_sql_cot_prompt(
                    question=item.get(self.input_question_key),
                    schema=item.get(self.input_schema_key)
            )
        elif prompt_type == 'omni-sql':
            generated_prompt = self.prompt.omni_sql_cot_prompt(
                    question=item.get(self.input_question_key),
                    schema=item.get(self.input_schema_key)
            )
        return generated_prompt
    
    def generate_cot_synthesis_prompts(self, item: Dict, is_backup=False) -> str:
        if not is_backup:
            cot_synthesis_prompt = self.cot_output.text2sql_cot_prompt(
                item.get(self.input_schema_key),
                item.get(self.input_question_key),
                item.get(self.input_sql_key)
            )
        else:
            cot_synthesis_prompt = self.cot_output.text2sql_cot_prompt_backup(
                item.get(self.input_schema_key),
                item.get(self.input_question_key),
                item.get(self.input_sql_key)
            )
    
        return cot_synthesis_prompt
    
    def execute_sql(self, sql, db_path, timeout=10):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout = 5000")
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            self.logger.error(f"SQL执行错误: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def extract_sql(self, response):
        pattern = r"```sql\s*(.*?)\s*```"
        
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            return ""
    
    def _parse_response(self, response: str, gold_sql: str, db_path) -> Tuple[Optional[str], bool]:
        generated_sql = self.extract_sql(response)
        if not generated_sql:
            return None, False
        
        try:
            gen_result = self.execute_sql(generated_sql, db_path)
            gold_result = self.execute_sql(gold_sql, db_path)
            
            if gen_result is None or gold_result is None:
                return generated_sql, False
                
            return generated_sql, gen_result == gold_result 
        except Exception as e:
            self.logger.warning(f"SQL执行失败: {e}")
            return generated_sql, False

    def _parse_backup_response(self, response: str) -> Tuple[Optional[str], bool]:
        response = response.strip()
        if not response:
            return None, False

        lower_response = response.lower()
        keywords = ["let"] 
        
        for keyword in keywords:
            idx = lower_response.find(keyword)
            if idx != -1:
                return response[idx:], True
        
        return None, False

    def _process_batch(self, batch_items: List[Dict]) -> List[Dict]:
        batch_results = []
        for item in batch_items:
            sft_prompt = self.generate_prompt(item, prompt_type="omni-sql")
            rl_prompt = self.generate_prompt(item, prompt_type="dail-sql")
            batch_results.append({
                **item,
                self.output_sft_prompt_key: sft_prompt if sft_prompt else '',
                self.output_rl_prompt_key: rl_prompt if rl_prompt else '',
                self.output_cot_key: '' 
            })
        
        max_retries = 3
        for retry_count in range(max_retries + 1):
            cot_prompts = []
            items_to_process = []
            
            for i, item in enumerate(batch_items):
                if not batch_results[i][self.output_cot_key]: 
                    cot_prompt = self.generate_cot_synthesis_prompts(item, False)
                    cot_prompts.append(cot_prompt)
                    items_to_process.append((i, item))
            
            if not cot_prompts:
                break
                
            try:
                cot_responses = self.llm_serving.generate_from_input(cot_prompts)
                
                for (i, item), response in zip(items_to_process, cot_responses):
                    db_id = item.get(self.input_dbid_key)
                    gold_sql = item.get(self.input_sql_key)
                    db_path = os.path.join(self.db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")
                    
                    parsed_response, flag = self._parse_response(response, gold_sql, db_path)
                    
                    if flag and parsed_response:
                        batch_results[i][self.output_cot_key] = parsed_response
                        
            except Exception as e:
                self.logger.warning(f"Batch processing attempt {retry_count} failed: {e}")
        
        backup_prompts = []
        backup_indices = []
        
        for i, item in enumerate(batch_items):
            if not batch_results[i][self.output_cot_key]:
                backup_prompt = self.generate_cot_synthesis_prompts(item, True)
                backup_prompts.append(backup_prompt)
                backup_indices.append(i)
        
        if backup_prompts:
            try:
                backup_responses = self.llm_serving.generate_from_input(backup_prompts)
                for i, response in zip(backup_indices, backup_responses):
                    parsed_backup, success = self._parse_backup_response(response)
                    batch_results[i][self.output_cot_key] = parsed_backup if success and parsed_backup else ''
            except Exception as e:
                self.logger.error(f"Backup batch processing failed: {e}")
        
        return batch_results

    def run(self, storage: DataFlowStorage, 
            input_sql_key: str = "SQL",
            input_question_key: str = "question",
            input_dbid_key: str = "db_id",
            input_schema_key: str = "ddl",
            output_sft_prompt_key: str = "sft_prompt",
            output_rl_prompt_key: str = "rl_prompt",
            output_cot_key: str = "sft_output"
        ):
        self.input_question_key = input_question_key
        self.input_sql_key = input_sql_key
        self.input_schema_key = input_schema_key
        self.input_dbid_key = input_dbid_key
        self.output_sft_prompt_key = output_sft_prompt_key
        self.output_rl_prompt_key = output_rl_prompt_key
        self.output_cot_key = output_cot_key
        
        self.logger.info("Starting prompt generation...")
        raw_dataframe = storage.read("dataframe")
        items = raw_dataframe.to_dict('records')
        final_results = []

        for i in range(0, len(items), self.batch_size):
            batch_items = items[i:i + self.batch_size]
            batch_results = self._process_batch(batch_items)
            final_results.extend(batch_results)
        
        if len(final_results) != len(items):
            self.logger.warning(f"Results count mismatch: expected {len(items)}, got {len(final_results)}")
        
        output_file = storage.write(pd.DataFrame(final_results))
        self.logger.info(f"Prompt generation completed, saved to {output_file}")

        return [self.output_sft_prompt_key, self.output_rl_prompt_key, self.output_cot_key]
