import pandas as pd
import os
import re
import sqlite3
import sys
from func_timeout import func_timeout, FunctionTimedOut
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class ExecutionClassifier(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, db_root_path: str, meta_time_out: float = 120.0, difficulty_config: dict = None, num_generations: int = 10, batch_size: int = 50):
        self.llm_serving = llm_serving     
        self.db_root_path = db_root_path
        self.meta_time_out = meta_time_out
        if difficulty_config is None:
            self.difficulty_config = {
                'thresholds': [2, 5, 9],
                'labels': ['extra', 'hard', 'medium', 'easy']
            }
        else:
            self.difficulty_config = difficulty_config
        self.num_generations = num_generations
        self.batch_size = batch_size 
        self.num_cpus = os.cpu_count() if os.cpu_count() else 20
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于根据大模型生成SQL的准确率对Text2SQL问题进行难度分类。\n\n"
                "输入参数：\n"
                "- input_sql_key：原始SQL的键名\n"
                "- input_dbid_key：数据库ID的键名\n"
                "- input_prompt_key：Text2SQL提示词的键名\n"
                "- db_root_path：数据库文件的根目录路径\n"
                "- num_cpus：并行执行的CPU线程数\n"
                "- meta_time_out：SQL执行的超时时间\n\n"
                "难度划分阈值：\n"
                "- easy_medium：正确SQL数 ≥ 该值为easy难度（默认：9）\n"
                "- medium_hard：正确SQL数 ≥ 该值且 < easy_medium为medium难度（默认：5）\n"
                "- hard_extra：正确SQL数 ≥ 该值且 < medium_hard为hard难度（默认：2）\n\n"
                "输出参数：\n"
                "- output_difficulty_key：SQL执行难度标签的键名"
            )
        elif lang == "en":
            return (
                "This operator classifies Text2SQL difficulty based on the accuracy of LLM-generated SQL.\n\n"
                "Input parameters:\n"
                "- input_sql_key: Key for original SQL (default: 'SQL')\n"
                "- input_dbid_key: Key for database ID (default: 'db_id')\n"
                "- input_prompt_key: Key for Text2SQL prompts (default: 'rl_prompt')\n"
                "- db_root_path: Root path of database files\n"
                "- num_cpus: Number of parallel CPU threads (default: 1)\n"
                "- meta_time_out: SQL execution timeout in seconds (default: 120.0)\n\n"
                "Difficulty thresholds:\n"
                "- easy_medium: Correct SQLs ≥ this value → easy (default: 9)\n"
                "- medium_hard: Correct SQLs ≥ this value and < easy_medium → medium (default: 5)\n"
                "- hard_extra: Correct SQLs ≥ this value and < medium_hard → hard (default: 2)\n\n"
                "Output parameters:\n"
                "- output_difficulty_key: Key for SQL difficulty labels (default: 'sql_execution_difficulty')"
            )
        else:
            return "Text2SQLDifficultyClassifier performs difficulty classification based on SQL execution accuracy."
    
    @staticmethod
    def parse_response(response, logger):
        pattern = r"```sql\s*(.*?)\s*```"
        
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            logger.warning(f"No SQL blocks found in {response}.")
            return response
        
    @staticmethod
    def execute_sql(sql, db_path):    
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            return result
        finally:
            if conn:
                conn.close()

    @staticmethod
    def execute_model(predicted_sqls, ground_truth, db_place, idx, meta_time_out, logger):
        results = []
        cnt_true = 0
        
        try:
            ground_truth_res = func_timeout(meta_time_out, ExecutionClassifier.execute_sql,
                                        args=(ground_truth, db_place))
            
            for predicted_sql in predicted_sqls:
                res = 0
                try:
                    predicted_res = func_timeout(meta_time_out, ExecutionClassifier.execute_sql,
                                            args=(predicted_sql, db_place))
                    if set(predicted_res) == set(ground_truth_res):
                        res = 1
                        cnt_true += 1
                    
                    result = {'res': res, 'sql': predicted_sql}
                except KeyboardInterrupt:
                    sys.exit(0)
                except FunctionTimedOut:
                    result = {'res': 0, 'sql': predicted_sql, 'error': 'timeout'}
                except Exception as e:
                    result = {'res': 0, 'sql': predicted_sql, 'error': str(e)}
                
                results.append(result)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
            sys.exit(0)
        except FunctionTimedOut:
            logger.warning(f"timeout when execute gold sql of question {idx}")
            cnt_true = -1  
        except Exception as e:
            logger.warning(f"error: {e} when execute gold sql of question {idx}")
            cnt_true = -1  

        return {"idx": idx, "cnt_true": cnt_true, "results": results}
    
    def run_sqls_parallel(self, datas, db_root_path, num_cpus=1, meta_time_out=30.0):
        pbar = tqdm(total=len(datas), desc="Executing SQLs")
        exec_result = []

        def wrap_task(data_pair, idx):
            predicted_sqls = data_pair[self.output_predicted_sqls_key]
            ground_truth = data_pair[self.input_sql_key]
            db_id = data_pair[self.input_dbid_key].replace('\n', '')
            db_id = re.sub(r'[^A-Za-z0-9_]', '', db_id)
            db_place = os.path.join(db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")
            return ExecutionClassifier.execute_model(predicted_sqls, ground_truth, db_place, idx, meta_time_out, self.logger)

        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = [
                executor.submit(wrap_task, data_pair, i)
                for i, data_pair in enumerate(datas)
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    exec_result.append(result)
                except Exception as e:
                    self.logger.error(f"Error in SQL execution: {e}")
                    exec_result.append(None)
                pbar.update()

        pbar.close()
        return exec_result

    def sort_results(self, list_of_dicts):
        return sorted(list_of_dicts, key=lambda x: x['idx'])
        
    def report_statistics(self, dataframe: pd.DataFrame):
        counts = dataframe[self.output_difficulty_key].value_counts()
        self.logger.info("SQL Difficulty Statistics")
        stats = [f"{difficulty.title()}: {counts.get(difficulty, 0)}" for difficulty in ['easy', 'medium', 'hard', 'extra']]
        self.logger.info(", ".join(stats))
        
    def process_batch_questions(self, questions):
        try:
            responses = self.llm_serving.generate_from_input(questions)
            if len(responses) != len(questions):
                self.logger.warning(f"Expected {len(questions)} responses but got {len(responses)}")
                while len(responses) < len(questions):
                    responses.append("")
            return responses
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return [""] * len(questions)
        
    def classify_difficulty(self, score):
        if score == -1:
            return "gold error"
        thresholds = self.difficulty_config['thresholds']
        labels = self.difficulty_config['labels']
        
        for i, threshold in enumerate(thresholds):
            if score <= threshold:
                return labels[i]
        return labels[-1]

    def run(self, storage: DataFlowStorage,
            input_dbid_key: str = "db_id",
            input_sql_key: str = "SQL",
            input_prompt_key: str = "rl_prompt",
            output_difficulty_key: str = "sql_execution_difficulty"
        ):
        self.input_sql_key = input_sql_key
        self.input_prompt_key = input_prompt_key
        self.input_dbid_key = input_dbid_key
        self.output_difficulty_key = output_difficulty_key
        
        self.output_predicted_sqls_key = "_temp_predicted_sqls"
        self.output_cnt_true_key = "_temp_cnt_true"
        
        dataframe = storage.read("dataframe")
        input_prompts = dataframe[self.input_prompt_key].tolist()
        
        self.logger.info(f"Processing {len(input_prompts)} questions, generating {self.num_generations} SQLs each...")
        
        repeated_questions = [q for q in input_prompts for _ in range(self.num_generations)]
        
        all_responses = []
        total_batches = (len(repeated_questions) + self.batch_size - 1) // self.batch_size
        
        for batch_start in range(0, len(repeated_questions), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(repeated_questions))
            batch_questions = repeated_questions[batch_start:batch_end]
            
            current_batch = batch_start // self.batch_size + 1
            self.logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_questions)} questions)")
            
            batch_responses = self.process_batch_questions(batch_questions)
            all_responses.extend(batch_responses)
        
        responses = all_responses
        
        datas = dataframe.to_dict(orient='records')
        
        for i, data in enumerate(datas):
            start_idx = i * self.num_generations
            end_idx = start_idx + self.num_generations
            question_responses = responses[start_idx:end_idx]
            
            parsed_sqls = []
            for response in question_responses:
                if response:
                    parsed_sql = self.parse_response(response, self.logger)
                    parsed_sqls.append(parsed_sql)
                else:
                    parsed_sqls.append("")
            
            data[self.output_predicted_sqls_key] = parsed_sqls

        exec_result = self.run_sqls_parallel(datas, self.db_root_path, 
                                            num_cpus=self.num_cpus, 
                                            meta_time_out=self.meta_time_out)
        exec_result = self.sort_results(exec_result)
        
        for execres in exec_result:
            if execres is not None:
                idx = execres["idx"]
                cnt_true = execres["cnt_true"]
                datas[idx][self.output_difficulty_key] = self.classify_difficulty(cnt_true)
                datas[idx][self.output_cnt_true_key] = cnt_true
        
        for data in datas:
            data.pop(self.output_predicted_sqls_key, None)
            data.pop(self.output_cnt_true_key, None)
        
        dataframe = pd.DataFrame(datas)
        
        self.report_statistics(dataframe)
        
        difficulty_counts = dataframe[self.output_difficulty_key].value_counts()
        self.logger.info("\nDifficulty Distribution:")
        for difficulty in ['easy', 'medium', 'hard', 'extra', 'gold error']:
            count = difficulty_counts.get(difficulty, 0)
            percentage = count / len(dataframe) * 100
            self.logger.info(f"  {difficulty}: {count} ({percentage:.1f}%)")
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Difficulty classification results saved to {output_file}")

        return [self.output_difficulty_key]