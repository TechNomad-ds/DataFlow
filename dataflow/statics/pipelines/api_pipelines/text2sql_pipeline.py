from dataflow.operators.generate.Text2SQL import *
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request


class Text2SQLPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/Text2SQLPipeline/pipeline.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        api_llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=100
        )

        # It is recommended to use better LLMs for the generation of Chain-of-Thought (CoT) reasoning process.
        cot_generation_api_llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o", # You can change to a more powerful model for CoT generation
            max_workers=100
        )

        component_difficulty_config = {
            'thresholds': [2, 4, 6],      
            'labels': ['easy', 'medium', 'hard', 'extra']
        }

        execution_difficulty_config = {
            'thresholds': [2, 4, 6],
            'labels': ['easy', 'medium', 'hard', 'extra']
        }

        # A demo database is provided. Download it from the following URL and update the path:  
        # https://huggingface.co/datasets/Open-Dataflow/dataflow-Text2SQL-database-example  
        db_root_path = ""  
        table_info_file = "../example_data/Text2SQLPipeline/dev_tables.jsonl"
        bach_size = 100
        
        self.sql_filter_step1 = SQLFilter(
            llm_serving=api_llm_serving,
            db_root_path=db_root_path,
            meta_time_out=120,
            bach_size=bach_size
        )

        self.component_classifier_step2 = ComponentClassifier(
            difficulty_config=component_difficulty_config
        )

        self.schema_extractor_step3 = SchemaExtractor(
            table_info_file=table_info_file,
            db_root_path=db_root_path
        )

        self.question_refiner_step4 = QuestionRefiner(
            llm_serving=api_llm_serving,
            bach_size=bach_size
        )

        self.qa_generator_step5 = QAGenerator(
            llm_serving=cot_generation_api_llm_serving,
            db_root_path=db_root_path,
            meta_time_out=60,
            bach_size=bach_size
        )

        self.execution_classifier_step6 = ExecutionClassifier(
            llm_serving=api_llm_serving,
            db_root_path=db_root_path,
            meta_time_out=120,
            difficulty_config=execution_difficulty_config,
            bach_size=bach_size
        )
        
        
    def forward(self):

        input_sql_key = "SQL"
        input_dbid_key = "db_id"
        input_question_key = "question"

        self.sql_filter_step1.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            input_dbid_key=input_dbid_key,
            input_question_key=input_question_key
        )

        self.component_classifier_step2.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            output_difficulty_key="sql_component_difficulty"
        )

        self.schema_extractor_step3.run(
            storage=self.storage.step(),
            input_dbid_key=input_dbid_key,
            input_sql_key=input_sql_key,
            output_raw_schema_key="whole_schema",
            output_ddl_key="ddl",
            output_whole_format_schema_key="whole_format_schema",
            output_used_schema_key="selected_schema"        
        )

        self.question_refiner_step4.run(
            storage=self.storage.step(),
            input_question_key=input_question_key,
            output_refined_question_key="refined_question"
        )

        self.qa_generator_step5.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            input_question_key=input_question_key,
            input_dbid_key=input_dbid_key,
            input_schema_key="ddl",
            output_sft_prompt_key="sft_prompt",
            output_rl_prompt_key="rl_prompt",
            output_cot_key="sft_output"
        )

        self.execution_classifier_step6.run(
            storage=self.storage.step(),
            input_dbid_key=input_dbid_key,
            input_sql_key=input_sql_key,
            input_prompt_key="rl_prompt",
            output_difficulty_key="sql_execution_difficulty"
        )
        
if __name__ == "__main__":
    model = Text2SQLPipeline()
    model.forward()

