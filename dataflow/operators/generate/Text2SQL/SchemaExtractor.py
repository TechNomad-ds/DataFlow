import pandas as pd
import os
import sqlite3
import json
import re
from tqdm import tqdm
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlglot.optimizer.qualify import qualify
from sqlglot import parse_one, exp
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


SQLITE_RESERVED_KEYWORDS = {
    "abort", "action", "add", "after", "all", "alter", "analyze", "and", "as", "asc", "attach", "autoincrement",
    "before", "begin", "between", "by", "cascade", "case", "cast", "check", "collate", "column", "commit", "conflict",
    "constraint", "create", "cross", "current_date", "current_time", "current_timestamp", "database", "default",
    "deferrable", "deferred", "delete", "desc", "detach", "distinct", "drop", "each", "else", "end", "escape", "except",
    "exclusive", "exists", "explain", "fail", "for", "foreign", "from", "full", "glob", "group", "having", "if",
    "ignore", "immediate", "in", "index", "indexed", "initially", "inner", "insert", "instead", "intersect", "into",
    "is", "isnull", "join", "key", "left", "like", "limit", "natural", "no", "not", "notnull", "null", "of",
    "offset", "on", "or", "order", "outer", "plan", "pragma", "primary", "query", "raise", "recursive", "references",
    "regexp", "reindex", "release", "rename", "replace", "restrict", "right", "rollback", "row", "savepoint", "select",
    "set", "table", "temp", "temporary", "then", "to", "trigger", "union", "unique", "update", "using",
    "vacuum", "values", "view", "virtual", "when", "where", "with", "without"
}


@OPERATOR_REGISTRY.register()
class SchemaExtractor(OperatorABC):
    def __init__(self, table_info_file: str, db_root_path: str = ""):
        self.table_info_file = table_info_file
        self.db_root_path = db_root_path
        self.num_threads = os.cpu_count() if os.cpu_count() else 20
        self._schema_cache = {}
        self._db_schema_cache = {}
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子整合了数据库Schema提取和SQL解析功能，可以同时提供多种格式的Schema信息。\n\n"
                "输入参数：\n"
                "- table_info_file：tables.jsonl文件路径，包含数据库Schema信息\n"
                "- db_root_path：数据库文件根目录路径（可选，用于提取示例数据）\n"
                "- input_sql_key：SQL语句键（可选，用于Schema解析）\n"
                "- input_dbid_key：db_id键，数据库名\n"
                "- num_threads：多线程并行数\n\n"
                "输出参数：\n"
                "- output_raw_schema_key：原始Schema的JSON格式\n"
                "- output_ddl_key：数据库DDL语句\n"
                "- output_formatted_schema_key：格式化的Schema信息\n"
                "- output_used_schema_key：SQL中实际使用的表和列（如果提供SQL）"
            )
        elif lang == "en":
            return (
                "This operator integrates database schema extraction and SQL parsing functions, providing multiple formats of schema information.\n\n"
                "Input parameters:\n"
                "- table_info_file: Path to tables.jsonl file containing database schema information\n"
                "- db_root_path: Database root directory path (optional, for extracting example data)\n"
                "- input_sql_key: SQL statement key (optional, for schema parsing)\n"
                "- input_dbid_key: db_id key, database name\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output parameters:\n"
                "- output_raw_schema_key: Raw schema in JSON format\n"
                "- output_ddl_key: Database DDL statements\n"
                "- output_formatted_schema_key: Formatted schema information\n"
                "- output_used_schema_key: Actually used tables and columns in SQL (if SQL provided)"
            )
        else:
            return "Unified schema processor for Text2SQL tasks."

    def load_schema_info(self):
        if self._schema_cache:
            return self._schema_cache
            
        try:
            with open(self.table_info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        schema_info = json.loads(line.strip())
                        db_id = schema_info['db_id']
                        self._schema_cache[db_id] = schema_info
                        
        except Exception as e:
            self.logger.error(f"Error loading schema info from {self.table_info_file}: {e}")
            
        return self._schema_cache

    def get_schema_for_db(self, db_id):
        schema_cache = self.load_schema_info()
        return schema_cache.get(db_id, {})

    def extract_detailed_schema(self, db_info: Dict, db_id: str) -> Dict:
        schema = {
            'tables': {},
            'foreign_keys': [],
            'primary_keys': []
        }
        
        table_names = db_info.get("table_names_original", [])
        column_names = db_info.get("column_names_original", [])
        column_types = db_info.get("column_types", [])
        primary_keys = db_info.get("primary_keys", [])
        foreign_keys = db_info.get("foreign_keys", [])
        
        for i, col_info in enumerate(column_names):
            if col_info[0] == -1:
                continue
                
            table_idx = col_info[0]
            if table_idx >= len(table_names):
                continue
                
            table_name = table_names[table_idx]
            col_name = col_info[1]
            col_type = column_types[i] if i < len(column_types) else "text"
            
            if table_name not in schema['tables']:
                schema['tables'][table_name] = {
                    'columns': {},
                    'primary_keys': []
                }
            
            schema['tables'][table_name]['columns'][col_name] = {
                'type': col_type,
                'examples': []
            }
        
        if self.db_root_path:
            db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                self._extract_examples(schema, db_path)
        
        for pk in primary_keys:
            if isinstance(pk, list):  
                table_idxs = {column_names[col_idx][0] for col_idx in pk if col_idx < len(column_names)}
                if len(table_idxs) != 1 or -1 in table_idxs:
                    continue
                    
                table_idx = table_idxs.pop()
                if table_idx >= len(table_names):
                    continue
                    
                table_name = table_names[table_idx]
                col_names = [
                    column_names[col_idx][1]
                    for col_idx in pk if col_idx < len(column_names)
                ]
                
                if table_name in schema['tables']:
                    schema['tables'][table_name]['primary_keys'].extend(col_names)
                    schema['primary_keys'].append({
                        'table': table_name,
                        'columns': col_names
                    })
            else:
                if pk >= len(column_names) or column_names[pk][0] == -1:
                    continue
                    
                table_idx = column_names[pk][0]
                if table_idx >= len(table_names):
                    continue
                    
                table_name = table_names[table_idx]
                col_name = column_names[pk][1]
                
                if table_name in schema['tables']:
                    schema['tables'][table_name]['primary_keys'].append(col_name)
                    schema['primary_keys'].append({
                        'table': table_name,
                        'column': col_name
                    })
        
        for fk in foreign_keys:
            if len(fk) != 2:
                continue
                
            src_col_idx, ref_col_idx = fk
            
            if (src_col_idx >= len(column_names) or ref_col_idx >= len(column_names) or
                column_names[src_col_idx][0] == -1 or column_names[ref_col_idx][0] == -1):
                continue
            
            src_table_idx = column_names[src_col_idx][0]
            ref_table_idx = column_names[ref_col_idx][0]
            
            if src_table_idx >= len(table_names) or ref_table_idx >= len(table_names):
                continue
                
            src_table = table_names[src_table_idx]
            src_col = column_names[src_col_idx][1]
            ref_table = table_names[ref_table_idx]
            ref_col = column_names[ref_col_idx][1]
            
            schema['foreign_keys'].append({
                'source_table': src_table,
                'source_column': src_col,
                'referenced_table': ref_table,
                'referenced_column': ref_col
            })
                
        return schema

    def _extract_examples(self, schema: Dict, db_path: str):
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                for table_name, table_info in schema['tables'].items():
                    for col_name, col_info in table_info['columns'].items():
                        try:
                            sql_query = f'SELECT "{col_name}" FROM "{table_name}" LIMIT 2'
                            cursor.execute(sql_query)
                            examples = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
                            col_info['examples'] = examples
                        except sqlite3.Error as e:
                            self.logger.debug(f"Unable to access examples for {table_name}.{col_name}: {e}")
                            
        except Exception as e:
            self.logger.warning(f"Error extracting examples from {db_path}: {e}")

    def generate_ddl_from_schema(self, schema: Dict) -> str:
        ddl_statements = []
        
        for table_name, table_info in schema['tables'].items():
            columns_ddl = []
            
            for col_name, col_info in table_info['columns'].items():
                sql_type = {
                    "number": "INTEGER",
                    "text": "TEXT",
                    "date": "DATE",
                    "time": "TIME",
                    "datetime": "DATETIME"
                }.get(col_info['type'].lower(), "TEXT")
                
                columns_ddl.append(f"    {col_name} {sql_type}")
            
            if table_info['primary_keys']:
                pk_columns = ", ".join(table_info['primary_keys'])
                columns_ddl.append(f"    PRIMARY KEY ({pk_columns})")
            
            for fk in schema['foreign_keys']:
                if fk['source_table'] == table_name:
                    columns_ddl.append(
                        f"    FOREIGN KEY ({fk['source_column']}) "
                        f"REFERENCES {fk['referenced_table']}({fk['referenced_column']})"
                    )
            
            create_table_sql = (
                f"CREATE TABLE {table_name} (\n" +
                ",\n".join(columns_ddl) +
                "\n);"
            )
            ddl_statements.append(create_table_sql)
        
        return "\n\n".join(ddl_statements)

    def generate_formatted_schema(self, schema: Dict) -> str:
        formatted = []
        
        for table_name, table_info in schema['tables'].items():
            formatted.append(f"## Table: {table_name}")
            
            if table_info['primary_keys']:
                formatted.append(f"Primary Key: {', '.join(table_info['primary_keys'])}")
            
            formatted.append("Column Information:")
            for col_name, col_info in table_info['columns'].items():
                examples = ", ".join(col_info['examples']) if col_info['examples'] else "No examples"
                formatted.append(
                    f"- {col_name} ({col_info['type']}) "
                    f"Example: {examples}"
                )
            
            table_fks = [
                fk for fk in schema['foreign_keys'] 
                if fk['source_table'] == table_name
            ]
            if table_fks:
                formatted.append("Foreign Key:")
                for fk in table_fks:
                    formatted.append(
                        f"- {fk['source_column']} → "
                        f"{fk['referenced_table']}.{fk['referenced_column']}"
                    )
            
            formatted.append("")
        
        return "\n".join(formatted)

    def get_simple_schema_for_parsing(self, db_id):
        db_info = self.get_schema_for_db(db_id)
        if not db_info:
            return {}
            
        schema = {}
        table_names_original = db_info.get('table_names_original', [])
        column_names_original = db_info.get('column_names_original', [])
        
        for table_idx, table_name in enumerate(table_names_original):
            schema[table_name.lower()] = []
            
        for col_info in column_names_original:
            table_idx, col_name = col_info
            if table_idx >= 0 and table_idx < len(table_names_original):
                table_name = table_names_original[table_idx].lower()
                schema[table_name].append(col_name.lower())
        
        return schema

    def normalize_sql_column_references(self, sql: str, schema: dict, alias_map: dict) -> str:
        col_to_table = {}
        all_tables = []
        
        for table, cols in schema.items():
            all_tables.append(table)
            for col in cols:
                if col not in col_to_table:
                    col_to_table[col] = []
                col_to_table[col].append(table)

        col_fix_map = {}
        for col, tables in col_to_table.items():
            if len(tables) == 1:
                table = tables[0]
                alias = None
                for a, t in alias_map.items():
                    if t == table:
                        alias = a
                        break
                if alias:
                    col_fix_map[col] = f'"{alias}"."{col}"'
                    
        alias_pattern1 = re.compile(r'\bAS\s+"?([a-zA-Z_][\w]*)"?', re.IGNORECASE)
        alias_names = set(m.group(1) for m in alias_pattern1.finditer(sql))
        
        alias_pattern2 = re.compile(r'\bAS\s+(?:"(?P<dq>[^"]+)"|`(?P<bq>[^`]+)`)', re.IGNORECASE)
        for m in alias_pattern2.finditer(sql):
            alias = m.group('dq') or m.group('bq')
            alias_names.add(alias)

        def replace_col(m):
            col = m.group(0).strip('"')
            bef = m.string[max(0, m.start()-10):m.start()]
            
            if ('.' in bef or col in all_tables or col in alias_names or 
                col in SQLITE_RESERVED_KEYWORDS):
                return m.group(0)
                
            if ((m.group(0).startswith('"') and not m.group(0).endswith('"')) or
                (not m.group(0).startswith('"') and m.group(0).endswith('"'))):
                return m.group(0)
                
            return col_fix_map.get(col, m.group(0))

        if col_fix_map:
            pattern = re.compile(
                r'(?<![\w.])("?(' + '|'.join(re.escape(c) for c in col_fix_map.keys()) + r')"?)(?![\w])'
            )
            sql = pattern.sub(replace_col, sql)

        return sql

    def extract_alias_table_map(self, ast, alias_map):
        for table in ast.find_all(exp.Table):
            table_name = table.name.lower() if table.name else ""
            if table.alias:
                alias_map[table.alias.lower()] = table_name
            else:
                alias_map[table_name] = table_name

        for subquery in ast.find_all(exp.Subquery):
            if subquery.alias:
                inner_table = subquery.this.find(exp.Table)
                source_name = inner_table.name.lower() if inner_table and inner_table.name else "subquery"
                alias_map[subquery.alias.lower()] = source_name

    def get_cols(self, ast, alias_map, schema, used_columns):
        columns = list(ast.find_all(exp.Column))
        col_to_table = {}
        
        for table, cols in schema.items():
            for col in cols:
                if col not in col_to_table:
                    col_to_table[col] = []
                col_to_table[col].append(table)
        
        stars = list(ast.find_all(exp.Star))
        for star in stars:
            for table_name in schema.keys():
                if table_name in alias_map.values():
                    for col in schema[table_name]:
                        used_columns.add((table_name, col))
                
        for col in columns:
            col_name = col.name.lower() if col.name else ""
            table_ref = col.table.lower() if col.table else ""
            
            if table_ref == '':
                if col_name in col_to_table:
                    for table in col_to_table[col_name]:
                        if table in alias_map.values():
                            used_columns.add((table, col_name))
                            break
            else:
                if table_ref in alias_map and col_name in col_to_table:
                    actual_table = alias_map[table_ref]
                    if actual_table in schema and col_name in schema[actual_table]:
                        used_columns.add((actual_table, col_name))

    def convert_to_grouped_schema(self, used_columns):
        grouped_schema = {}
        for table, column in used_columns:
            if table not in grouped_schema:
                grouped_schema[table] = []
            if column not in grouped_schema[table]:
                grouped_schema[table].append(column)
        
        for table in grouped_schema:
            grouped_schema[table].sort()
            
        return grouped_schema

    def extract_used_schema_from_sql(self, sql, db_id):
        schema = self.get_simple_schema_for_parsing(db_id)
        if not schema:
            self.logger.warning(f"No schema found for database {db_id}")
            return {}
            
        used_columns = set()
        
        try:
            sql = sql.replace('`', '"')
            ast = parse_one(sql.lower())
            alias_map = {}
            self.extract_alias_table_map(ast, alias_map)

            sql = self.normalize_sql_column_references(sql.lower(), schema, alias_map)
            
            ast = parse_one(sql.lower())
            try:
                qualify(ast)
            except:
                self.logger.debug("qualify error. Skip qualify.")
                
            alias_map = {}
            self.extract_alias_table_map(ast, alias_map)
            self.get_cols(ast, alias_map, schema, used_columns)
            
        except Exception as e:
            self.logger.error(f"Error parsing SQL for {db_id}: {sql[:100]}... Error: {e}")
            
        return self.convert_to_grouped_schema(used_columns)

    def _process_item(self, item: Dict) -> Dict:
        try:
            db_id = item[self.input_dbid_key]
            db_id_clean = re.sub(r'[^A-Za-z0-9_]', '', str(db_id).replace('\n', ''))
            
            db_info = self.get_schema_for_db(db_id_clean)
            if not db_info:
                self.logger.warning(f"No schema found for database {db_id_clean}")
                return {
                    **item,
                    self.output_raw_schema_key: {},
                    self.output_ddl_key: "",
                    self.output_formatted_schema_key: "",
                    self.output_used_schema_key: {} if hasattr(self, 'input_sql_key') else None
                }
            
            detailed_schema = self.extract_detailed_schema(db_info, db_id_clean)
            
            result = {
                **item,
                self.output_raw_schema_key: detailed_schema,
                self.output_ddl_key: self.generate_ddl_from_schema(detailed_schema),
                self.output_formatted_schema_key: self.generate_formatted_schema(detailed_schema)
            }
            
            if hasattr(self, 'input_sql_key') and self.input_sql_key in item:
                sql = item[self.input_sql_key]
                if sql and sql.strip():
                    used_schema = self.extract_used_schema_from_sql(sql, db_id_clean)
                    result[self.output_used_schema_key] = used_schema
                else:
                    result[self.output_used_schema_key] = {}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing item: {e}")
            error_result = {
                **item,
                self.output_raw_schema_key: {},
                self.output_ddl_key: "",
                self.output_formatted_schema_key: "",
                '_error': str(e)
            }
            if hasattr(self, 'input_sql_key'):
                error_result[self.output_used_schema_key] = {}
            return error_result

    def run(self, storage: DataFlowStorage,
            input_dbid_key: str = "db_id",
            input_sql_key: str = "", 
            output_raw_schema_key: str = "raw_schema",
            output_ddl_key: str = "ddl",
            output_formatted_schema_key: str = "formatted_schema",
            output_used_schema_key: str = "used_schema"
        ):
        
        self.input_dbid_key = input_dbid_key
        self.input_sql_key = input_sql_key
        self.output_raw_schema_key = output_raw_schema_key
        self.output_ddl_key = output_ddl_key
        self.output_formatted_schema_key = output_formatted_schema_key
        self.output_used_schema_key = output_used_schema_key

        self.load_schema_info()
        
        dataframe = storage.read("dataframe")
        
        required_cols = [input_dbid_key]
        required_cols.append(input_sql_key)
            
        missing_cols = [col for col in required_cols if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        items = dataframe.to_dict(orient='records')
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_item, item): idx 
                for idx, item in enumerate(tqdm(items, desc="Submitting tasks"))
            }
                
            results = [None] * len(items)
                
            with tqdm(total=len(items), desc="Processing") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        self.logger.error(f"Error processing index={idx}: {e}")
                        results[idx] = items[idx].copy()
                        results[idx]['_error'] = str(e)
                            
                    pbar.update(1)
            
            results = [r for r in results if r is not None]
        
        output_keys = [
            self.output_raw_schema_key,
            self.output_ddl_key,
            self.output_formatted_schema_key
        ]
        
        output_file = storage.write(pd.DataFrame(results))
        self.logger.info(f"Schema processing completed, saved to {output_file}")

        return output_keys