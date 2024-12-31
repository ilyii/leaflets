from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime
import logging

class DatabaseQueries:
    def __init__(self, config):
        self.config = config
        self.engine = create_engine(config['database']['path'])
        self.Session = sessionmaker(bind=self.engine)

    def execute_query(self):
        query_type = self.config['run_queries']['query_type']

        if query_type == 'active_deals':
            results = self._get_active_deals()
        elif query_type == 'price_comparison':
            results = self._get_price_comparison()
        elif query_type == 'category_summary':
            results = self._get_category_summary()
        else:
            raise ValueError(f"Unknown query type: {query_type}")

        self._export_results(results)

    def _export_results(self, results):
        export_path = Path(self.config['run_queries']['export_path'])
        export_path.mkdir(exist_ok=True)

        filename = f"{self.config['run_queries']['query_type']}_{datetime.now().strftime('%Y%m%d')}"
        filepath = export_path / f"{filename}.{self.config['run_queries']['export_format']}"

        if self.config['run_queries']['export_format'] == 'csv':
            results.to_csv(filepath, index=False)
        elif self.config['run_queries']['export_format'] == 'json':
            results.to_json(filepath, orient='records')

        logging.info(f"Exported results to {filepath}")