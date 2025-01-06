from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import pandas as pd
from datetime import datetime
import logging
from utils import load_db_path

class DatabaseQueries:
    def __init__(self, config):
        self.config = config
        self.engine = create_engine(load_db_path())
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

    def _query_db(self, query, values=None):
        session = self.Session()
        if values:
            results = session.execute(text(query), values)
        else:
            results = session.execute(text(query))
        session.close()
        return results

    def update_query(self, query, values):
        session = self.Session()
        session.execute(text(query), values)
        session.commit()
        session.close()
