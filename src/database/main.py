# database/main.py
import logging
from pathlib import Path
import yaml
from datetime import datetime

from utils import setup_logging, load_config
from create import DatabaseCreator
from update import DatabaseUpdater
from delete import DatabaseDeleter
from queries import DatabaseQueries
from extract_deals import DealExtractor

def main():
    # Load configuration
    config = load_config('src/database/configs.yaml')
    setup_logging(config['logging'])

    # Get the action to execute
    action = config['action']

    try:
        if action == 'create_database':
            creator = DatabaseCreator(config)
            creator.create_database()

        elif action == 'update_leaflets':
            updater = DatabaseUpdater(config)
            updater.update_leaflets()

        elif action == 'extract_deals':
            extractor = DealExtractor(config)
            extractor.process_leaflets()

        elif action == 'delete_deals':
            deleter = DatabaseDeleter(config)
            deleter.delete_old_deals()

        elif action == 'run_queries':
            queries = DatabaseQueries(config)
            queries.execute_query()

        else:
            raise ValueError(f"Unknown action: {action}")

        logging.info(f"Action '{action}' completed successfully")

    except Exception as e:
        logging.error(f"Error executing action '{action}': {str(e)}")
        raise

if __name__ == "__main__":
    main()