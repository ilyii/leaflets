from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from schema import Deal
from datetime import datetime, timedelta
import logging

class DatabaseDeleter:
    def __init__(self, config):
        self.config = config
        self.engine = create_engine(config['database']['path'])
        self.Session = sessionmaker(bind=self.engine)

    def delete_old_deals(self):
        session = self.Session()
        try:
            days = self.config['delete_deals']['older_than_days']
            threshold_date = datetime.now() - timedelta(days=days)

            query = session.query(Deal).filter(Deal.crawl_date < threshold_date)

            if self.config['delete_deals']['specific_supermarkets']:
                query = query.filter(
                    Deal.supermarket_id.in_(
                        self.config['delete_deals']['specific_supermarkets']
                    )
                )

            deleted_count = query.delete(synchronize_session=False)
            session.commit()
            logging.info(f"Deleted {deleted_count} old deals")

        finally:
            session.close()