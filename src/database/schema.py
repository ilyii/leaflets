from sqlalchemy import create_engine, Column, Integer, Boolean, String, Float, Date, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Supermarket(Base):
    __tablename__ = 'supermarket'

    supermarket_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    last_update = Column(DateTime, default=datetime.utcnow)

    # Relationships
    leaflets = relationship("Leaflet", back_populates="supermarket", cascade="all, delete-orphan")

class Leaflet(Base):
    __tablename__ = 'leaflet'

    leaflet_id = Column(Integer, primary_key=True)
    supermarket_id = Column(Integer, ForeignKey('supermarket.supermarket_id'), nullable=False)
    supermarket_leaflet_name = Column(String, nullable=False)
    num_pages = Column(Integer)
    downloaded_pages = Column(Integer)
    crawl_date = Column(DateTime, default=datetime.utcnow)
    valid_from_date = Column(Date)
    valid_to_date = Column(Date)
    url = Column(String)
    processed = Column(Boolean, default=False)

    # Relationships
    supermarket = relationship("Supermarket", back_populates="leaflets")
    deals = relationship("Deal", back_populates="leaflet", cascade="all, delete-orphan")

class Deal(Base):
    __tablename__ = 'deals'

    id = Column(Integer, primary_key=True)
    leaflet_id = Column(Integer, ForeignKey('leaflet.leaflet_id'), nullable=False)

    # Attributes from polygon extraction
    page_num = Column(Integer)
    deal_category = Column(String)
    img_name = Column(String, nullable=True)
    orig_img_size = Column(String)  # Stored as "width,height"
    deal_img_size = Column(String)  # Stored as "width,height"
    polygon_points_abs = Column(String)  # Stored as JSON string
    polygon_points_rel = Column(String)  # Stored as JSON string
    bbox_points_abs = Column(String)    # Stored as JSON string
    bbox_points_rel = Column(String)    # Stored as JSON string
    polygon_conf = Column(Float)


    # Attributes from ocr
    title = Column(String, nullable=True)
    clean_title = Column(String, nullable=True)
    description = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    price_old = Column(Float, nullable=True)
    discount = Column(Float, nullable=True)

    # Other attributes
    category = Column(String, nullable=True)

    # Relationships
    leaflet = relationship("Leaflet", back_populates="deals")

# Database initialization function
def init_db(db_path='sqlite:///deals.db'):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine