from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, ForeignKey
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
    deals = relationship("Deal", back_populates="supermarket", cascade="all, delete-orphan")

class Leaflet(Base):
    __tablename__ = 'leaflet'

    leaflet_id = Column(Integer, primary_key=True)
    supermarket_id = Column(Integer, ForeignKey('supermarket.supermarket_id'), nullable=False)
    num_pages = Column(Integer)
    downloaded_pages = Column(Integer)
    crawl_date = Column(DateTime, default=datetime.utcnow)
    valid_from_date = Column(Date)
    valid_to_date = Column(Date)
    url = Column(String)

    # Relationships
    supermarket = relationship("Supermarket", back_populates="leaflets")
    deals = relationship("Deal", back_populates="leaflet", cascade="all, delete-orphan")

class Deal(Base):
    __tablename__ = 'deals'

    id = Column(Integer, primary_key=True)
    leaflet_id = Column(Integer, ForeignKey('leaflet.leaflet_id'), nullable=False)
    supermarket_id = Column(Integer, ForeignKey('supermarket.supermarket_id'), nullable=False)
    page_num = Column(Integer)
    title = Column(String)
    clean_title = Column(String)
    description = Column(String)
    price = Column(Float)
    price_old = Column(Float)
    discount = Column(Float)
    category = Column(String)
    img_name = Column(String)
    orig_img_size = Column(String)  # Stored as "width,height"
    deal_img_size = Column(String)  # Stored as "width,height"
    polygon_points_abs = Column(String)  # Stored as JSON string
    polygon_points_rel = Column(String)  # Stored as JSON string
    bbox_points_abs = Column(String)    # Stored as JSON string
    bbox_points_rel = Column(String)    # Stored as JSON string
    polygon_conf = Column(Float)
    category_conf = Column(Float)

    # Relationships
    leaflet = relationship("Leaflet", back_populates="deals")
    supermarket = relationship("Supermarket", back_populates="deals")

# Database initialization function
def init_db(db_path='sqlite:///deals.db'):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine