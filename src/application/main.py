import torch
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import sqlite3
import os
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime
import uuid
from tqdm import tqdm

from ultralytics import YOLO
# from pdf2image import convert_from_path
import fitz

from PIL import Image, ImageDraw  # Ensure ImageDraw is imported
from deal_processing import process_image  # import process_image from deal_processing.py

SEGMENTATION_CACHE = {}

load_dotenv(override=True)

PROJECT_DIR = os.getenv("PROJECT_DIR", ".")
DB_PATH = os.path.join(PROJECT_DIR, "supermarket_leaflets.db")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DEALS_DIR = os.path.join(PROJECT_DIR, "deals")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(
    "/onedrive-images",
    StaticFiles(directory=DEALS_DIR),
    name="onedrive-images"
)
YOLOV11_PATH = os.path.join(MODELS_DIR, "model.pt")
model = YOLO(YOLOV11_PATH, verbose=False)

all_images = {}

for root, dirs, files in tqdm(os.walk(DEALS_DIR), desc="Loading image paths"):
    for file in files:
        if file.endswith(".png") and "annotated" not in file:
            all_images[file] = f"/onedrive-images{root.replace(DEALS_DIR, '')}/{file}"

templates = Jinja2Templates(directory="templates")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # return a list of all supermarket names to filter by
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM supermarket")
    supermarkets = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Deals Übersicht",
        "supermarkets": [supermarket["name"] for supermarket in supermarkets]
    })


@app.get("/api/deals")
async def get_deals(
    page: int = 1,
    search: Optional[str] = None,
    supermarket: Optional[str] = None,
    valid_only: bool = False,
    metadata_only: bool = False,
    sort_by_date: bool = False
):
    clear_tmp_folder()
    clear_cache()
    conn = get_db_connection()
    cursor = conn.cursor()

    base_query = """
    SELECT
        d.*,
        s.name AS supermarket_name,
        l.valid_from_date,
        l.valid_to_date
    FROM deals d
    JOIN leaflet l ON d.leaflet_id = l.leaflet_id
    JOIN supermarket s ON l.supermarket_id = s.supermarket_id
    WHERE 1=1
    """

    params = []

    if search:
        base_query += " AND d.clean_title LIKE ?"
        params.append(f"%{search}%")

    if supermarket:
        base_query += " AND s.name = ?"
        params.append(supermarket)

    if valid_only:
        today = datetime.now().date().isoformat()
        base_query += " AND l.valid_from_date <= ? AND l.valid_to_date >= ?"
        params.append(today)
        params.append(today)

    if metadata_only:
        base_query += """
        AND (d.clean_title != '' OR d.price != '' OR d.price_old != '' OR d.description != '')
        """

    if sort_by_date:
        base_query += " ORDER BY l.valid_from_date DESC"
    else:
        base_query += " ORDER BY d.rowid DESC"

    limit = 15
    offset = (page - 1) * limit
    base_query += f" LIMIT {limit} OFFSET {offset}"

    cursor.execute(base_query, params)
    deals = cursor.fetchall()
    conn.close()

    ret_dict = {"deals": [dict(deal) for deal in deals]}
    # edit img_name to full path
    for deal in ret_dict["deals"]:
        deal["img_name"] = all_images.get(deal["img_name"], None)
    return ret_dict

# -------------------------------------------------------------------

@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": "Prospekt Upload"
    })

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. save pdf
    pdf_contents = await file.read()
    filename = file.filename
    pdf_path = os.path.join("tmp", filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_contents)

    # 2. convert pdf to images
    # Pdf2image
    # pages = convert_from_path(pdf_path)

    # PyMuPDF
    # zoom_x = 2.0  # horizontal zoom
    # zoom_y = 2.0  # vertical zoom
    # mat = pymupdf.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
    # pix = page.get_pixmap(matrix=mat)  # use 'mat' instead of the identity matrix
    pages = []
    doc = fitz.open(pdf_path)
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    # new unique session_id for this pdf
    session_id = str(uuid.uuid4())
    SEGMENTATION_CACHE[session_id] = []

    # 3. process each page with YOLO and (optionally later) to structured text...
    for idx, page_img in tqdm(enumerate(pages), desc="Processing PDF pages"):
        image_filename = f"{session_id}_page_{idx+1}.png"
        image_path = os.path.join("tmp", image_filename)
        page_img.save(image_path, "PNG")

        results_list = model(
            page_img,
            iou=0.4,
            conf=0.5,
            # half=True,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            verbose=False
        )

        # batch size = 1, so we only have one result
        results = results_list[0]
        polygons_data = []

        if results.masks is not None:
            # list of arrays of polygons with shape (N, 2)
            mask_polygons = results.masks.xy
            # per polygon class id and confidence so each (N, 1)
            class_ids = results.boxes.cls.tolist()
            confs = results.boxes.conf.tolist()

            for i, polygon_array in enumerate(mask_polygons):
                polygon_list = polygon_array.tolist()

                cls_id = int(class_ids[i])
                conf = float(confs[i])
                label = results.names[cls_id] if cls_id in results.names else f"class_{cls_id}"

                polygons_data.append({
                    "label": label,
                    "polygon": polygon_list,
                    "meta": {"confidence": conf}
                })

        # 4. save results to session cache
        SEGMENTATION_CACHE[session_id].append({
            "page_image": image_filename,
            "polygons": polygons_data,
            "original_size": [page_img.width, page_img.height]
        })

    # 5. return session_id to upload page t
    return {"session_id": session_id, "message": "PDF verarbeitet"}

@app.post("/api/process_deal")
async def process_deal(request: Request):
    """
    Processes the deal extraction by:
    1. Cropping the deal area using the polygon mask from the page image.
    2. Using process_image to extract structured information.
    Expect JSON:
    {
        "session_id": "...",
        "page_index": 0,
        "polygon_index": 0
    }
    """
    data = await request.json()
    session_id = data.get("session_id")
    page_index = data.get("page_index")
    polygon_index = data.get("polygon_index")

    if session_id not in SEGMENTATION_CACHE:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    pages = SEGMENTATION_CACHE[session_id]
    try:
        page_data = pages[page_index]
    except IndexError:
        return JSONResponse({"error": "Invalid page index"}, status_code=400)

    polygons = page_data.get("polygons", [])
    try:
        poly_data = polygons[polygon_index]
    except IndexError:
        return JSONResponse({"error": "Invalid polygon index"}, status_code=400)

    # Load the page image from the tmp directory
    image_filename = page_data.get("page_image")
    image_path = os.path.join("tmp", image_filename)
    if not os.path.exists(image_path):
        return JSONResponse({"error": "Page image not found"}, status_code=404)

    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        return JSONResponse({"error": "Could not open image"}, status_code=500)

    # Get the polygon points
    polygon = poly_data.get("polygon", [])
    if not polygon:
        return JSONResponse({"error": "Polygon data missing"}, status_code=400)

    # Compute bounding box from polygon points
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x, min_y = int(min(xs)), int(min(ys))
    max_x, max_y = int(max(xs)), int(max(ys))

    # Crop the image to the bounding box region
    cropped = image.crop((min_x, min_y, max_x, max_y))

    # Create a mask image for the polygon (relative to the cropped region)
    width, height = cropped.size
    mask = Image.new("L", (width, height), 0)
    polygon_relative = [(x - min_x, y - min_y) for (x, y) in polygon]
    ImageDraw.Draw(mask).polygon(polygon_relative, fill=255)

    # Apply the mask to the cropped image
    result = Image.new("RGBA", cropped.size)
    result.paste(cropped, (0, 0), mask)

    # Save the resulting cropped image with polygon mask
    crop_filename = f"{session_id}_page_{page_index}_deal_{polygon_index}.png"
    crop_path = os.path.join("tmp", crop_filename)
    result.save(crop_path, "PNG")

    # Use process_image to extract structured deal info from the cropped image
    # deal_info = process_image(crop_path, model="llama3.2-vision")
    # deal_info = process_image(crop_path, model="llama3.2-vision:11b-instruct-q8_0")
    deal_info = process_image(crop_path, model="minicpm-v:latest")
    # deal_info = {}
    extracted_info = f"""
    Brand: {deal_info.get("brand", "unknown")}<br>
    Product Name: {deal_info.get("productname", "unknown")}<br>
    Original Price: {deal_info.get("original_price", "unknown")}<br>
    Deal Price: {deal_info.get("deal_price", "unknown")}<br>
    Amount: {deal_info.get("weight", "unknown")}<br>
    """.strip()

    return JSONResponse({
        "crop_image": crop_filename,
        "extracted_info": extracted_info
    })

@app.get("/preview", response_class=HTMLResponse)
def preview_page(request: Request, session_id: str):
    if session_id not in SEGMENTATION_CACHE:
        return HTMLResponse(content="Session not found", status_code=404)
    return templates.TemplateResponse("preview.html", {
        "request": request,
        "title": "Vorschau",
        "session_id": session_id
    })

@app.get("/api/preview_data")
def get_preview_data(session_id: str):
    if session_id not in SEGMENTATION_CACHE:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {"pages": SEGMENTATION_CACHE[session_id]}

os.makedirs("tmp", exist_ok=True)
app.mount("/tmp", StaticFiles(directory="tmp"), name="tmp")

def clear_tmp_folder():
    for root, dirs, files in os.walk("tmp"):
        for file in files:
            os.remove(os.path.join(root, file))

def clear_cache():
    SEGMENTATION_CACHE.clear()

clear_tmp_folder()
clear_cache()
