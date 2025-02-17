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

# YOLO
from ultralytics import YOLO
# PDF -> Images
from pdf2image import convert_from_path

# -------------------------------------------------------------------
# Globale Variablen / "In-Memory-Speicher" für Demo
# -------------------------------------------------------------------
# Wir speichern Segmentationsergebnisse in einem Dictionary:
# { session_id: [ { "page_image": "...", "detections": [ ... ] }, ... ] }
# In einer echten Applikation würdest du vermutlich einen persistenten Speicher verwenden
# oder das Ergebnis nach einer Zeit verwerfen.
SEGMENTATION_CACHE = {}

# -------------------------------------------------------------------
# Projektordner aus .env laden
# -------------------------------------------------------------------
load_dotenv(override=True)

PROJECT_DIR = os.getenv("PROJECT_DIR", ".")
DB_PATH = os.path.join(PROJECT_DIR, "crawled_leaflets", "supermarket_leaflets.db")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DEALS_DIR = os.path.join(PROJECT_DIR, "deals")
YOLOV11_PATH = r"./models/model.pt"

# Erstelle FastAPI-App
app = FastAPI()
model = YOLO(YOLOV11_PATH, verbose=False)

# Static Files (CSS, JS etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(
    "/onedrive-images",
    StaticFiles(directory=DEALS_DIR),
    name="onedrive-images"
)

all_images = {}

for root, dirs, files in tqdm(os.walk(DEALS_DIR), desc="Loading image paths"):
    for file in files:
        if file.endswith(".png") and "annotated" not in file:
            # also replace DEALS_DIR with /onedrive-images/dirs_to_file/file
            all_images[file] = f"/onedrive-images{root.replace(DEALS_DIR, '')}/{file}"

# Templates (Jinja2)
templates = Jinja2Templates(directory="templates")

# SQLite-Verbindung
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------------------------------------------------
# Hauptseite -> index.html
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Rendert die Hauptseite (index.html).
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Deals Übersicht"
    })

# -------------------------------------------------------------------
# Deals-API-Endpunkt -> Filtern, Sortieren, Pagination
# -------------------------------------------------------------------
@app.get("/api/deals")
async def get_deals(
    page: int = 1,
    search: Optional[str] = None,
    supermarket: Optional[str] = None,
    valid_only: bool = False,
    metadata_only: bool = False,
    sort_by_date: bool = False
):
    """
    Liefert Deals als JSON für das Frontend (AJAX-Call).
    """
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

    # Suche
    if search:
        base_query += " AND d.clean_title LIKE ?"
        params.append(f"%{search}%")

    # Filter Supermarkt
    if supermarket:
        base_query += " AND s.name = ?"
        params.append(supermarket)

    # Nur gültige Deals
    if valid_only:
        today = datetime.now().date().isoformat()
        base_query += " AND l.valid_from_date <= ? AND l.valid_to_date >= ?"
        params.append(today)
        params.append(today)

    # Nur Deals mit Metadaten
    if metadata_only:
        base_query += """
        AND (d.clean_title != '' OR d.price != '' OR d.price_old != '' OR d.description != '')
        """

    # Sortierung
    if sort_by_date:
        base_query += " ORDER BY l.valid_from_date DESC"
    else:
        base_query += " ORDER BY d.rowid DESC"

    # Pagination
    limit = 20
    offset = (page - 1) * limit
    base_query += f" LIMIT {limit} OFFSET {offset}"

    cursor.execute(base_query, params)
    deals = cursor.fetchall()
    conn.close()

    ret_dict = {"deals": [dict(deal) for deal in deals]}
    # edit img_name to full path
    for deal in ret_dict["deals"]:
        deal["img_name"] = all_images.get(deal["img_name"], None)
        print(deal["img_name"])
    return ret_dict

# -------------------------------------------------------------------
# Upload-Seite -> upload.html
# -------------------------------------------------------------------
@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    """
    Seite zum Hochladen eines Prospekt-PDFs.
    """
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": "Prospekt Upload"
    })

# -------------------------------------------------------------------
# POST /api/upload_pdf: PDF entgegennehmen, YOLO ausführen (Demo)
# -------------------------------------------------------------------
@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. PDF entgegennehmen
    2. Konvertieren in Bilder
    3. YOLO-Segmentation (Polygone) ausführen
    4. Ergebnisse im In-Memory-Cache speichern
    5. session_id zurückgeben
    """
    # 1. PDF speichern
    pdf_contents = await file.read()
    filename = file.filename
    pdf_path = os.path.join("tmp", filename)
    with open(pdf_path, "wb") as f:
        f.write(pdf_contents)

    # 2. PDF -> einzelne Seiten (Liste von PIL-Images)
    pages = convert_from_path(pdf_path)

    # Neue Session anlegen
    session_id = str(uuid.uuid4())
    SEGMENTATION_CACHE[session_id] = []

    # 3. Jede Seite in PNG speichern + YOLO-Prediction
    for idx, page_img in tqdm(enumerate(pages), desc="Processing PDF pages"):
        image_filename = f"{session_id}_page_{idx+1}.png"
        image_path = os.path.join("tmp", image_filename)
        page_img.save(image_path, "PNG")

        # YOLO auf diese Seite anwenden
        results_list = model(
            image_path,
            # iou=0.75,
            # half=True,
            device="cuda:0",
            verbose=False
        )

        # YOLO v8 gibt ein List[Results],
        # für ein einzelnes Bild ist meist nur results_list[0] relevant
        results = results_list[0]

        # Hier liegen die Polygone (Segmentation) in results.masks.xy
        # - results.masks.xy ist eine Liste von NumPy-Arrays (jeweils Nx2)
        # - results.boxes.cls sind die Klassen-IDs
        # - results.boxes.conf sind die Konfidenzen
        # - results.names ist ein dict ID->Label
        polygons_data = []

        if results.masks is not None:  # Falls YOLO nichts gefunden hat, kann das None sein
            mask_polygons = results.masks.xy  # list of arrays
            class_ids = results.boxes.cls.tolist()  # z.B. [0, 1, 0, ...]
            confs = results.boxes.conf.tolist()     # z.B. [0.95, 0.87, ...]

            for i, polygon_array in enumerate(mask_polygons):
                # polygon_array ist ein Nx2 NumPy-Array
                polygon_list = polygon_array.tolist()

                # Klassendaten
                cls_id = int(class_ids[i])
                conf = float(confs[i])
                label = results.names[cls_id] if cls_id in results.names else f"class_{cls_id}"

                polygons_data.append({
                    "label": label,
                    "polygon": polygon_list,    # Liste von [x, y]
                    "meta": {"confidence": conf}
                })

        # 4. In den In-Memory Cache schreiben
        SEGMENTATION_CACHE[session_id].append({
            "page_image": image_filename,
            "polygons": polygons_data,
            "original_size": [page_img.width, page_img.height]
        })

    # 5. session_id zurückgeben
    return {"session_id": session_id, "message": "PDF verarbeitet"}

# -------------------------------------------------------------------
# GET /preview: Vorschau-Seite (preview.html) mit "Blätterfunktion"
# -------------------------------------------------------------------
@app.get("/preview", response_class=HTMLResponse)
def preview_page(request: Request, session_id: str):
    """
    Zeigt eine Vorschau-Seite (Canvas), wo Polygone angezeigt werden
    """
    if session_id not in SEGMENTATION_CACHE:
        return HTMLResponse(content="Session not found", status_code=404)
    return templates.TemplateResponse("preview.html", {
        "request": request,
        "title": "Vorschau",
        "session_id": session_id
    })

# -------------------------------------------------------------------
# GET /api/preview_data: Liefert die Segmentierungsdaten aus dem Cache
# -------------------------------------------------------------------
@app.get("/api/preview_data")
def get_preview_data(session_id: str):
    """
    Liefert die (page_image, polygons) für die Preview-Ansicht
    """
    if session_id not in SEGMENTATION_CACHE:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {"pages": SEGMENTATION_CACHE[session_id]}

# -------------------------------------
# Zum direkten Ausliefern der Bilder im tmp/ Ordner
# -------------------------------------
app.mount("/tmp", StaticFiles(directory="tmp"), name="tmp")

# clear tmp folder
def clear_tmp_folder():
    for root, dirs, files in os.walk("tmp"):
        for file in files:
            os.remove(os.path.join(root, file))

def clear_cache():
    SEGMENTATION_CACHE.clear()

clear_tmp_folder()  # beim Start aufrufen
clear_cache()  # beim Start aufrufen
