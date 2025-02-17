let pagesData = [];
let currentPageIndex = 0;

let pageImageEl;
let canvasEl;
let ctx;
let highlightedPolygonIndex = -1; // Index des Polygons, das gerade gehighlightet ist

let modal;
let modalTitle;
let modalText;
let closeModalBtn;

document.addEventListener("DOMContentLoaded", async () => {
    pageImageEl = document.getElementById("pageImage");
    canvasEl = document.getElementById("overlayCanvas");
    ctx = canvasEl.getContext("2d");

    modal = document.getElementById("modal");
    modalTitle = document.getElementById("modalTitle");
    modalText = document.getElementById("modalText");
    closeModalBtn = document.getElementById("closeModal");

    document.getElementById("prevBtn").addEventListener("click", prevPage);
    document.getElementById("nextBtn").addEventListener("click", nextPage);

    // Modal schließen
    closeModalBtn.onclick = () => { modal.style.display = "none"; };
    window.onclick = (evt) => {
        if (evt.target === modal) {
            modal.style.display = "none";
        }
    };

    // Daten vom Server laden
    await loadPreviewData();
    renderPage();
});

async function loadPreviewData() {
    const url = `/api/preview_data?session_id=${sessionId}`;
    const res = await fetch(url);
    if (!res.ok) {
        alert("Fehler beim Laden der Preview-Daten.");
        return;
    }
    const data = await res.json();
    pagesData = data.pages || [];
    if (pagesData.length === 0) {
        alert("Keine Seiten enthalten.");
    }
    currentPageIndex = 0;
}

function renderPage() {
    const pageData = pagesData[currentPageIndex];
    const imgSrc = `/tmp/${pageData.page_image}`;
    pageImageEl.src = imgSrc;

    // Wenn das Bild geladen ist
    pageImageEl.onload = () => {
        // Canvas anpassen
        canvasEl.width = pageImageEl.width;
        canvasEl.height = pageImageEl.height;

        // Originalgröße laut Server
        const origWidth = pageData.original_size[0];
        const origHeight = pageData.original_size[1];

        // Aktuelle Anzeigegröße
        const dispWidth = pageImageEl.width;
        const dispHeight = pageImageEl.height;

        // Skalierungsfaktoren
        const ratioX = dispWidth / origWidth;
        const ratioY = dispHeight / origHeight;

        // Polygone runterskalieren
        const scaledPolygons = pageData.polygons.map(det => {
            const scaledPoly = det.polygon.map(([x, y]) => {
                return [x * ratioX, y * ratioY];
            });
            return {
                label: det.label,
                polygon: scaledPoly,
                meta: det.meta
            };
        });

        // Zeichnen
        drawPolygons(scaledPolygons, -1);

        // Mouse-Events
        canvasEl.onmousemove = e => handleMouseMove(e, scaledPolygons);
        canvasEl.onclick = e => handleMouseClick(e, scaledPolygons);

        // ...
    };
}

function drawPolygons(polygons, highlightIndex) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    polygons.forEach((item, idx) => {
        const poly = item.polygon;
        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0], poly[i][1]);
        }
        ctx.closePath();

        // Farbe wählen
        if (idx === highlightIndex) {
            ctx.fillStyle = "rgba(255, 0, 0, 0.3)"; // Rote Füllung mit Transparenz
            ctx.strokeStyle = "red";
            ctx.lineWidth = 2;
        } else {
            ctx.fillStyle = "rgba(0, 255, 0, 0.2)"; // Grüne Füllung
            ctx.strokeStyle = "lime";
            ctx.lineWidth = 1;
        }
        ctx.fill();
        ctx.stroke();
    });
}

function handleMouseMove(evt, polygons) {
    const mousePos = getMousePos(canvasEl, evt);
    // Test, ob Maus in Polygon
    const foundIndex = polygons.findIndex((p) =>
        isPointInPolygon(mousePos, p.polygon)
    );
    if (foundIndex !== highlightedPolygonIndex) {
        highlightedPolygonIndex = foundIndex;
        drawPolygons(polygons, highlightedPolygonIndex);
    }
}

function handleMouseClick(evt, polygons) {
    if (highlightedPolygonIndex < 0) return;
    const polyData = polygons[highlightedPolygonIndex];
    // Modal anzeigen
    openModal(polyData);
}

function openModal(polyData) {
    modal.style.display = "block";
    modalTitle.innerText = polyData.label || "Kein Label";
    // Zeige Metadaten an
    let text = "";
    for (const [k, v] of Object.entries(polyData.meta || {})) {
        text += `${k}: ${v}<br/>`;
    }
    modalText.innerHTML = text;
}

/**
 * Hilfsfunktion: Position der Maus relativ zum Canvas ermitteln
 */
function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

/**
 * Point in Polygon (Ray casting oder Winding rule)
 * Hier eine einfache Implementierung
 */
function isPointInPolygon(pt, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        const intersect = ((yi > pt.y) !== (yj > pt.y)) &&
            (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// Blättern
function nextPage() {
    if (currentPageIndex < pagesData.length - 1) {
        currentPageIndex++;
        renderPage();
    }
}
function prevPage() {
    if (currentPageIndex > 0) {
        currentPageIndex--;
        renderPage();
    }
}
