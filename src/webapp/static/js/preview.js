let pagesData = [];
let currentPageIndex = 0;

let pageImageEl;
let canvasEl;
let ctx;
let highlightedPolygonIndex = -1;

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

    closeModalBtn.onclick = () => { modal.style.display = "none"; };
    window.onclick = (evt) => {
        if (evt.target === modal) {
            modal.style.display = "none";
        }
    };

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

    pageImageEl.onload = () => {
        canvasEl.width = pageImageEl.width;
        canvasEl.height = pageImageEl.height;

        const origWidth = pageData.original_size[0];
        const origHeight = pageData.original_size[1];

        const dispWidth = pageImageEl.width;
        const dispHeight = pageImageEl.height;

        const ratioX = dispWidth / origWidth;
        const ratioY = dispHeight / origHeight;

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

        drawPolygons(scaledPolygons, -1);

        canvasEl.onmousemove = e => handleMouseMove(e, scaledPolygons);
        canvasEl.onclick = e => handleMouseClick(e, scaledPolygons);
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

        if (idx === highlightIndex) {
            ctx.fillStyle = "rgba(255, 0, 0, 0.23)";
            ctx.strokeStyle = "red";
            ctx.lineWidth = 2;
        } else {
            ctx.fillStyle = "rgba(0, 255, 0, 0.15)";
            ctx.strokeStyle = "lime";
            ctx.lineWidth = 1;
        }
        ctx.fill();
        ctx.stroke();
    });
}

function handleMouseMove(evt, polygons) {
    const mousePos = getMousePos(canvasEl, evt);
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
    openModal(polyData);
}

function openModal(polyData) {
    modal.style.display = "block";
    modalTitle.innerText = polyData.label || "Kein Label";
    let text = "";
    for (const [k, v] of Object.entries(polyData.meta || {})) {
        text += `${k}: ${v}<br/>`;
    }
    modalText.innerHTML = text;

    // Check if extracting on demand is enabled
    const extractCheckbox = document.getElementById("extractOnDemand");
    if (extractCheckbox && extractCheckbox.checked) {
        // Call backend to process deal image crop out
        processDealExtraction(sessionId, currentPageIndex, highlightedPolygonIndex);
    } else {
        // Clear any previous crop
        document.getElementById("dealCropContainer").innerHTML = "";
    }
}

async function processDealExtraction(sessionId, pageIndex, polygonIndex) {
    try {
        const res = await fetch("/api/process_deal", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                session_id: sessionId,
                page_index: pageIndex,
                polygon_index: polygonIndex
            })
        });
        const data = await res.json();
        if (data.crop_image) {
            // Display the cropped image and placeholder metadata in the modal
            const dealCropContainer = document.getElementById("dealCropContainer");
            dealCropContainer.innerHTML = `
                <img src="/tmp/${data.crop_image}" alt="Deal Crop" style="max-width:100%; margin-top:1rem;"/>
                <p>${data.extracted_info}</p>
            `;
        } else {
            console.error("No crop image received");
        }
    } catch (error) {
        console.error("Error processing deal extraction:", error);
    }
}

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

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
