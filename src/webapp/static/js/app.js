let currentPage = 1;

async function fetchDeals(page = 1) {
    currentPage = page;

    const search = document.getElementById("searchInput").value.trim();
    const metadataOnly = document.getElementById("metadataOnly").checked;
    const supermarket = document.getElementById("supermarketSelect").value;
    const validOnly = document.getElementById("validOnly").checked;
    const sortByDate = document.getElementById("sortByDate").checked;

    const params = new URLSearchParams();
    params.append("page", page);
    if (search) params.append("search", search);
    if (supermarket) params.append("supermarket", supermarket);
    if (validOnly) params.append("valid_only", "true");
    if (metadataOnly) params.append("metadata_only", "true");
    if (sortByDate) params.append("sort_by_date", "true");

    const url = "/api/deals?" + params.toString();

    try {
        const res = await fetch(url);
        const data = await res.json();
        renderDeals(data.deals);
    } catch (error) {
        console.error("Error fetching deals:", error);
    }
}

function renderDeals(deals) {
    const container = document.getElementById("dealsContainer");
    container.innerHTML = ""; // Reset

    deals.forEach((deal) => {
        const dealDiv = document.createElement("div");
        dealDiv.classList.add("deal-card");

        // Bild
        const img = document.createElement("img");
        // Du hast in der DB den Pfad, z.B. "img_name" -> hier anpassen:
        img.src = deal.img_name ? deal.img_name : "/static/img/placeholder.png";
        dealDiv.appendChild(img);

        // Supermarkt + Produktname
        const title = document.createElement("h2");
        title.innerText = `${deal.supermarket_name || ""}: ${deal.clean_title || "Ohne Titel"}`;
        dealDiv.appendChild(title);

        // Preis
        const priceInfo = document.createElement("p");
        priceInfo.innerText = `Preis: ${deal.price || "?"} (alt: ${deal.price_old || "?"})`;
        dealDiv.appendChild(priceInfo);

        // Discount
        if (deal.discount) {
            const discount = document.createElement("p");
            discount.innerText = `Discount: ${deal.discount}`;
            dealDiv.appendChild(discount);
        }

        // Beschreibung
        if (deal.description) {
            const desc = document.createElement("p");
            desc.innerText = deal.description;
            dealDiv.appendChild(desc);
        }

        container.appendChild(dealDiv);
    });

    // Update Seitenzähler
    document.getElementById("pageNum").innerText = currentPage;
}

function prevPage() {
    if (currentPage > 1) {
        fetchDeals(currentPage - 1);
    }
}

function nextPage() {
    fetchDeals(currentPage + 1);
}

// Beim Laden der Seite Deals der ersten Seite laden
window.onload = () => {
    fetchDeals(1);
};
