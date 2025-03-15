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

function capitalizeFirstLetter(val) {
    return String(val).charAt(0).toUpperCase() + String(val).slice(1);
}

function renderDeals(deals) {
    const container = document.getElementById("dealsContainer");
    container.innerHTML = "";

    deals.forEach((deal) => {
        const card = document.createElement("div");
        card.classList.add("deal-card");

        const header = document.createElement("div");
        header.classList.add("deal-header");
        header.innerText = capitalizeFirstLetter(deal.supermarket_name) || "Unbekannt";
        card.appendChild(header);

        const imgContainer = document.createElement("div");
        imgContainer.classList.add("deal-image-container");
        const img = document.createElement("img");
        img.src = deal.img_name || "/static/img/placeholder.png";
        imgContainer.appendChild(img);
        card.appendChild(imgContainer);

        const content = document.createElement("div");
        content.classList.add("deal-content");

        const titleEl = document.createElement("div");
        titleEl.classList.add("deal-title");
        titleEl.innerText = deal.clean_title || "Ohne Titel";
        content.appendChild(titleEl);

        const priceWrapper = document.createElement("div");
        if (deal.price_old && deal.price_old > 0) {
            const oldPriceEl = document.createElement("span");
            oldPriceEl.classList.add("deal-old-price");
            oldPriceEl.innerText = `${deal.price_old} €`;
            priceWrapper.appendChild(oldPriceEl);
        }
        if (deal.price) {
            const newPriceEl = document.createElement("span");
            newPriceEl.classList.add("deal-price");
            newPriceEl.innerText = `${deal.price} €`;
            priceWrapper.appendChild(newPriceEl);
        }
        if (deal.discount && deal.discount > 0) {
            const discountEl = document.createElement("span");
            discountEl.classList.add("deal-discount");
            discountEl.innerText = ` (${(deal.discount * 100).toFixed(0)}%)`;
            priceWrapper.appendChild(discountEl);
        }
        content.appendChild(priceWrapper);
        if (deal.description) {
            const descEl = document.createElement("div");
            descEl.classList.add("deal-description");
            descEl.innerText = deal.description;
            content.appendChild(descEl);
        }
        card.appendChild(content);

        // Open modal on click
        card.addEventListener("click", () => {
            openDealModal(deal);
        });

        container.appendChild(card);
    });

    // Change current page in pagination
    document.getElementById("pageNum").innerText = currentPage;
}

function openDealModal(deal) {
    const modal = document.getElementById("dealModal");
    const modalContent = document.getElementById("dealModalContent");

    // Populate modal content; feel free to customize this template structure as needed
    modalContent.innerHTML = `
        <h2>${deal.clean_title || "Ohne Titel"}</h2>
        <p><strong>Supermarkt:</strong> ${capitalizeFirstLetter(deal.supermarket_name) || "Unbekannt"}</p>
        <img src="${deal.img_name || '/static/img/placeholder.png'}" style="max-width:100%; display: block; margin: auto;" alt="Deal Image" />
        <p>
          ${deal.description || ""}
        </p>
        <p>
            <strong>Preis:</strong> ${deal.price ? deal.price + " €" : "Keine Angabe"}
            ${deal.price_old && deal.price_old > 0 ? `<span style="text-decoration: line-through; color:gray;"> ${deal.price_old} €</span>` : ""}
            ${deal.discount && deal.discount > 0 ? `<span> (${(deal.discount * 100).toFixed(0)}% Rabatt)</span>` : ""}
        </p>
    `;

    modal.style.display = "block";
}

// Close modal functionality
document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("dealModal");
    const closeBtn = document.getElementById("closeDealModal");

    closeBtn.addEventListener("click", () => {
        modal.style.display = "none";
    });

    window.addEventListener("click", (event) => {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });
});

function prevPage() {
    if (currentPage > 1) {
        fetchDeals(currentPage - 1);
    }
}

function nextPage() {
    fetchDeals(currentPage + 1);
}

// load data on page load
window.onload = () => {
    fetchDeals(1);
};
