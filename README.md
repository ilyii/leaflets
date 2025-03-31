<!-- HEADER -->
<h1 align="center">🛒 Leaflets: Supermarket Deal Extraction</h1>

<p align="center">
    <strong>Automated Information Extraction from German Supermarket Deals</strong>  
</p>

<p align="center">
    <!-- <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Project Status" />
    <img src="https://img.shields.io/badge/OCR-Enabled-blue.svg" alt="OCR Enabled" />
    <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License" /> -->
</p>

<!-- AUTHORS -->
<h3 align="center">👨‍💻 Authors</h3>
<p align="center">
    <a href="https://github.com/Gabriel9753"><img src="https://img.shields.io/badge/GitHub-Gabriel-blue?logo=github" alt="Gabriel Schurr" /></a>
    <a href="https://github.com/ilyii"><img src="https://img.shields.io/badge/GitHub-Ilyesse-blue?logo=github" alt="Ilyesse Hettenbach" /></a>
</p>

---

## 📌 Overview

This project is a research initiative at **HKA** focused on automating the extraction of supermarket deals from leaflets using computer vision and OCR techniques. The pipeline consists of multiple stages, including:

- **Crawling**: Automating the retrieval of supermarket leaflets (**src/crawling**)
- **Deal Detection**: Identifying and segmenting deals in leaflets using object detection (**src/deal_detection**)
- **Information Extraction**: Extracting product name, price, discount, and other details from deals (**src/information_extraction**)
- **Deployment**: Providing a web interface for users to interact with extracted data (**src/application**)

### 🚀 Roadmap

- Resolve deals and integrate into a structured database
- Implement price history tracking and comparison features
- Enhance information extraction with advanced OCR and NLP techniques

---

## 🛠️ Installation

To set up the project, follow these steps:

### 1️⃣ Create and Activate Virtual Environment
```bash
uv venv
.venv/Scripts/activate
```

### 2️⃣ Upgrade pip and Sync Dependencies
```bash
python.exe -m pip install --upgrade pip
uv sync
```

### 3️⃣ Run Initial Scripts
```bash
uv run src/scripts/crawling.py
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add the following configuration:
```bash
PROJECT_DIR="path/to/the/project/dir"
```

---

## 📬 Contact
For inquiries or contributions, feel free to open an issue or reach out via GitHub.

📌 **Repository:** [GitHub Link](https://github.com/leaflets)

