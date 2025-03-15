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

## Overview
- Project work at HKA
- Automating crawling of leaflets (src/crawling)
- Detecting deals using visual object detection (src/deal_detection)
- Extracting information on deal-level (src/information_extraction)
- Deploying as web application (src/application)
- TODO: Resolve deals and map into database
- TODO: Price history and price comparison
- TODO: Advanced information extraction

## Get Started

### Installation

```bash
uv venv
.venv/Scripts/activate
python.exe -m pip install --upgrade pip
uv sync
uv run src/scripts/crawling.py
```

Create .env file in the root directory and add the following:

```bash
PROJECT_DIR="path/to/the/project/dir"
```