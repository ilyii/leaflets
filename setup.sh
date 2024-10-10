py -V:3.12 -m venv .venv
.venv\Scripts\activate
pip install uv
uv pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv sync
