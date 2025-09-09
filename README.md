cd myml <br>
python -m venv .venv <br>
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    <br>
./.venv/Script/activate <br>
pip install pillow maturin fastapi numpy uvicorn python-multipart<br>
maturin develop --release <br>
cd ../ <br>
uvicorn app.main:app --reload --port 8000  <br>

