cd myml
python -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    
./.venv/Script/activate
pip install pillow maturin fastapi numpy uvicorn
maturin develop --release
cd ../
uvicorn app.main:app --reloead --port 8000          

