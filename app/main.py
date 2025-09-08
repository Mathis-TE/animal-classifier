# app/main.py
from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io, math, logging
import numpy as np
from PIL import Image
from .load_json_gz import load_state_json_gz

logging.basicConfig(level=logging.INFO)

LABELS = ["ecureuil", "hibou", "renard"]

# === ÉTAT GLOBAL ===
NET = None
CURRENT_MODEL_PATH: Path | None = None

# === CHEMIN EN DUR (mets ton fichier ici) =========================
MODEL_PATH = Path(r"H:\ESGI\L3\S2\PA2\trained_models\best_mse_20250905_024315.json.gz")
# ==================================================================

def _sqrt_int(n: int) -> int | None:
    r = int(round(math.sqrt(n)))
    return r if r*r == n else None

def deduce_shape_from_input(n0: int) -> tuple[int, int, int]:
    # Déduit (H,W,C) depuis la taille d'entrée
    if n0 % 3 == 0:
        s3 = _sqrt_int(n0 // 3)
        if s3 is not None: return (s3, s3, 3)
    s1 = _sqrt_int(n0)
    if s1 is not None: return (s1, s1, 1)
    for s in (32, 28, 64):
        if s*s*3 == n0: return (s, s, 3)
        if s*s    == n0: return (s, s, 1)
    return (32, 32, 3) if abs(n0 - 3072) < abs(n0 - 1024) else (32, 32, 1)

def decode_to_vec_auto(b: bytes, n0_expected: int) -> list[float]:
    H, W, C = deduce_shape_from_input(n0_expected)
    im = Image.open(io.BytesIO(b))
    if C == 1:
        im = im.convert("L").resize((W, H))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    else:
        im = im.convert("RGB").resize((W, H))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    x = arr.ravel().tolist()
    if len(x) != n0_expected:
        raise ValueError(f"Prétraitement={H}x{W}x{C} -> {len(x)} features, attendu {n0_expected}.")
    logging.info(f"[MyML] preprocess -> {H}x{W}x{C} ({len(x)})")
    return x

def load_model_at(path: Path):
    """Charge un modèle à l'emplacement donné (chemin absolu)."""
    global NET, CURRENT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable : {path}")
    NET = load_state_json_gz(path)
    CURRENT_MODEL_PATH = path
    logging.info(f"[MyML] Modèle chargé : {path} | arch0={NET.layers()[0]}")

app = FastAPI()

# CORS pour Live Server (:5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def warmup():
    load_model_at(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True, "path": str(CURRENT_MODEL_PATH) if CURRENT_MODEL_PATH else None,
            "arch0": NET.layers()[0], "labels": LABELS}

@app.get("/model")
def model_meta():
    return {"type": "MLP", "arch": NET.layers(),
            "activation_id": NET.activation_id(),
            "labels": LABELS, "path": str(CURRENT_MODEL_PATH)}

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        data = await file.read()
        arch0 = NET.layers()[0]
        x = decode_to_vec_auto(data, arch0)
        probs = NET.feed_forward(x)
        k = int(np.argmax(probs))
        return {"klass": LABELS[k], "probs": probs, "arch": NET.layers()}
    except Exception as e:
        logging.exception("Erreur /predict")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
