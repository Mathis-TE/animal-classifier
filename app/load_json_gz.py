# loader_json_gz.py
from __future__ import annotations
from pathlib import Path
import gzip, json, urllib.request
from typing import Any


from myml import Network

def load_state_json_gz(path: str | Path) -> Network:
    """Charge un Network depuis un .json.gz sauvegardé par save_state_tuple()."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        obj: Any = json.load(f)

    # Ton format actuel = tuple/list de 5 éléments
    if isinstance(obj, list) and len(obj) == 5:
        layers, learning_rate, activation_id, weights, biases = obj

    # Bonus: support d’un format dict (si tu décides d’y passer plus tard)
    elif isinstance(obj, dict):
        layers         = obj["layers"]
        learning_rate  = obj["learning_rate"]
        activation_id  = obj.get("activation_id", 0)
        weights        = obj["weights"]
        biases         = obj["biases"]
    else:
        raise ValueError("Format JSON inconnu (attendu liste[5] ou dict avec clés).")

    # Laisse Network.from_state faire les vérifs de shapes
    return Network.from_state(
        list(map(int, layers)),
        float(learning_rate),
        int(activation_id),
        weights,
        biases,
    )

def fetch_once(url: str, cache: str | Path) -> Path:
    cache = Path(cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache)  # simple et suffisant pour une démo
    return cache
