# mini_overfit.py
# Usage:
#   python mini_overfit.py --data D:\ESGI\L3\S2\PA2\dataset\train --size 32 --hidden 32 --lr 0.02 --epochs 3000 --act relu --plot
# Prérequis: pip install pillow (I/O images), et ton module Rust "myml" installé via maturin.

import argparse
import random
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from myml import Network, SIGMOID, TANH  # 2 / 0 / 1

CLASSES = ["owl", "squirrel", "fox"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---------- utils ----------
def one_hot(idx: int, n: int = 3) -> List[float]:
    v = [0.0] * n
    v[idx] = 1.0
    return v

def image_to_vec(path: Path, size: Tuple[int, int]) -> List[float]:
    im = Image.open(path).convert("L").resize(size)
    return [p / 255.0 for p in im.getdata()]  # 0..1

def pick_k_per_class(root: Path, k: int = 1, size: Tuple[int,int]=(32,32), seed: int = 42) -> Tuple[List[List[float]], List[List[float]], List[Path]]:
    random.seed(seed)
    images = []
    labels = []
    paths = []
    for i, cls in enumerate(CLASSES):
        folder = root / cls
        if not folder.exists():
            raise SystemExit(f"Missing class dir: {folder}")
        cls_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
        if len(cls_files) < k:
            raise SystemExit(f"Not enough images in class '{cls}' ({len(cls_files)} found, {k} needed)")
        chosen = random.sample(cls_files, k)
        for f in chosen:
            images.append(image_to_vec(f, size))
            labels.append(one_hot(i))
            paths.append(f)
    return images, labels, paths

def argmax(v: List[float]) -> int:
    return max(range(len(v)), key=lambda i: v[i])

def accuracy_on_set(net: Network, X: List[List[float]], Y: List[List[float]]) -> float:
    ok = 0
    for x, y in zip(X, Y):
        ok += int(argmax(net.feed_forward(x)) == argmax(y))
    return ok / len(X)

# ---------- mapping activation ----------
def act_const(name: str) -> int:
    name = name.lower()
    if name == "sigmoid": return SIGMOID
    if name == "tanh": return TANH
    raise ValueError("act doit être parmi: sigmoid, tanh")

# ---------- routine principale ----------
def run(data_train: Path, size: int, hidden: int, lr: float, act_name: str, epochs: int, seed: int, plot: bool):
    size_xy = (size, size)
    # 1) échantillonner 1 image par classe
    Xtr, Ytr, paths = pick_k_per_class(data_train, 100,size=size_xy, seed=seed,)
    print("Mini-set choisi (1 par classe) :")
    for p in paths:
        print(" -", p)

    in_dim = len(Xtr[0])
    print(f"in_dim={in_dim} (size={size}×{size}) | hidden={hidden} | lr={lr} | act={act_name} | epochs={epochs}")

    # 2) modèle (même activation pour toutes les couches dans ton implémentation)
    net = Network([in_dim, hidden, len(CLASSES)], lr, act_const(act_name))

    # 3) entraînement sur ces 3 images uniquement (mini-overfit)
    hist = net.train(Xtr, Ytr, epochs)

    # 4) résultats
    acc = accuracy_on_set(net, Xtr, Ytr)
    print(f"\nMSE début : {hist[0]:.6e}")
    print(f"MSE fin   : {hist[-1]:.6e}")
    print(f"Accuracy mini-set (3 images) : {acc*100:.1f}%")

    # détails prédictions
    for i, (x, y) in enumerate(zip(Xtr, Ytr)):
        out = net.feed_forward(x)
        print(f"[{CLASSES[argmax(y)]}] pred={CLASSES[argmax(out)]} | scores={[round(s,3) for s in out]}")

    # 5) (optionnel) courbe de loss
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.plot(hist)
            plt.xlabel("epoch")
            plt.ylabel("MSE (train)")
            plt.title("Mini-overfit (1 image par classe)")
            plt.show()
        except Exception as e:
            print("Matplotlib indisponible:", e)

def parse_args():
    ap = argparse.ArgumentParser(description="Mini-overfit myml sur 3 images (1 par classe).")
    ap.add_argument("--data", type=Path, required=True, help="dossier TRAIN contenant owl/squirrel/fox (ex: .../dataset/train)")
    ap.add_argument("--size", type=int, default=32, help="redimension carré (px), ex: 32 => 32×32")
    ap.add_argument("--hidden", type=int, default=32, help="neurones de la couche cachée (petit pour mini-overfit)")
    ap.add_argument("--lr", type=float, default=0.02, help="learning rate (ex: 0.02 pour ReLU)")
    ap.add_argument("--act", choices=["sigmoid","tanh"], default="sigmoid", help="fonction d'activation")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", help="afficher la courbe MSE")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.size, args.hidden, args.lr, args.act, args.epochs, args.seed, args.plot)
