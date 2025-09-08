import random, time,tqdm,os,gzip,json,datetime
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from tqdm import trange, tqdm
from PIL import Image, ImageChops
from myml import Network, SIGMOID, TANH
import matplotlib.pyplot as plt



DATASET_PATH = Path(r"H:\ESGI\L3\S2\PA2\dataset_1")
HIDDEN_LAYERS       = [16]                      # nombre couche cachée et neurones par couche
LR           = 0.01                                 # learning rate
ACT          = "sigmoid"                            # "tanh" ou "sigmoid ou relu"
EPOCHS       = 120                                  # nb total d'epochs
BLOCK        = 10                                   # nb d'epochs par appel Rust <=> Python
LOG_EVERY    = 5                                    # fréquence d'affichage des métriques
SEED         = 42

CLASSES = ["owl", "squirrel", "fox"]
EXTS = {".jpg", ".jpeg", ".png"}
SIZE=(24,24)


# Sauvegarde de notre modèle
SAVE_DIR = Path(r"H:\ESGI\L3\S2\PA2\trained_models"); 

def save_model_json_gz(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def save_state_tuple(net, path: Path):
    state = net.get_state()
    save_model_json_gz(state, path)

#Défini nos 3 classes 
def one_hot(i):
    v=[0.0]*len(CLASSES)
    v[i]=1.0
    return v

def image_to_vec(path:Path):
    image = Image.open(path).convert("L").resize(SIZE)

    vect= []
    for px in image.getdata():
        vect.append(px/255.0)
    if ACT.lower() == "tanh":
        vect = [2*v - 1 for v in vect]
    return vect



def load_split(path: Path):
    images = []
    labels = []
    for i, cls in enumerate(CLASSES):
        folder = path / cls
        if not folder.exists():
            raise SystemExit(f"Missing class dir: {folder}")
        for f in folder.iterdir(): 
            if f.is_file() and f.suffix.lower() in EXTS:
                images.append(image_to_vec(f))
                labels.append(one_hot(i))
    if not images:
        raise SystemExit(f"No image found in {path}")
    return images, labels

def best_score(v):
    for i in range(len(v)):
        if v[i]==max(v):
            return i

def accuracy(network:Network, images,labels):
    n = len(images)
    if n == 0:
        return 0.0
    correct = 0
    for feat, lab in zip(images, labels):
        pred = network.feed_forward(feat)  
        if best_score(pred) == best_score(lab):
            correct += 1
    return correct / n

def mse_eval(network:Network, images, labels):
    sum_sq_err=0.0
    for img,lab in zip(images,labels):
        output=network.feed_forward(img)
        for out,targ in zip(output,lab):
            d=targ-out; sum_sq_err+=d*d
    return sum_sq_err/(len(images)*len(labels[0])) 

def act_const(name:str)->int:
    if name.lower()=="sigmoid": 
        return 0
    if name.lower()=="tanh": 
        return 1
    if name.lower()=="relu":
        return 2
    raise ValueError("activation function must be one of: 0-sigmoid, 1-tanh")



def run():
    random.seed(SEED)
    train_folder = DATASET_PATH / "train"
    val_folder   = DATASET_PATH / "val"
    images_train, labels_train = load_split(train_folder)
    images_val, labels_val = load_split(val_folder)

    input_dimension = len(images_train[0])
    layers = [input_dimension] + HIDDEN_LAYERS + [len(CLASSES)]
    net = Network(layers, LR, act_const(ACT))
    print(f"input_dimension={input_dimension} | hidden={HIDDEN_LAYERS} | lr={LR} | "
          f"act={ACT} | epochs={EPOCHS} | block={BLOCK} ")

    # Historiques
    mse_tr_hist = []           # MSE(train) par epoch (continu)
    acc_epochs  = []           # epochs où on logge val
    mse_val_hist = []          # MSE(val) aux points de log
    acc_tr_hist  = []          # acc(train) aux points de log
    acc_val_hist = []          # acc(val)   aux points de log
    # Pour sauvegarder le meilleur modèle    
    best_acc = -1.0
    best_mse = float('inf')
    best_epoch = -1

    metric = "mse"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = SAVE_DIR / f"best_{metric}_{stamp}.json.gz"
    # Entraînement
    done = 0
    for start in tqdm(range(0, EPOCHS, BLOCK), desc="Training", leave=True):
        # shuffle train
        idx = list(range(len(images_train))); random.shuffle(idx)
        images_train = [images_train[i] for i in idx];  labels_train = [labels_train[i] for i in idx]

        mini = min(BLOCK, EPOCHS - start)
        block_hist = net.train(images_train, labels_train, mini)
        mse_tr_hist.extend(block_hist)
        done += mini

        epochs_done = start + min(BLOCK, EPOCHS - start)
        if (done % LOG_EVERY) == 0 or done == EPOCHS:
            acc_tr  = accuracy(net, images_train, labels_train)
            acc_val = accuracy(net, images_val, labels_val)
            mse_val = mse_eval(net, images_val, labels_val)

            acc_epochs.append(done)
            mse_val_hist.append(mse_val)
            acc_tr_hist.append(acc_tr)
            acc_val_hist.append(acc_val)

            tqdm.write(
                f"Epoch {done}/{EPOCHS} - "
                f"MSE(tr)={mse_tr_hist[-1]:.4e} - MSE(val)={mse_val:.4e} - "
                f"acc(tr)={acc_tr:.3f} - acc(val)={acc_val:.3f}"
            )
    
            improved = False
            if metric == "acc":
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_mse = mse_val
                    best_epoch = epochs_done
                    improved = True
            else:
                if mse_val < best_mse:
                    best_mse = mse_val
                    best_acc = acc_val
                    best_epoch = epochs_done
                    improved = True

            if improved:
                save_state_tuple(net, best_model_path)
                tqdm.write(f"Saved best ({metric}) at epoch {best_epoch} -> {best_model_path.name}")


    print("Hyperparameters:",
          f"hidden={HIDDEN_LAYERS} | lr={LR} | act={ACT} | epochs={EPOCHS} | block={BLOCK} ")
    print(f"Saved best ({metric}) at epoch {best_epoch} -> {best_model_path.name}")
    print(f"\nFinal → MSE(tr)={mse_tr_hist[-1]:.4e} "
          f"\n MSE(val)={mse_eval(net, images_val, labels_val):.4e} "
          f"\n acc(tr)={accuracy(net, images_train, labels_train):.3f} "
          f"\n acc(val)={accuracy(net, images_val, labels_val):.3f}")

    #MSE
    plt.figure()
    plt.plot(range(1, len(mse_tr_hist)+1), mse_tr_hist, label="MSE train")
    plt.plot(acc_epochs, mse_val_hist, marker="o", linestyle="--", label="MSE val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("MSE (train & val)")
    plt.legend(); plt.grid(True, alpha=0.3)

    #Accuracy
    plt.figure()
    plt.plot(acc_epochs, acc_tr_hist, marker="o", label="acc train")
    plt.plot(acc_epochs, acc_val_hist, marker="o", label="acc val")
    plt.xlabel("epoch"); plt.ylabel("Accuracy"); plt.ylim(0, 1)
    plt.title("Accuracy (train & val)")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.show()


if __name__=="__main__":
    run()

 