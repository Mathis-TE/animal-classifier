from pathlib import Path
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import random

CLASSES = ["owl","squirrel","fox"]
DATASET_PATH = Path(r"H:\ESGI\L3\S2\PA2\dataset_1")
SIZE = (24,24)
AUG_NOISE_SIGMA = 8.0    # granularité du bruit (0..255)
AUG_NOISE_ALPHA = 0.35   # intensité visuelle (0..1)


def image_to_vec_noisy(path:Path):
    image = Image.open(path).convert("L").resize(SIZE)
    noise = Image.effect_noise(image.size, 50)
    image = ImageChops.add(image, noise, scale=1.0, offset=-128)
    image.show()


if __name__ == "__main__":
    image_to_vec_noisy("H:\ESGI\L3\S2\PA2\dataset_1/train/owl/image-1")  # choisis "owl"/"squirrel"/"fox"
