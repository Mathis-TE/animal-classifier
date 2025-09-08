from duckduckgo_search import DDGS
import requests
import os

def download_images(query, folder, max_results=10):
    os.makedirs(folder, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_results)
        for i, result in enumerate(results):
            url = result["image"]
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(f"{folder}/{query}_{i}.jpg", "wb") as f:
                        f.write(response.content)
            except Exception as e:
                print(f"[!] Erreur pour l'image {i}: {e}")
                continue


download_images("fox animal", "dataset-ddg/fox", max_results=10)
download_images("owl", "dataset-ddg/owl", max_results=10)
download_images("squirrel", "dataset-ddg/squirrel", max_results=10)