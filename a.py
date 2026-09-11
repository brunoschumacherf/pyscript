import time
import requests

URL = "https://juntosefaz.com.br/nikolas"

while True:
    try:
        response = requests.get(URL, timeout=15)
        print(f"GET {URL} -> {response.status_code}")
    except Exception as e:
        print(f"Erro: {e}")

    time.sleep(180)  # 3 minutos