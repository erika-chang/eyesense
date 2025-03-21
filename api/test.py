import requests 

url = "http://127.0.0.1:8000/predict"
image_path = "0_left.jpg"

with open(image_path, "rb") as img_file: 
    files = {"img": img_file}
    response = requests.post(url, files=files)

print(response.json()) # Verifica o resultado