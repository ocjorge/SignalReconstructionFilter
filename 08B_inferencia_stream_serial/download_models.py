import kagglehub
import shutil
import os

# descargar modelo
path = kagglehub.model_download(
    "jorgeoc/emg-modelos-v1/pyTorch/v1-myoware-pipeline"
)

print("Modelos descargados en:", path)

# crear carpeta local
os.makedirs("models", exist_ok=True)

# copiar los .pt
for f in os.listdir(path):
    if f.endswith(".pt"):
        src = os.path.join(path, f)
        dst = os.path.join("models", f)
        shutil.copy(src, dst)
        print("Copiado:", dst)
