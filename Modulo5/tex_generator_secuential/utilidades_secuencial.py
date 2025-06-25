import os
from datetime import datetime

def guardar_texto_generado(texto, modelo):
    os.makedirs("textos_generados", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{modelo}_{timestamp}.txt"
    ruta = os.path.join("textos_generados", nombre_archivo)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    return nombre_archivo
