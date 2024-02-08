import numpy as np
from PIL import Image
import re

# Pfad zur Datei mit den RGB-Werten
file_path = 'picture.txt'

# Einlesen der Datei
with open(file_path, 'r') as file:
    content = file.read()

# Extrahieren der RGB-Daten aus dem Text mithilfe einer regulären Expression
rgb_values = re.findall(r'\d+', content)

# Konvertierung der extrahierten Daten in ein NumPy-Array vom Typ uint8
rgb_array = np.array(rgb_values, dtype=np.uint8)

# Bildgröße
height, width = 200, 200
expected_size = height * width * 3  # 3 Kanäle (R, G, B) pro Pixel

# Kürzen oder Ergänzen des Arrays, um es an die Größe 200x200 anzupassen
if len(rgb_array) > expected_size:
    # Kürzen des Arrays, wenn es zu lang ist
    rgb_array = rgb_array[:expected_size]
elif len(rgb_array) < expected_size:
    # Ergänzen des Arrays mit schwarzen Pixeln (0, 0, 0), wenn es zu kurz ist
    rgb_array = np.pad(rgb_array, (0, expected_size - len(rgb_array)), mode='constant', constant_values=0)

# Umformen des Arrays in (Höhe, Breite, Farbkanäle)
rgb_array = rgb_array.reshape((height, width, 3))

# Erstellen des Bildes aus dem NumPy-Array
image = Image.fromarray(rgb_array, 'RGB')

# Speichern des Bildes
output_image_path = '/mnt/data/output_image.png'
image.save(output_image_path)
