import os
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

# Directorio de las imágenes
folder_path = 'results/MIP_FINAL_SAGITTAL'

# Obtener la lista de imágenes en el directorio y ordenarlas por número
image_files = sorted([f for f in os.listdir(folder_path) if re.match(r'Projection_\d+\.png', f)])

# Crear una lista de objetos de imagen para la animación
animation_data = []
for file_name in image_files:
    file_path = os.path.join(folder_path, file_name)
    img = Image.open(file_path)
    animation_data.append([plt.imshow(img, animated=True)])

# Crear la animación
fig = plt.figure()
anim = animation.ArtistAnimation(fig, animation_data, interval=250, blit=True)

# Guardar la animación en un archivo GIF
anim.save('results/MIP_FINAL_SAGITTAL/Animation.gif', writer='pillow')  # Requiere la biblioteca Pillow

# Mostrar la animación
plt.show()
