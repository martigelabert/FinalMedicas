

# TODO: https://github.com/pydicom/pydicom/blob/master/examples/image_processing/reslice.py
# TODO: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# DICOM loading and visualization
"""
# Load the segmentation image, and the corresponding CT image with PyDicom. Rearrange
# the image and segmentation ‘pixel array’ given by PyDicom based on the headers. Some
# relevant headers include:
    # – ‘Acquisition Number’.
    # – ‘Slice Index’.
    # – ‘Per-frame Functional Groups Sequence’ Ñ ‘Image Position Patient’.
    # – ‘Segment Identification Sequence’ Ñ ‘Referenced Segment Number’.
"""

import pydicom
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import scipy
from matplotlib.widgets import Slider
import matplotlib

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)
def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)
def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def get_dcm_paths(path):
    def _custom_sort(filename):
        x = int(filename.split('-')[1].split('.')[0])
        return x
    path_names = sorted(os.listdir(path), key=_custom_sort)
    return list(map(lambda filename: os.path.join(path, filename), path_names))

#paths = get_dcm_paths('HCC_005/01-23-1999-NA-ABDPELVIS-36548/2.000000-PRE LIVER-81126')


pelvis01 = ['HCC_005/01-23-1999-NA-ABDPELVIS-36548/2.000000-PRE LIVER-81126',
            'HCC_005/01-23-1999-NA-ABDPELVIS-36548/4.000000-Recon 2 LIVER 3 PHASE AP-90548',
            'HCC_005/01-23-1999-NA-ABDPELVIS-36548/103.000000-LIVER 3 PHASE AP-85837',
            'HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660'
            ]

# USING LAST PHASE
paths = get_dcm_paths(pelvis01[2])

frames = []
slices = []
for i in paths:
    ds = pydicom.dcmread(i)
    # Access the pixel spacing along the x and y axes
    #pixel_spacing = ds.PixelSpacing
    #pixel_spacing_x, pixel_spacing_y = pixel_spacing[0], pixel_spacing[1]

    # Access the slice thickness along the z-axis
    #slice_thickness = ds.SliceThickness

    #pixel_len_mm = [slice_thickness, pixel_spacing_y, pixel_spacing_x]  # Pixel length in mm [z, y, x]
    #print(pixel_len_mm)
    if hasattr(ds, 'SliceLocation'):
        slices.append(ds)
print(f'num of slices is {len(slices)}')
slices = sorted(slices, key=lambda s: s.SliceLocation)

# Crear una figura y un eje para mostrar la imagen
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Mostrar la primera imagen
current_index = 0
current_image = slices[current_index].pixel_array
image_plot = ax.imshow(current_image, cmap=matplotlib.colormaps['bone'])

# Crear un slider para seleccionar la imagen
slider_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
slider = Slider(slider_ax, 'slide_n', 0, len(slices)-1, valinit=current_index, valstep=1)

# Función para actualizar la imagen al cambiar el valor del slider
def update_image(val):
    index = int(val)
    current_image = slices[index].pixel_array
    image_plot.set_data(current_image)
    fig.canvas.draw_idle()

# Conectar la función de actualización al evento del slider
slider.on_changed(update_image)

# Mostrar la figura
plt.show()


# Save the frames as an animated GIF
#frames[0].save('output.gif', format='GIF',
#               append_images=frames[1:],
#               save_all=True,
#               duration=200,  # Duration between frames in milliseconds
#               loop=0)  # Set loop value to 0 for infinite loop

# Acces to atributtes
#ds.AcquisitionNumber
def showSegmentation(path):
    ds = pydicom.dcmread(path)
    frames = []
    print(ds.pixel_array.shape)
    image = []
    for i in ds.pixel_array:
        image.append(i)
        #frames.append(i)
        plt.imshow(i)
        print(i)
        plt.show()

    #plt.imshow(ds.pixel_array)
    #plt.show()

    # Crear una figura y un eje para mostrar la imagen
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Mostrar la primera imagen
    current_index = 0
    current_image = image[current_index]
    image_plot = ax.imshow(current_image, cmap=matplotlib.colormaps['bone'])

    # Crear un slider para seleccionar la imagen
    slider_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider = Slider(slider_ax, 'slide_n', 0, len(image) - 1, valinit=current_index, valstep=1)

    # Función para actualizar la imagen al cambiar el valor del slider
    def update_image(val):
        index = int(val)
        current_image = image[index]
        image_plot.set_data(current_image)
        fig.canvas.draw_idle()

    # Conectar la función de actualización al evento del slider
    slider.on_changed(update_image)

    # Mostrar la figura
    plt.show()

showSegmentation('HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660/1-1.dcm')

