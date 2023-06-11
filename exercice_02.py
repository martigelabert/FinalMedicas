from ctypes import py_object

import pydicom
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import scipy
import matplotlib
import re
import pydicom_seg
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import cv2
from mpl_toolkits.mplot3d import Axes3D

_atlasPath = 'Project/AAL3_1mm.dcm'
_panthomPath = 'Project/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm'


def voxelView(voxels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels)
    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Voxel Plot')


def detect_landmarks(gray):
    # Crear el detector SIFT
    sift = cv2.SIFT_create()

    # Detectar puntos clave y descriptores en la imagen
    keypoints, _ = sift.detectAndCompute(gray, None)

    # Extraer las coordenadas de los puntos clave
    landmarks = np.array([kp.pt for kp in keypoints])

    return landmarks


def get_amygdala_mask(img_atlas: np.ndarray) -> np.ndarray:
    # Your code here:
    #   ...
    amygdala_mask = np.zeros_like(img_atlas)



    enable = [127, 121,147, 137, 135]
    #127
    #121
    #147
    #137
    #135

    for i in enable:
        amygdala_mask[img_atlas == i] = 1
    #amygdala_mask[img_atlas == 46] = 1
    return amygdala_mask



def get_dcm_paths(path):
    def _custom_sort(filename):
        x = int(filename.split('.')[0])
        return x
    path_names = sorted(os.listdir(path), key=_custom_sort)
    return list(map(lambda filename: os.path.join(path, filename), path_names))

if __name__ == '__main__':

    dc_atlas = pydicom.dcmread(_atlasPath)
    dc_phantom = pydicom.dcmread(_panthomPath)

    files = get_dcm_paths('Project/RM_Brain_3D-SPGR')
    slices = []
    for f in files:
        slices.append(pydicom.dcmread(f))
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    slides3d = np.array([s.pixel_array for s in slices])

    images_atlas = dc_atlas.pixel_array
    # Crop phantom to atlas size
    images_phantom = dc_phantom.pixel_array[6:-6, 6:-6, 6:-6]

    # Si realizas crop no se requiere esta parte
    padded_images_atlas = [np.pad(img_atlas,[(int((images_phantom.shape[0] - images_atlas.shape[0]) / 2),),
                                             (int((images_phantom.shape[1] - images_atlas.shape[1]) / 2),)],mode='constant') for img_atlas in images_atlas]

    padded_images_atlas = np.array(padded_images_atlas)


    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(images_atlas[100,:,:], cmap='bone')
    axs[1].imshow(images_phantom[100, :, :], cmap='tab20')
    fig.show()



    for img_patient in slides3d:

        print(img_patient.shape)
        # Crear figura y subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Subplot 1
        axs[0, 0].imshow(img_patient)
        #axs[0, 0].set_title('Imagen 1')

        # Subplot 2
        axs[0, 1].imshow(img_patient)
        #axs[0, 1].set_title('Imagen 2')

        # Subplot 3
        axs[1, 0].imshow(img_patient)
        #axs[1, 0].set_title('Imagen 3')

        # Subplot 4
        axs[1, 1].imshow(img_patient)
        #axs[1, 1].set_title('Imagen 4')

        # Ajustar espaciado entre subplots
        plt.tight_layout()

        # Ocultar ejes de coordenadas
        for ax in axs.flat:
            ax.axis('off')

        # Mostrar el gráfico
        plt.show()




    #imgs_atlas = atlas.pixel_array
    #print(imgs_atlas.shape)
    # este
    #plt.imshow(imgs_atlas[80])
    #plt.show()

    #voxelView(imgs_atlas)
    #for i in imgs_atlas:
    # atlas
    #padding = [np.pad(img_atlas,[(int())])]





    # the phantom es inmutable
    # osea que voy a tener que transformar los otros (rotarl al segor que mira de perfir y rotarlo para que se
    # ajuste a la rotación del fantom)

    # el que esta rotado es el phantom
