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

    files = get_dcm_paths('Project/RM_Brain_3D-SPGR')
    slices = []
    for f in files:
        pass
        #slices.append(pydicom.dcmread(f))
    #slices = sorted(slices, key=lambda s: s.SliceLocation)
    #slides3d = np.array([s.pixel_array for s in slices])

    atlas = pydicom.dcmread(_atlasPath)
    imgs_atlas = atlas.pixel_array
    #print(imgs_atlas.shape)
    plt.imshow(imgs_atlas[80])
    plt.show()

    voxelView(imgs_atlas)



    # the phantom es inmutable
    # osea que voy a tener que transformar los otros (rotarl al segor que mira de perfir y rotarlo para que se
    # ajuste a la rotación del fantom)

    # el que esta rotado es el phantom
