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

def get_dcm_paths(path):
    def _custom_sort(filename):
        x = int(filename.split('-')[1].split('.')[0])
        return x
    path_names = sorted(os.listdir(path), key=_custom_sort)
    return list(map(lambda filename: os.path.join(path, filename), path_names))



if __name__ == '__main__':
    files = get_dcm_paths('Project/RM_Brain_3D-SPGR')
    slices = []
    for f in files:
        slices.append(pydicom.dcmread(f))
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    slides3d = np.array([s.pixel_array for s in slices])

    # the phantom es inmutable
    # osea que voy a tener que transformar los otros (rotarl al segor que mira de perfir y rotarlo para que se
    # ajuste a la rotación del fantom)

    # el que esta rotado es el phantom
