

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

# USING LAST PHASE
paths = get_dcm_paths('HCC_005/01-23-1999-NA-ABDPELVIS-36548/103.000000-LIVER 3 PHASE AP-85837')

frames = []
slices = []
for i in paths:
    ds = pydicom.dcmread(i)
    #print(ds)
    # Access the pixel spacing along the x and y axes
    pixel_spacing = ds.PixelSpacing
    pixel_spacing_x, pixel_spacing_y = pixel_spacing[0], pixel_spacing[1]

    # Access the slice thickness along the z-axis
    slice_thickness = ds.SliceThickness

    pixel_len_mm = [slice_thickness, pixel_spacing_y, pixel_spacing_x]  # Pixel length in mm [z, y, x]
    print(pixel_len_mm)
    if hasattr(ds, 'SliceLocation'):
        slices.append(ds)

    print(f'Acquisition Number {ds.AcquisitionNumber}')
    # Access the Slice Location attribute
    #slice_location = ds.SliceLocation

    # Access the Image Position (Patient) attribute
    #image_position = ds.ImagePositionPatient[2]

    # Print the slice index
    #slice_index = (slice_location - image_position) / ds.SliceThickness
    #print("Slice Index:", slice_index)

    # images
    #frames.append(Image.fromarray(ds.pixel_array))

    #plt.imshow(ds.pixel_array)
    #plt.show()

print(f'num of slices is {len(slices)}')
slices = sorted(slices, key=lambda s: s.SliceLocation)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1]/ps[0]
sag_aspect = ps[1]/ss
cor_aspect = ss/ps[0]

# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

# plot 3 orthogonal slices
a1 = plt.subplot(2, 2, 1)
plt.imshow(img3d[:, :, img_shape[2]//2])
a1.set_aspect(ax_aspect)

a2 = plt.subplot(2, 2, 2)
plt.imshow(img3d[:, img_shape[1]//2, :])
a2.set_aspect(sag_aspect)

a3 = plt.subplot(2, 2, 3)
plt.imshow(img3d[img_shape[0]//2, :, :].T)
a3.set_aspect(cor_aspect)

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
    #for i in ds.pixel_array:
        #frames.append(i)
        #plt.imshow(i)
        #plt.show()

    #plt.imshow(ds.pixel_array)
    #plt.show()

showSegmentation('HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660/1-1.dcm')

