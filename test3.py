
# watch https://www.youtube.com/watch?v=0yy83vTnx-4
# TODO: https://github.com/pydicom/pydicom/blob/master/examples/image_processing/reslice.py
# TODO: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# TODO: https://www.youtube.com/watch?v=lhcJd3uKE2k
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
    #print(pixel_len_mm)
    if hasattr(ds, 'SliceLocation'):
        slices.append(ds)

    #print(f'Acquisition Number {ds.AcquisitionNumber}')
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

slides3d =  np.array([s.pixel_array for s in slices])
print(f'slides3d con tamaño {slides3d.shape}')

ds = pydicom.dcmread('HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660/1-1.dcm')
segmentation = []
print(f'mask {ds}')
for i in ds.pixel_array:
    segmentation.append(i)
segmentation = np.array(segmentation)
if False:
    masked = []
    for index,s in enumerate(slides3d):
        rgb_image = plt.cm.gray(s)
        alpha = 0.2
        mask = segmentation[index]
        masked_overlay = np.copy(rgb_image)

        for i in range(len(masked_overlay)):
            for j in range(masked_overlay.shape[1]):
                if mask[i][j] == 1:
                    masked_overlay[i][j][0]= int(100) # Red channel

        #masked_overlay[:, :, mask]= 255
        #masked_overlay[..., 1:3] *= alpha  # Green and blue channels
        #masked.append(Image.fromarray((masked_overlay * 255).astype(np.uint8)))
        #masked.append(Image.fromarray(masked_overlay))

        #masked.append((masked_overlay * 255).astype(np.uint8))

        #plt.imshow(masked_overlay)
        #plt.show()

        masked = np.array(masked)

###################################################################33
# Create projections varying the angle of rotation
#   Configure visualization colormap
#img_dcm = slides3d
slides3d[segmentation[0:89]] = 255
img_dcm = slides3d
img_min = np.amin(img_dcm)
img_max = np.amax(img_dcm)
cm = matplotlib.colormaps['bone']
fig, ax = plt.subplots()
#   Configure directory to save results
os.makedirs('results/MIP/', exist_ok=True)
#   Create projections
n = 45
projections = []
for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
    rotated_img = rotate_on_axial_plane(img_dcm, alpha)
    projection = MIP_sagittal_plane(rotated_img)
    plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
    plt.savefig(f'results/MIP/Projection_{idx}.png')  # Save animation
    projections.append(projection)  # Save for later animation
# Save and visualize animation
animation_data = [
    [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
    for img in projections
]
anim = animation.ArtistAnimation(fig, animation_data,
                                 interval=250, blit=True)
anim.save('results/MIP/Animation.gif')  # Save animation
plt.show()  # Show animation
#########################################################################################################################3
#for i in slides3d:
#    plt.imshow(MIP_sagittal_plane(slides3d), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0] / pixel_len_mm[1])
#    plt.show()


if False:
    masked[0].save('masked1.gif', format='GIF',
                   append_images=masked[1:],
                   save_all=True,
                   duration=200,  # Duration between frames in milliseconds
                   loop=0)  # Set loop value to 0 for infinite loop

if False:
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

