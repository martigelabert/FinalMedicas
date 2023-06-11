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
from scipy.ndimage import zoom
import math
import scipy
from scipy.optimize import least_squares

_atlasPath = 'Project/AAL3_1mm.dcm'
_panthomPath = 'Project/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm'

def get_amygdala_mask(img_atlas: np.ndarray) -> np.ndarray:
    # Your code here:
    #   ...
    amygdala_mask = np.zeros_like(img_atlas)
    #enable = [127, 121,147, 137, 135]
    #for i in enable:
    #    amygdala_mask[img_atlas == i] = 1

    #amygdala_mask[img_atlas == 46] = 1
    amygdala_mask[(121 <= img_atlas) & (img_atlas <= 147)] = 1
    #amygdala_mask[img_atlas <= 147] = 1

    return amygdala_mask


def visualize_axial_slice(# Activity 8
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        ):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    # Your code here
    #   Remember `matplotlib.colormaps['cmap_name'](...)`
    #   See also `matplotlib.colors.Normalize(vmin=..., vmax=...)`
    #   ...
    img_slice = img[mask_centroid[0].astype('int'), :, :]
    mask_slice = mask[mask_centroid[0].astype('int'), :, :]

    cmap = matplotlib.colormaps['bone']
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))
    fused_slice = \
        0.5*cmap(norm(img_slice))[..., :3] + \
        0.5*np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)
    return fused_slice

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

# Activity5

def multiply_quaternions(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    )


def conjugate_quaternion(
        q: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q[0], -q[1], -q[2], -q[3]
    )

from skimage import exposure
def matchIntensityValues(input, reference):
    input_flat = input.flatten()
    reference_flat = reference.flatten()
    matched_flat = exposure.match_histograms(input_flat, reference_flat)
    mapped = np.reshape(matched_flat, input.shape)
    return mapped


def translation(
        point: tuple[float, float, float],
        translation_vector: tuple[float, float, float]
        ) -> tuple[float, float, float]:
    """ Perform translation of `point` by `translation_vector`. """
    x, y, z = point
    v1, v2, v3 = translation_vector
    # Your code here
    # ...
    return (x+v1, y+v2, z+v3)

def axial_rotation(
        point: tuple[float, float, float],
        angle_in_rads: float,
        axis_of_rotation: tuple[float, float, float]) -> tuple[float, float, float]:
    """ Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`. """
    x, y, z = point
    v1, v2, v3 = axis_of_rotation
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    # Your code here:
    #   ...
    #   Quaternion associated to point.
    p = (0, x, y, z)
    #   Quaternion associated to axial rotation.
    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    q = (cos, sin * v1, sin * v2, sin * v3)
    #   Quaternion associated to image point
    q_star = conjugate_quaternion(q)
    p_prime = multiply_quaternions(q, multiply_quaternions(p, q_star))
    #   Interpret as 3D point (i.e. drop first coordinate)
    return p_prime[1], p_prime[2], p_prime[3]




def get_dcm_paths(path):
    def _custom_sort(filename):
        x = int(filename.split('.')[0])
        return x
    path_names = sorted(os.listdir(path), key=_custom_sort)
    return list(map(lambda filename: os.path.join(path, filename), path_names))

def mean_squared_error(a, b):
    return np.mean((a - b)**2)


def translation_rotation(point, params):
    x, y, z = point
    t1, t2, t3, angle_in_rads, v1, v2, v3 = params
    t_x, t_y, t_z = translation([x, y, z], [t1, t2, t3])
    r_x, r_y, r_z = axial_rotation([t_x, t_y, t_z], angle_in_rads,[v1, v2, v3])
    return [r_x, r_y, r_z]


if __name__ == '__main__':

    dc_atlas = pydicom.dcmread(_atlasPath)
    dc_phantom = pydicom.dcmread(_panthomPath)

    files = get_dcm_paths('Project/RM_Brain_3D-SPGR')
    slices = []

    _sliceThickness = None
    for f in files:
        slices.append(pydicom.dcmread(f))

    _sliceThickness = slices[0].SliceThickness
    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing_x, pixel_spacing_y = pixel_spacing[0], pixel_spacing[1]
    pixel_len_mm = [_sliceThickness, pixel_spacing_y, pixel_spacing_x]

    slices = sorted(slices, key=lambda s: s.SliceLocation)
    slides3d = np.array([s.pixel_array for s in slices])

    images_atlas = dc_atlas.pixel_array
    # Crop phantom to atlas size
    images_phantom = dc_phantom.pixel_array[6:-6, 6:-7, 6:-6]

    mask_atlas = get_amygdala_mask(images_atlas)

    def find_centroid(mask: np.ndarray) -> np.ndarray:
        # Your code here:
        #   Consider using `np.where` to find the indices of the voxels in the mask
        #   ...
        idcs = np.where(mask == 1)
        centroid = np.stack([
            np.mean(idcs[0]),
            np.mean(idcs[1]),
            np.mean(idcs[2]),
        ])
        return centroid

    mask_centroids_index = find_centroid(mask_atlas)[0].astype('int')

    # manual shaping and croping of patient data
    # slides3d
    # Checked with 3d slicer to make it more or less accuarated
    match_slices = slides3d[
                   (slides3d.shape[0] - 181) :(slides3d.shape[0] - 181) + 181,
                   42:430,
                   80:430]

    aspect_ratio_zoom = (mask_atlas.shape[0] / match_slices.shape[0],
                         (mask_atlas.shape[1] -1 ) / match_slices.shape[1],
                      mask_atlas.shape[2] / match_slices.shape[2])

    final_match_slices = zoom(match_slices, aspect_ratio_zoom, order=1)

    # try and error
    rotated_final_match_slices = \
        scipy.ndimage.rotate(final_match_slices, 5, axes=(1, 2), reshape=False)

    # BORRAR
    #processed_input = normalize_intensity(rotated_final_match_slices, images_phantom)
    rotated_final_match_slices = matchIntensityValues(rotated_final_match_slices, images_phantom)
    # Generate landmarks
    _normalizedLandmarks = rotated_final_match_slices / np.max(rotated_final_match_slices)
    landMarks_images_pacient = np.round(_normalizedLandmarks, 2) * 100

    def start_corregistration(landMarksA, landMarksB):
        params_init =np.array([0, 0, 0,
                 0,
                 1, 0, 0
                 ])
        #translation = [0, 0, 0]
        #angle = 0
        #axialRotation = [1,0,0]

        cA = np.mean(landMarksA, axis=0)
        cB= np.mean(landMarksB, axis=0)

        params_init[0] = cA[0] - cB[0]
        params_init[1] = cA[1] - cB[1]
        params_init[2] = cA[2] - cB[2]

        def minimize(params):
            #t1,t2,t3, angle, v1,v2,v3 = params
            #translation = np.array([ t1,t2,t3])
            #axialRotation = np.array([v1,v2,v3])
            tmpLandMarks = []
            for point in landMarksB:
                tmpPoint = translation_rotation(point, params)
                tmpLandMarks.append(tmpPoint)
            tmpLandMarks = np.array(tmpLandMarks)
            # distance
            ressiduals = np.linalg.norm(landMarksA - tmpLandMarks, axis=1)
            print(ressiduals)
            return ressiduals

        return least_squares(minimize,x0=params_init, verbose=2)

    # Coregister landmarks, we want to map the phantom and our image landmarks
    # So we transform our data into points
    # landMarksA = images_phantom.reshape(-1, 3)
    # landMarksB = landMarks_images_pacient.reshape(-1, 3)
    #optimal_params = start_corregistration(landMarksA, landMarksB)
    # withouth downsampling is untractable

    landMarksADownSampled = images_phantom[::4, ::4, ::4].reshape(-1,3)
    #down_sampled_inp_shape = processed_input[::4, ::4, ::4].shape
    landMarksBDownSampled = landMarks_images_pacient[::4, ::4, ::4].reshape(-1,3)

    optimal_params = start_corregistration(landMarksADownSampled, landMarksBDownSampled)

    #optimal_params = start_corregistration(landMarksA, landMarksB)
    # optimal_params = [2, 3, 2.1396e+09, 1.38e+09, 3.80e+00,2.40e+07    ]
    # [1.5, -2.00, -2.38, 0.15, 0.62, 0.51, 0.62]
    print(optimal_params)
    a = optimal_params
    tranformationMatrix = [
        [1.0041,-0.0281733,0.00628774, 0.551022],
        [0.0370751, 1.02498, 0.119226, -11.8917],
        [0.0118107,-0.0851302, 1.02398, 11.9543],
        #[0,0,0,1]
    ]
    #tranformationMatrix = np.array(tranformationMatrix)

    #a = slides3d.dot(tranformationMatrix.T)
    #for i in a:
    #    print("i")












    # Si realizas crop no se requiere esta parte
    padded_images_atlas = [np.pad(img_atlas,[(int((images_phantom.shape[0] - images_atlas.shape[0]) / 2),),
                                             (int((images_phantom.shape[1] - images_atlas.shape[1]) / 2),)],mode='constant') for img_atlas in images_atlas]

    padded_images_atlas = np.array(padded_images_atlas)

    for i,img_atlas in enumerate(images_atlas):
        fig, axs = plt.subplots(3, 3)
        axs[0,0].imshow(padded_images_atlas[i,:,:], cmap='bone')
        axs[0,1].imshow(padded_images_atlas[:, i, :], cmap='bone')
        axs[0,2].imshow(padded_images_atlas[:, :, i], cmap='bone')

        axs[1,0].imshow(images_phantom[i,:,:], cmap='bone')
        axs[1,1].imshow(images_phantom[:, i, :], cmap='bone')
        axs[1,2].imshow(images_phantom[:, :, i], cmap='bone')

        axs[2,0].imshow(slides3d[i,:,:], cmap='bone' )
        axs[2,1].imshow(slides3d[:, i, :], cmap='bone', aspect=_sliceThickness)
        axs[2,2].imshow(slides3d[:, :, i], cmap='bone' , aspect=_sliceThickness)
        fig.show()
        plt.show()

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
