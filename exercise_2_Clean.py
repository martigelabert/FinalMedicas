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
import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares


import os
from typing import Sequence


def filepath(filename, subfolder='data'):
    folder = os.path.dirname(__file__)
    return f'{folder}/{subfolder}/{filename}'


def str_floats(sequence_of_floats: Sequence[float]) -> str:
    if sequence_of_floats is None:
        return '(None)'
    return f'({", ".join([f"{coord:0.02f}" for coord in sequence_of_floats])})'


def str_quaternion(q: tuple[float, float, float, float]) -> str:
    if q is None:
        return '[None]'
    return f'{q[0]} + {q[1]}i + {q[2]}j + {q[3]}k'


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


def translation_then_axialrotation(point: tuple[float, float, float], parameters: tuple[float, ...]):
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """
    x, y, z = point
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    # Your code here:
    #   ...
    x, y, z = translation(point=(x, y, z), translation_vector=(t1, t2, t3))
    x, y, z = axial_rotation(point=(x, y, z), angle_in_rads=angle_in_rads, axis_of_rotation=(v1, v2, v3))
    return x, y, z


def screw_displacement(point: tuple[float, float, float], parameters: tuple[float, ...]):
    """ Apply to `point` the screw displacement defined by `parameters`. """
    x, y, z = point
    v1, v2, v3, angle_in_rads, displacement = parameters
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    # Your code here:
    #   ...
    x, y, z = translation(point=(x, y, z),
                                     translation_vector=(displacement * v1, displacement * v2, displacement * v3))
    x, y, z = axial_rotation(point=(x, y, z), angle_in_rads=angle_in_rads, axis_of_rotation=(v1, v2, v3))
    return x, y, z


def vector_of_residuals(ref_points: np.ndarray, inp_points: np.ndarray) -> np.ndarray:
    """ Given arrays of 3D points with shape (point_idx, 3), compute vector of residuals as their respective distance """
    # Your code here:
    #   ...
    return np.sqrt(np.sum((ref_points - inp_points) ** 2, axis=1))


def coregister_landmarks(ref_landmarks: np.ndarray, inp_landmarks: np.ndarray):
    """ Coregister two sets of landmarks using a rigid transformation. """
    initial_parameters = [
        0, 0, 0,  # Translation vector
        0,  # Angle in rads
        1, 0, 0,  # Axis of rotation
    ]
    # Find better initial parameters
    centroid_ref = np.mean(ref_landmarks, axis=0)
    centroid_inp = np.mean(inp_landmarks, axis=0)
    # Your code here:
    #   ...
    initial_parameters[0] = centroid_ref[0] - centroid_inp[0]
    initial_parameters[1] = centroid_ref[1] - centroid_inp[1]
    initial_parameters[2] = centroid_ref[2] - centroid_inp[2]

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""
        # Your code here:
        #   ...
        inp_landmarks_transf = np.asarray(
            [translation_then_axialrotation(point, parameters) for point in inp_landmarks])
        return vector_of_residuals(ref_landmarks, inp_landmarks_transf)

    # Apply least squares optimization
    result = least_squares(
        function_to_minimize,
        x0=initial_parameters,
        verbose=1)
    return result

def get_amygdala_mask(img_atlas: np.ndarray) -> np.ndarray:
    # Your code here:
    #   ...
    amygdala_mask = np.zeros_like(img_atlas)
    amygdala_mask[(121 <= img_atlas) & (img_atlas <= 147)] = 1
    return amygdala_mask

def get_dcm_paths(path):
    def _custom_sort(filename):
        x = int(filename.split('.')[0])
        return x
    path_names = sorted(os.listdir(path), key=_custom_sort)
    return list(map(lambda filename: os.path.join(path, filename), path_names))

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

def preprocessed_input(input):
    normalizedInput = input / np.max(input)
    res = np.round(normalizedInput, 2) * 100
    return res.astype(int)

def mean_absolute_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MAE between two images. """
    return np.mean(np.abs(img_input - img_reference))


def mean_squared_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MSE between two images. """
    return np.mean((img_input - img_reference)**2)


def mutual_information(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the Shannon Mutual Information between two images. """
    nbins = [10, 10]
    # Compute entropy of each image
    hist = np.histogram(img_input.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_input = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    hist = np.histogram(img_reference.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_reference = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    # Compute joint entropy
    joint_hist = np.histogram2d(img_input.ravel(), img_reference.ravel(), bins=nbins)[0]
    prob_distr = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))
    # Compute mutual information
    return entropy_input + entropy_reference - joint_entropy


_atlasPath = 'Project/AAL3_1mm.dcm'
_panthomPath = 'Project/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm'


if __name__ == '__main__':
    ########################################
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

    # Crop phantom to atlas size, play with dimensions to make it convertible to 3d cords- in order to perform
    # coregistration
    images_phantom = dc_phantom.pixel_array[6:-6, 6:-7, 6:-6] # we could also apply padding
    mask_atlas = get_amygdala_mask(images_atlas)

    mask_centroids = find_centroid(mask_atlas)
    centroid_idx = mask_centroids[0].astype('int')

    fig, ax = plt.subplots(1, 4, figsize=(10, 10))
    images = [slides3d[centroid_idx], images_phantom[centroid_idx], images_atlas[centroid_idx],mask_atlas[centroid_idx]]
    titles = ["Pacient (Pre - Corregistration)", "Phantom", "Atlas" ,"Masked Atlas (amygdala)"]
    for i in range(4):
        ax[i].imshow(images[i], cmap='bone')
        ax[i].set_title(titles[i])

    fig.suptitle("Data")
    fig.tight_layout()
    plt.show()

    #########################################
    # Clean our CT
    # manual shaping and crop of CT of the patient
    match_slices = slides3d[
                   (slides3d.shape[0] - 181) + 3 :(slides3d.shape[0] - 181) + 181,
                   45:450,
                   80:430]

    aspect_ratio_zoom = (mask_atlas.shape[0] / match_slices.shape[0],
                        (mask_atlas.shape[1] -1) / match_slices.shape[1], # mask_atlas.shape[1] -1 is to make it match with the cropped atlas
                        mask_atlas.shape[2] / match_slices.shape[2])

    zoomCT = zoom(match_slices, aspect_ratio_zoom, order=1)

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    images = [zoomCT[centroid_idx], images_phantom[centroid_idx], images_atlas[centroid_idx]]
    titles = ["zoomed CT", "Phantom", "Atlas"]
    for i in range(3):
        ax[i].imshow(images[i], cmap='bone')
        ax[i].set_title(titles[i])

    fig.suptitle("")
    fig.tight_layout()
    plt.show()

    #########################################
    def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
        return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

    rotatedCT = rotate_on_axial_plane(zoomCT, -2)  # Rotate the data on the axial plane

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    images = [zoomCT[centroid_idx], rotatedCT[centroid_idx]]
    titles = ["zoomed CT", "Rotated Zoomed CT"]
    for i in range(2):
        ax[i].imshow(images[i], cmap='bone')
        ax[i].set_title(titles[i])

    fig.suptitle("")
    fig.tight_layout()
    plt.show()
    ##########################################
    # Preprocessing input (landmarks)
    processedCT = preprocessed_input(rotatedCT)

    # Computing The landmarks (https://gist.github.com/tschreiner/8f971bbbd40606e58f1e4fb1852e8b8e)
    landmarks_ref = images_phantom[::15, ::15, ::15].reshape(-1,3)
    landmarks_input = processedCT[::15, ::15, ::15].reshape(-1,3)

    # for visualization propuses
    all_ref  = images_phantom.reshape(-1,3)
    all_input = processedCT.reshape(-1,3)

    residuals_tpm = vector_of_residuals(landmarks_ref, landmarks_input)
    print(f'>> Mean residual value: {np.mean(residuals_tpm.flatten())}.') # Before Corregistration
    ############################################

    fig = plt.figure(figsize=(15, 15))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax2.scatter(all_input[..., 0], all_input[..., 1], all_input[..., 2], marker='^', color='blue')
    ax2.set_title('Input')

    ax1.scatter(all_ref[..., 0], all_ref[..., 1], all_ref[..., 2], marker='o', color='green')
    ax1.set_title('Reference')

    for ax in [ax1, ax2]:
        ax.set_box_aspect([1, 1, 1])

    all_points = np.concatenate([all_ref, all_input], axis=0)
    ax1.set_xlim3d(np.min(all_points), np.max(all_points))
    ax1.set_ylim3d(np.min(all_points), np.max(all_points))
    ax1.set_zlim3d(np.min(all_points), np.max(all_points))
    ax2.set_xlim3d(np.min(all_points), np.max(all_points))
    ax2.set_ylim3d(np.min(all_points), np.max(all_points))
    ax2.set_zlim3d(np.min(all_points), np.max(all_points))

    for ax in [ax1, ax2]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig.suptitle("Landmark Before coregistration")
    fig.tight_layout()
    plt.show()

    #############################
    # find our optimal params
    print("Starting Corregistration")
    result = coregister_landmarks(landmarks_ref, landmarks_input)
    optimal_params = result.x

    print('optimal_params parameters:')
    print(f'  >> Best parameters: ({optimal_params}')

    ###############################
    # Apply our ideal params to the landmarks on each point
    finalLandMarks = np.asarray([translation_then_axialrotation(point, optimal_params) for point in all_input[:]])

    fig = plt.figure(figsize=(15, 15))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax2.scatter(finalLandMarks[..., 0], finalLandMarks[..., 1], finalLandMarks[..., 2], marker='^', color='blue')
    ax2.set_title('Transformed Input')

    ax1.scatter(all_ref[..., 0], all_ref[..., 1], all_ref[..., 2], marker='o', color='green')
    ax1.set_title('Reference')

    for ax in [ax1, ax2]:
        ax.set_box_aspect([1, 1, 1])

    all_points = np.concatenate([all_ref, finalLandMarks], axis=0)
    ax1.set_xlim3d(np.min(all_points), np.max(all_points))
    ax1.set_ylim3d(np.min(all_points), np.max(all_points))
    ax1.set_zlim3d(np.min(all_points), np.max(all_points))
    ax2.set_xlim3d(np.min(all_points), np.max(all_points))
    ax2.set_ylim3d(np.min(all_points), np.max(all_points))
    ax2.set_zlim3d(np.min(all_points), np.max(all_points))

    for ax in [ax1, ax2]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig.suptitle("Landmarks coregistered")
    fig.tight_layout()
    plt.show()
    #########################################

    coregistered_image = finalLandMarks.reshape(181, 216, 181)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    images = [coregistered_image[centroid_idx], images_phantom[centroid_idx]]
    titles = ["Coregistered image", "Phantom"]
    for i in range(3):
        ax[i].imshow(images[i], cmap='bone')
        ax[i].set_title(titles[i])

    fig.suptitle("")
    fig.tight_layout()
    plt.show()

    #########################################

    

    #########################################