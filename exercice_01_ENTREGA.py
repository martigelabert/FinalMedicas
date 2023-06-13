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

def get_segmentation_masks_correctly_place():
    ds = pydicom.dcmread('HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660/1-1.dcm')
    segmentation = []
    mask1 = []
    mask2 = []
    mask3 = []
    mask4 = []

    for index,i in enumerate(ds['PerFrameFunctionalGroupsSequence']):
        image_position = i.PlanePositionSequence[0].ImagePositionPatient # última componente

        segment_seq = i.SegmentIdentificationSequence
        if segment_seq is not None:
            segment_number = segment_seq[0].ReferencedSegmentNumber # esto te dice en que segmentación estas

            if segment_number == 1:
                mask1.append((ds.pixel_array[index], image_position[2]))

            if segment_number == 2:
                mask2.append((ds.pixel_array[index], image_position[2]))

            if segment_number == 3:
                mask3.append((ds.pixel_array[index], image_position[2]))

            if segment_number == 4:
                mask4.append((ds.pixel_array[index], image_position[2]))

    def sortMaks(mask):
        lista_ordenada = sorted(mask, key=lambda tupla: tupla[1])
        lista_A = [tupla[0] for tupla in lista_ordenada]
        return lista_A

    mask1 = sortMaks(mask1)
    mask2 = sortMaks(mask2)
    mask3 = sortMaks(mask3)
    mask4 = sortMaks(mask4)

    return mask1, mask2, mask3, mask4

def execute_code(path, type, only_animation=False):
    #paths = get_dcm_paths('HCC_005/01-23-1999-NA-ABDPELVIS-36548/103.000000-LIVER 3 PHASE AP-85837')
    paths = get_dcm_paths(path)

    ds = pydicom.dcmread(paths[0])
    # Access the pixel spacing along the x and y axes
    pixel_spacing = ds.PixelSpacing
    pixel_spacing_x, pixel_spacing_y = pixel_spacing[0], pixel_spacing[1]
    # Access the slice thickness along the z-axis
    slice_thickness = ds.SliceThickness
    pixel_len_mm = [slice_thickness, pixel_spacing_y, pixel_spacing_x]  # Pixel length in mm [z, y, x]

    slices = []
    for i in paths:
        ds = pydicom.dcmread(i)
        if hasattr(ds, 'SliceLocation'):
            slices.append(ds)
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # stack the slices to obtain a 89xIMG_SIZExIMG_SIZE
    slides3d = np.array([s.pixel_array for s in slices])

    # get the mask for each class
    mask1, mask2, mask3, mask4 = get_segmentation_masks_correctly_place()#get_segmentation_masks()

    img_dcm = slides3d
    img_min = np.amin(img_dcm)
    img_max = np.amax(img_dcm)
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()

    #  Configure directory to save results
    os.makedirs('results/MIP_FINAL_SAGITTAL/', exist_ok=True)
    os.makedirs('results/MIP_FINAL_CORONAL/', exist_ok=True)

    if not(only_animation):
        #   Create projections
        n = 24
        projections = []
        for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
            rotated_img = rotate_on_axial_plane(img_dcm, alpha)
            projection = MIP_sagittal_plane(rotated_img)

            rotated_mask1 = rotate_on_axial_plane(mask1, alpha)
            rotated_mask2 = rotate_on_axial_plane(mask2, alpha)
            rotated_mask3 = rotate_on_axial_plane(mask3, alpha)
            rotated_mask4 = rotate_on_axial_plane(mask4, alpha)

            if type == 0:
                mask1_projection = MIP_sagittal_plane(rotated_mask1)
                mask2_projection = MIP_sagittal_plane(rotated_mask2)
                mask3_projection = MIP_sagittal_plane(rotated_mask3)
                mask4_projection = MIP_sagittal_plane(rotated_mask4)
            elif type==1:
                mask1_projection = MIP_coronal_plane(rotated_mask1)
                mask2_projection = MIP_coronal_plane(rotated_mask2)
                mask3_projection = MIP_coronal_plane(rotated_mask3)
                mask4_projection = MIP_coronal_plane(rotated_mask4)


            mask1_projection = np.ma.masked_equal(mask1_projection, 0)
            mask2_projection = np.ma.masked_equal(mask2_projection, 0)
            mask3_projection = np.ma.masked_equal(mask3_projection, 0)
            mask4_projection = np.ma.masked_equal(mask4_projection, 0)

            plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
            plt.imshow(mask1_projection, cmap="jet", aspect=pixel_len_mm[0] / pixel_len_mm[1], alpha=0.5)
            plt.imshow(mask2_projection, cmap="hot", aspect=pixel_len_mm[0] / pixel_len_mm[1], alpha=0.5)
            plt.imshow(mask3_projection, cmap="viridis", aspect=pixel_len_mm[0] / pixel_len_mm[1], alpha=0.5)
            plt.imshow(mask4_projection, cmap="inferno", aspect=pixel_len_mm[0] / pixel_len_mm[1], alpha=0.5)

            if type == 0:
                plt.savefig(f'results/MIP_FINAL_SAGITTAL/Projection_{idx}.png', bbox_inches='tight', pad_inches=0)  # Save animation
            elif type ==1:
                plt.savefig(f'results/MIP_FINAL_CORONAL/Projection_{idx}.png', bbox_inches='tight', pad_inches=0)  # Save animation
    #####
    path_type = ['MIP_FINAL_SAGITTAL', 'MIP_FINAL_CORONAL']
    # Directorio de las imágenes
    folder_path = f'results/{path_type[type]}'

    # Obtener la lista de imágenes en el directorio y ordenarlas por número
    image_files = sorted([f for f in os.listdir(folder_path) if re.match(r'Projection_\d+\.png', f)])

    # Crear una lista de objetos de imagen para la animación
    animation_data = []
    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path)
        animation_data.append(img)
    #    animation_data.append([plt.imshow(img, animated=True)])

    animation_data[0].save("results/"+path_type[type]+"/Animation.gif", save_all=True, append_images=animation_data[1:], optimize=False, duration=250, loop=1)

if __name__ == '__main__':
    # Select level
    SAGITTAL = 0
    CORONAL  = 1
    execute_code('HCC_005/01-23-1999-NA-ABDPELVIS-36548/103.000000-LIVER 3 PHASE AP-85837', CORONAL, False)


