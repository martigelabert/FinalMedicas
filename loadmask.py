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




ds = pydicom.dcmread('HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660/1-1.dcm')

import pydicom_seg

dcm = pydicom.dcmread('HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660/1-1.dcm')

reader = pydicom_seg.SegmentReader()
result = reader.read(dcm)

for segment_number in result.available_segments:
    image_data = result.segment_data(segment_number)  # directly available
    image = result.segment_image(segment_number)  # lazy construction
    #print(f'{result["segment_infos"][1]}')

#for segment_number in result.available_segments: # numero de segmentos
#    result.segment_infos[segment_number].SegmentLabel # nombre del label
#    result.segment_data(segment_number) # labels de la imagen