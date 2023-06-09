import pydicom
import numpy as np

dataset = pydicom.dcmread("HCC_005/01-23-1999-NA-ABDPELVIS-36548/300.000000-Segmentation-06660/1-1.dcm")

segmentation_sequence = dataset.PerFrameFunctionalGroupsSequence

for segment_item in segmentation_sequence:
    # Get the frame index for the current segmentation slice
    frame_index = segment_item.FrameIndex

    # Find the instance number corresponding to the frame index
    instance_number = None
    for index, frame in enumerate(dataset.FrameIndexSequence):
        if frame.FrameIndex == frame_index:
            instance_number = frame.DimensionIndexValues[0]
            break

    if instance_number is not None:
        slice_index = instance_number - 1  # Adjusting for 0-based indexing

        # Do something with the slice index...
        print("Slice Index:", slice_index)
