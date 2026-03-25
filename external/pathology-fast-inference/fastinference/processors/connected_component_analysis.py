from ..async_tile_processor import async_tile_processor
import os
import numpy as np
import scipy.ndimage.measurements as measurements
from copy import deepcopy

class connected_component_analysis(async_tile_processor):
    """
    Subclassed from the async_tile_processor. Will perform connected component analysis
    (see doctring of ConnectedComponentAnalyser).
    """
    def __init__(self, read_queue, write_queues, model_path, **kwargs):
        async_tile_processor.__init__(self, read_queue, write_queues, model_path, **kwargs)
        self.cc_analyzer = ConnectedComponentAnalyser(label_size_map={1:7, 2:100, 3: 10, 4: 50, 5:7}, pixel_spacing=0.5)
        print("Using custom processor: connected component analysis.")

    def _run_loop(self):
        while True:
            tile_info = self._fast_read_queue.get()
            writer_nr = tile_info[-1]
            if tile_info[0] == 'finish_image':
                self._write_queues[writer_nr].put(tile_info[:-1])
                continue
            output_filename, sequence_nr, tile_batch, mask_batch, info, _ = tile_info
            result_batch = self._process_tile_batch(tile_batch, info)
            
            # Perform postprocessing
            result_batch = self.cc_analyzer.connected_component_filter(result_batch) 
            
            self._write_queues[writer_nr].put(('write_tile', output_filename, sequence_nr, result_batch, mask_batch, info))

#----------------------------------------------------------------------------------------------------

class ConnectedComponentAnalyser(object):
    """
    This class can perform connected component analysis and filter away
    connected components of labels that are smaller than a given size.
    """

    def __init__(self, label_size_map, pixel_spacing):
        """
        Initialize the object.

        Args:
            label_size_map (Dict): Dictionary mapping labels (keys) to minimum size (values) in micrometer of connected component.
            pixel_spacing (float): Pixel spacing in micrometer.

        """
        
        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__label_size_map = label_size_map
        self.__pixel_spacing = pixel_spacing


    def __connected_component_filter(self, image):
        """
        For every label type present in an image, find its connected components and
        fill it with the surrounding label if it does not exceed filter size.

        Args:
            image (np.ndarray): Array containing predictions.
        
        Returns:
            np.ndarray: Array containing size-filtered predictions
        """
        
        # Create copy of input image for output.
        #
        output = np.copy(image)

        # Convert unit in label size map from micrometers to pixels for easy comparison.
        #
        converted_label_size_map = {}
        for label in self.__label_size_map:
            converted_label_size_map[label] = int(self.__label_size_map[label] / self.__pixel_spacing)
        
        for label in converted_label_size_map:
            binary_image = np.equal(image.squeeze(), label)
            connected_components, _ = measurements.label(binary_image)
            objects = measurements.find_objects(connected_components)
            
            for object_ in objects:
                patch = image[object_]
                
                # If size of object is smaller than cutoff, fill it with the most occuring surrounding label.
                #
                if np.max(patch.shape) < converted_label_size_map[label]:
                    labels, counts = np.unique(patch, return_counts=True)

                    # Guard against cases where previous iterations may have already filled this detected object.
                    #
                    if label not in labels:
                        break

                    # If label covers complete patch, expand search around patch until new label is found
                    #
                    delta = 0 
                    while len(labels) <= 1:
                        delta += 10

                        # Clamp start and end indexes to 0 and image.shape[i] to make sure we stay within patch
                        #
                        x_start = max(0, object_[0].start - delta)
                        y_start = max(0, object_[1].start - delta)
                        x_end = max(object_[0].stop + delta, patch.shape[0])
                        y_end = max(object_[1].stop + delta, patch.shape[1])

                        updated_boundaries = (np.s_[x_start:x_end], np.s_[y_start:y_end])

                        expanded_patch = image[updated_boundaries]
                        labels, counts = np.unique(expanded_patch, return_counts=True)

                    pairs = dict(zip(labels, counts))
                    pairs.pop(label)
                    most_occuring_label = max(pairs.keys(), key=lambda key: pairs[key])

                    patch[patch == label] = most_occuring_label

                    # Insert patch back into output image.
                    # 
                    output[object_] = patch

        return output.squeeze()

    def connected_component_filter(self, image):
        """
        For every label type present in an image, find its connected components and 
        fill it with the surrounding label if it does not exceed filter size.

        Args:
            image (np.ndarray): Array containing predictions.
        
        Returns:
            np.ndarray: Array containing size-filtered predictions
        """

        # Copy input image so that the function does call-by-value.
        input_image = deepcopy(image)

        return self.__connected_component_filter(input_image)
