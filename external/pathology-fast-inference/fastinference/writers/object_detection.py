from numpy.core.fromnumeric import nonzero
from ..async_wsi_writer import async_wsi_writer
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time

from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

class object_detection(async_wsi_writer):
    def __init__(self, **kwargs):
        async_wsi_writer.__init__(self, **kwargs)
        print("Using custom writer: object_detection")
        self._filepath = None
        self._COLORMAP = get_cmap("viridis")

    def _postprocess_batch(self, image_batch, mask_batch, batch_info):
        """
        Do postprocessing on batch:
            * Remove predictions outside of mask
            * Remove predictions in padded margin (to remove edge artifacts)
            * Translate local patch coordinates to global WSI coordinates.
        """
        for i, (image, mask, info) in enumerate(zip(image_batch, mask_batch, batch_info)):
            # Do not postprocess images with no predictions
            if len(image) == 0:
                continue

            # Calculate indexes of predictions that are in padded margins
            _, _, lost = self._rec_info
            lost_l, lost_r, lost_t, lost_b = lost
            def inside_padding(x, y): return x < lost_l or y < lost_t or x > self._write_tile_size - lost_r or y > self._write_tile_size - lost_b
            nonpadded_idxs = [i for i, (x, y) in enumerate(image[:, :2]) if not inside_padding(x, y)]

            # Calculate indexes of predictions outside of mask
            image_nonpadded = image[nonpadded_idxs, :].copy()
            nonzero_idxs = [i for i, (x, y) in enumerate(image_nonpadded[:, :2]) if mask[int(y), int(x)] == 1]

            # Translate predictions to global WSI
            x_topleft, y_topleft = info[0], info[1]
            translated_image = image_nonpadded[nonzero_idxs, :].copy()
            translated_image[:, 0] += x_topleft - lost_l
            translated_image[:, 1] += y_topleft - lost_t

            # Only keep x, y and class
            translated_image = translated_image[:, [0, 1, 5]]
            image_batch[i] = translated_image
        return image_batch

    def _create_file_handle(self, filepath, output_shape, spacing, resample_size):
        print("creating image: {}".format(filepath), flush=True)
        sliding_window = None  # Superfluous

        # Create ASAP XML tree
        xml_root = ET.Element("ASAP_Annotations")
        annos = ET.SubElement(xml_root, "Annotations")
        anno_groups = ET.SubElement(xml_root, "AnnotationGroups")

        # Add inference group
        inference_group_attribs = {"Name": "inference", "PartOfGroup": "None", "Color": "#000000"}
        inference_group = ET.Element("Group", inference_group_attribs)
        _ = ET.SubElement(inference_group, "Attributes")
        anno_groups.append(inference_group)

        write_row = 0
        local_sequence_nr = 0
        sequence_list = []
        final_sequence_number = -1
        filename = os.path.basename(filepath)
        self._filepath = filepath
        self._file_handle_dict[filename] = (xml_root, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size)

    def _write_tile_batch(self, filename, sequence_nr, image_batch, mask_batch, batch_info):
        write_tiles = self._postprocess_batch(image_batch, mask_batch, batch_info)
        (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size) = self._file_handle_dict[filename]

        sequence_list.append([sequence_nr, write_tiles, batch_info])
        if sequence_nr != local_sequence_nr:
            return
        else:
            sequence_list.sort(key=lambda x: x[0])
            while len(sequence_list) != 0 and sequence_list[0][0] == local_sequence_nr:
                _, write_tiles, batch_info = sequence_list.pop(0)
                write_row = self._write_tiles(writer, batch_info, write_row, write_tiles, sliding_window, resample_size, filename)
                local_sequence_nr += 1
                if local_sequence_nr == final_sequence_number:
                    self._finish_and_close_writer(write_row, sliding_window, writer, filename)
                    break

        self._file_handle_dict[filename] = (writer, sliding_window, write_row, local_sequence_nr, sequence_list, final_sequence_number, resample_size)

    def _write_tiles(self, writer, batch_info, write_row, write_tiles, sliding_window, resample_size, filename):
        print(f"writing at {batch_info[0][0]} in file: {filename}, in {time.time() - self._test_time}")
        self._test_time = time.time()
        for tile, info in zip(write_tiles, batch_info):
            if info[0] < 0 or len(tile) == 0:
                continue

            annos = writer[0]
            for (x, y, c) in tile:
                anno_element_names = [anno.attrib['Name'] for anno in annos.iter("Annotation")]

                # Add Annotation element if it doesn't exist already for that class
                tile_name = f'Annotation {int(c)}'
                if tile_name not in anno_element_names:
                    attrib_dict = {"Name": tile_name, "Type": "PointSet", "PartOfGroup": "inference", "Color": rgb2hex(self._COLORMAP(int(c)))}
                    anno = ET.Element("Annotation", attrib_dict)
                    annos.append(anno)
                else:
                    idx = anno_element_names.index(tile_name)
                    anno = annos[idx]

                # Add Coordinates element if it doesn't exist already in the Annotation element
                coords = anno.find("Coordinates") if anno.find("Coordinates") else ET.SubElement(anno, "Coordinates")

                # Add Coordinate element if it doesn't already exist in the Coordinates
                order = str(int(coords.findall("Coordinate")[-1].attrib["Order"]) + 1) if coords.findall("Coordinate") else "0"
                attrib_dict = {"Order": order, "X": str(x), "Y": str(y)}
                coord = ET.Element("Coordinate", attrib_dict)
                coords.append(coord)

        return write_row

    def _finish_and_close_writer(self, write_row, sliding_window, writer, filename):
        print("finishing writing of {}".format(filename), flush=True)
        xml_str = minidom.parseString(ET.tostring(writer)).toprettyxml(indent="    ")
        with open(self._filepath, "w") as file:
            file.write(xml_str)
        del self._file_handle_dict[filename]
        _ = self._consumer_queue.get()
        self._consumer_queue.task_done()
