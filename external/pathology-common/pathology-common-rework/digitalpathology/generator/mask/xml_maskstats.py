import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict

class XML_MaskStats(object):
    """This class can collect and handle statistics from xml file."""

    def __init__(self, file, spacing_factor=None, spacing_tolerance=None, mask_labels=None):
        """
        Initialize the object and extract and store necessary mask data in memory for efficient patch extraction.
        """

        self._bounding_boxes = {}
        self._xml_path = ''           # Path of the mask mr-image.
        self._mask_labels = []          # List of labels that are processed.
        self._label_map = None         # Label value to index map for all values.
        self._initdata(file=file, spacing_factor=spacing_factor, spacing_tolerance=spacing_tolerance, mask_labels=mask_labels)

    def _initdata(self, file, spacing_factor, spacing_tolerance, mask_labels):
        """
        Initialize by either processing the mask file or loading the pre-processed information from a stat file.
        """
        if file is None:
            raise IOError(file)

        self._load_xml(xml_filepath=file, spacing_factor=spacing_factor, mask_labels=mask_labels)
        self._label_map = {mask_labels[n]:n for n in range(len(mask_labels))}

    def _parse_ASAP_XML_file(self):
        """
        Returns X,Y coordinate lists for every annotation as a per-label dictionary
        """
        tree = ET.parse(self._xml_path)
        root = tree.getroot()
        coord_array = []
        annot_array = defaultdict(list)

        for coord in root.iter('Annotation'):
            group = coord.get('PartOfGroup')
            if group not in self._mask_labels:
                continue
            for element in coord.iter('Coordinate'):
                X = int(float(element.get('X')))
                Y = int(float(element.get('Y')))
                coord_array.append((X, Y))
            annot_array[group].append(coord_array)
            coord_array = []
        return annot_array

    def _annotations_to_bounding_boxes(self, annot_array, spacing_factor):
        bounding_boxes = defaultdict(list)
        for k, v in annot_array.items():
            for coord in v:
                xmin = min([x[0] for x in coord]) // spacing_factor
                ymin = min([x[1] for x in coord]) // spacing_factor
                xmax = max([x[0] for x in coord]) // spacing_factor
                ymax = max([x[1] for x in coord]) // spacing_factor
                bounding_boxes[k].append([ymin, xmin , ymax, xmax])
        return bounding_boxes

    def _load_xml(self, xml_filepath, spacing_factor, mask_labels):
        """
        Extract and store necessary mask data in memory for efficient patch extraction.
        """
        self._xml_path = xml_filepath
        self._mask_labels = mask_labels

        if type(xml_filepath) is str:
            coord_array = self._parse_ASAP_XML_file()
        else:
            raise IOError(xml_filepath)

        self._bounding_boxes = self._annotations_to_bounding_boxes(coord_array, spacing_factor)
        self._bounding_boxes = {k: sorted(self._bounding_boxes[k], key=lambda x: x[0]) for k,v in self._bounding_boxes.items()}

    @property
    def path(self):
        return self._xml_path

    @property
    def spacing(self):
        return None

    @property
    def shape(self):
        return None

    @property
    def labels(self):
        return list(self._bounding_boxes.keys())

    @property
    def counts(self):
        """
        Get the number of pixels per label.
        """
        return {class_name:len(annotations) for class_name, annotations in self._bounding_boxes.items()}

    def indextocoorindate(self, index_array, label):
        """
        Convert labeled pixel indices to pixel coordinates for the given label.
        """
        label_classname = label
        coordinate_array = np.empty((len(index_array), 3), dtype=np.int32)
        for sample_index in range(len(index_array)):
            bbox = self._bounding_boxes[label_classname][index_array[sample_index]]
            x = np.random.randint(bbox[0], bbox[2] + 1)
            y = np.random.randint(bbox[1], bbox[3] + 1)
            coordinate_array[sample_index, :] = [x,y, self._label_map[label]]
        return coordinate_array

    def construct(self, row, col, height, width):
        bboxes = self._get_bounding_boxes_for_coordinate(row,col, height, width)
        return bboxes

    def _get_bounding_boxes_for_coordinate(self, y, x, height, width):
        patch_box = [y-height//2, x-width//2, y+height//2, x+width//2]
        bboxes = defaultdict(list)
        for k, v in self._bounding_boxes.items():
            if k not in self._label_map.keys():
                continue
            for candidate_bbox in v:
                if self._check_overlap_bounding_boxes(candidate_bbox, patch_box):
                    candidate_bbox = self._anchor_bbox(candidate_bbox, patch_box)
                    bboxes[k].append(candidate_bbox)
        return bboxes


    def _check_overlap_bounding_boxes(self, a, b):
        bottom = np.max([a[0], b[0]])
        top = np.min([a[2], b[2]])
        left = np.max([a[1], b[1]])
        right = np.min([a[3], b[3]])
        do_intersect = bottom < top and left < right
        return do_intersect

    def _anchor_bbox(self, a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[0], a[3] - b[1]]



