# DigitalPathology Coding Style Guide

## Introduction

This document contains the coding style guidelines for the DigitalPathology library. In general the code should follow the style described in [PEP 8](https://www.python.org/dev/peps/pep-0008). This document describes the extensions of deviations from PEP 8 that are used in the DigitalPathology library.

## Docstrings

```python
"""
This file contains class for sampling patches from whole slide images.
"""

import ...

#----------------------------------------------------------------------------------------------------

class PatchSampler(object):
   """
   This class can sample patches from an image considering various conditions like mask and
   probability.
   """

   def __init__(self, patch_source, input_channels):
      """
      Initialize the object: load the image, load a mask for the configured image and check 
      compatibility, extract and store necessary mask data in memory for efficient patch extraction.

      Args:
         patch_source (dptpatchsource.PatchSource): Image patch source descriptor. 
         input_channels (list): Desired channels that are extracted for each patch.

      Raises:
         ImageChannelIndexError: A channel index is outside the range of available channels.
         MaskLabelListMismatchError: Mask or stat content does not match the current label settings.
         ImageLevelError: The specified level is invalid for the mask image.
         StatShapeMismatchError: The shape of the loaded stat and the given mask file cannot be matched.

         DigitalPathologyImageError: Image errors.
         DigitalPathologyLabelError: Label errors.
      """
```

* Use Google style docstring comments.
* Every file have to have a docstring.
* Every function and class declaration are to have docstring.
* Use every field:
  * Description
  * Args
  * Returns
  * Raises
* Use complete descriptions of the arguments, including types.
* Leave out the empty fields.
* The raised exceptions from the same module should be listed one-by-one, while exceptions from a different module can be grouped by base class. 

## Comments

```python
import ...

#----------------------------------------------------------------------------------------------------

def serialize_ndarray(array):
    """
    <docstring>
    """

    # Stream the content of the numpy ndarray to a byte buffer.
    #
    content_buffer = io.BytesIO()
    np.save(file=content_buffer, arr=array, allow_pickle=False)

    # Return the byte array representation of the numpy ndarray.
    #
    return content_buffer.getvalue()

#----------------------------------------------------------------------------------------------------

def reconstruct_ndarray(content):
    """
    <docstring>
    """
```

* Class declarations and non-member function declarations are separated with a comment line of 100 dashes, without any space in the comment.
* The import statements are separated from the rest of the file with the same separator comment line.
* Comments end with punctuation.
* Non-inline comments are separated from the next line with an empty comment line.

## Naming

### Files
```bash
batchgenerator.py
data_set_distributed.yaml
generate_documentation.sh
```

* Python files: All lower case, without space of underscore.
* Configuration files: All lower case, words separated by underscore.
* Shell script files: All lower case, words separated by underscore.

### Source

```python
class PatchBuffer(object):
    """<docstring>"""

    def __init__(self, shapes, input_channels, label_maps, weight_maps, cache_size, chunk_size):
        """
        <docstring>
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__read_index = 0   # Start index of the next batch to read from the buffer.
        self.__write_index = 0  # Start index of the next batch to write to the buffer.

        ...

    def size(self):
       """
       <docstring>
       """

       return self.__buffer_size if self.__buffer is not None else 0

    def setsize(self, size):
        """
        <docstring>
        """

        ...

#----------------------------------------------------------------------------------------------------

def save_buffer(file_path):
        """
        <docstring>
        """

        ...
```

* Classes are named using CamelCase.
* Member functions are named all lower case, without underscore.
* Non-member function are named all lower case, with separating underscores.
* Arguments and variables are named all lower case, with separating underscores.
* All names should be meaningful. (No, *"a", "aa", "in", "val"* names are not meaningful.)
* Getter methods does not have a get- prefix.
* Setter methods have a set- prefix.

## Spacing

```python
    self.__configurelabels(label_mapper=label_mapper, label_dist=label_dist, label_mode=label_mode)
    
    self.__configuresampler(patch_sources=patch_sources,
                            category_dist=category_dist,
                            strict_selection=strict_selection,
                            patch_augmenter=patch_augmenter,
                            process_count=sampler_process_count,
                            pool_size=sampler_pool_size,
                            join_timeout=join_timeout,
                            response_timeout=response_timeout,
                            poll_timeout=poll_timeout,
                            multi_threaded=multi_threaded,
                            chunk_size=sampler_chunk_size)
                        
    # Return parsed values.
    #
    return (parsed_image_path,
            parsed_overrides,
            parsed_input_level,
            parsed_normalizer,
            parsed_target_range,
            parsed_source_range,
            parsed_soft_mode,
            parsed_channels,
            parsed_quantize,
            parsed_keep_intermediates,
            parsed_overwrite)
```
* The line length is 200 characters.
* Function declarations, function calls and tuple definitions that span multiple lines should be separated into one argument per line layout.

## Other

```python
"""
This module can load a network model and apply classification on a whole slide.
"""

import imageprocessing.classification.classification as imgprocclass

import digitalpathology.batch.batchsource as dptbatchsource
import digitalpathology.utils.foldercontent as dptfoldercontent

import argparse
import os

#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    ...
    
    # Assemble job quartets: (image path, mask path, mask level, output path, interval path).
    #
    job_list, file_mode = assemble_jobs(image_path=image_path,
                                        mask_path=mask_path,
                                        mask_level=mask_level,
                                        purpose_filter=purposes,
                                        path_overrides=overrides,
                                        output_path=output_path,
                                        interval_path=quantization_interval_path)
```

* Always use the ```__name__ == "__main__"``` checking in callable scripts.
* Order the imports from local to python packages to built in libraries, from special to more general.
* Group the imports by separating the groups with an empty line.
* Always use the named arguments for function calls with external libraries.
