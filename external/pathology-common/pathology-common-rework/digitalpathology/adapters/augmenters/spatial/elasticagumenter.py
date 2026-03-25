"""
This file contains a class for augmenting patches from whole slide images by applying elastic transformation.
"""

from . import spatialaugmenterbase as dptspatialaugmenterbase

from ....errors import augmentationerrors as dptaugmentationerrors

import scipy.ndimage.interpolation
import scipy.ndimage.filters
import numpy as np

#----------------------------------------------------------------------------------------------------

class ElasticAugmenter(dptspatialaugmenterbase.SpatialAugmenterBase):
    """Apply elastic deformation to patch. Deformation maps are created when the first patch is deformed."""

    def __init__(self, sigma_interval, alpha_interval, map_count, interpolation_order=1):
        """
        Initialize the object.

        Args:
            sigma_interval (tuple): Interval for sigma selection for Gaussian filter map. Example: [40.0, 80.0]
            alpha_interval (tuple): Interval for alpha selection for the severity of the deformation. Example: [4000.0, 6000.0]
            map_count (int): Amount of deformation maps to precalculate. Example: 64.
            interpolation_order (int): Interpolation order from the range [0, 5].

        Raises:
            InvalidElasticSigmaIntervalError: The interval of sigma for elastic deformation is invalid.
            InvalidElasticAlphaIntervalError: The interval of alpha for elastic deformation is invalid.
            InvalidElasticMapCountError: The number of elastic deformation maps to precalculate is invalid.
            InvalidElasticInterpolationOrderError: The interpolation order for elastic transformation is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='elastic')

        # Initialize members.
        #
        self.__sigma_interval = []      # Sigma.
        self.__alpha_interval = []      # Alpha.
        self.__map_count = 0            # Number of deformation maps to pre-calculate.
        self.__interpolation_order = 0  # Interpolation order.
        self.__deformation_maps = {}    # Deformation maps per patch shape.
        self.__map_choice = 0           # Selected deformation map.

        # Save configuration.
        #
        self.__cofiguredeformationmaps(sigma_interval=sigma_interval, alpha_interval=alpha_interval, map_count=map_count, interpolation_order=interpolation_order)

    def __cofiguredeformationmaps(self, sigma_interval, alpha_interval, map_count, interpolation_order):
        """
        Configure the deformation map calculation parameters.

        Args:
            sigma_interval (tuple): Interval for sigma selection for Gaussian filter map.
            alpha_interval (tuple): Interval for alpha selection for the severity of the deformation.
            map_count (int): Amount of deformation maps to precalculate.
            interpolation_order (int): Interpolation order from the range [0, 5].

        Raises:
            InvalidElasticSigmaIntervalError: The interval of sigma for elastic deformation is invalid.
            InvalidElasticAlphaIntervalError: The interval of alpha for elastic deformation is invalid.
            InvalidElasticMapCountError: The number of elastic deformation maps to precalculate is invalid.
            InvalidElasticInterpolationOrderError: The interpolation order for elastic transformation is not valid.
        """

        # Check the sigma interval.
        #
        if len(sigma_interval) != 2 or sigma_interval[1] < sigma_interval[0] or sigma_interval[0] <= 0.0:
            raise dptaugmentationerrors.InvalidElasticSigmaIntervalError(sigma_interval)

        # Check the alpha interval.
        #
        if len(alpha_interval) != 2 or alpha_interval[1] < alpha_interval[0] or alpha_interval[0] <= 0.0:
            raise dptaugmentationerrors.InvalidElasticAlphaIntervalError(alpha_interval)

        # Check if the map count is positive.
        #
        if map_count <= 0:
            raise dptaugmentationerrors.InvalidElasticMapCountError(map_count)

        # Check the interpolation order.
        #
        if interpolation_order < 0 or 5 < interpolation_order:
            raise dptaugmentationerrors.InvalidElasticInterpolationOrderError(interpolation_order)

        # Store the settings.
        #
        self.__sigma_interval = list(sigma_interval)
        self.__alpha_interval = list(alpha_interval)
        self.__map_count = int(map_count)
        self.__interpolation_order = int(interpolation_order)

    def __createdeformationmaps(self, image_shape):
        """
        Elastic deformation of images as described in Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis",
        in Proc. of the International Conference on Document Analysis and Recognition, 2003.

        Args:
            image_shape (tuple): Image shape to deform.

        Raises:
            InvalidElasticImageShapeError: Not a 2D grayscale or 3 channel image is transformed.
        """

        # Test image shape. Maps are generated for 2D and 3D images, where the last channel in the colors of length 3.
        #
        if len(image_shape) != 2 and not (len(image_shape) == 3 and image_shape[2] == 3):
            raise dptaugmentationerrors.InvalidElasticImageShapeError(image_shape)

        image_shape_2d = image_shape[:2]
        image_shape_3d = image_shape[:2] + (3,)

        # Generate all maps.
        #
        self.__deformation_maps[image_shape_2d] = []
        self.__deformation_maps[image_shape_3d] = []

        for _ in range(self.__map_count):
            # Randomize parameters from teh given ranges.
            #
            alpha = np.random.uniform(low=self.__alpha_interval[0], high=self.__alpha_interval[1], size=None)
            sigma = np.random.uniform(low=self.__sigma_interval[0], high=self.__sigma_interval[1], size=None)

            # Calculate 2D displacements.
            #
            row_displacement = np.random.rand(*image_shape_2d) * 2 - 1
            row_displacement = scipy.ndimage.filters.gaussian_filter(input=row_displacement, sigma=sigma, mode='constant', cval=0) * alpha

            col_displacement = np.random.rand(*image_shape_2d) * 2 - 1
            col_displacement = scipy.ndimage.filters.gaussian_filter(input=col_displacement, sigma=sigma, mode='constant', cval=0) * alpha

            row, col = np.meshgrid(np.arange(image_shape_2d[0]), np.arange(image_shape_2d[1]), indexing='ij')
            indices = (np.add(row, row_displacement).flatten(), np.add(col, col_displacement).flatten())

            self.__deformation_maps[image_shape_2d].append(indices)

            # Calculate 3D displacements by repeating the 2D displacements for each channel.
            #
            row_displacement = np.repeat(row_displacement[..., None], repeats=image_shape_3d[-1], axis=-1)
            col_displacement = np.repeat(col_displacement[..., None], repeats=image_shape_3d[-1], axis=-1)

            row, col, channel = np.meshgrid(np.arange(image_shape_3d[0]), np.arange(image_shape_3d[1]), np.arange(image_shape_3d[2]), indexing='ij')
            indices = (np.add(row, row_displacement).flatten(), np.add(col, col_displacement).flatten(), channel.flatten())

            self.__deformation_maps[image_shape_3d].append(indices)

    def transform(self, patch):
        """
        Deform the image with a random deformation map.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.

        Raises:
            InvalidElasticImageShapeError: Not a 2D grayscale or 3 channel image is transformed.
        """

        # Initialize the deformation maps.
        #
        if patch.shape not in self.__deformation_maps:
            self.__createdeformationmaps(patch.shape)

        # Apply elastic deformation.
        #
        indices = self.__deformation_maps[patch.shape][self.__map_choice]
        order = 0 if patch.ndim < 3 else self.__interpolation_order
        patch_transformed = scipy.ndimage.interpolation.map_coordinates(input=patch, coordinates=indices[:patch.ndim], order=order, mode='reflect').reshape(patch.shape)

        # Clip the values between the original range.
        #
        if 1 < order:
            patch_transformed = np.clip(patch_transformed, a_min=patch.min(), a_max=patch.max())

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize the transformation map.
        #
        self.__map_choice = np.random.randint(low=0, high=self.__map_count - 1) if 1 < self.__map_count else 0
