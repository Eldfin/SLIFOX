import numpy as np
import imageio as io
from numba import njit, prange
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import tifffile
from .utils import imread

# Generate fom in same manner as SLIX

class Colormap:
    @staticmethod
    def prepare(direction: np.ndarray, inclination: np.ndarray,
                offset: float = 0) -> (np.ndarray, np.ndarray):
        if direction.size == 0 or inclination.size == 0:
            return direction, inclination

        if np.abs(offset) > np.pi:
            offset = np.deg2rad(offset)
        if direction.max(axis=None) > np.pi and not np.isclose(direction.max(axis=None), np.pi):
            direction = np.deg2rad(direction)
        if inclination.max(axis=None) > np.pi and not np.isclose(inclination.max(axis=None), np.pi):
            inclination = np.deg2rad(inclination)
        direction = (direction + offset) % np.pi

        # If inclination is only 2D and direction is 3D, we need to make sure that the
        # inclination matches the shape of the direction.
        if inclination.ndim == 2 and direction.ndim == 3:
            inclination = inclination[..., np.newaxis]
        if inclination.ndim == 3 and inclination.shape[-1] != direction.shape[-1]:
            inclination = np.repeat(inclination, direction.shape[-1], axis=-1)

        return direction, inclination
    @staticmethod
    def hsv_black(direction: np.ndarray, inclination: np.ndarray, offset: float = 0) -> np.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination, offset)

        hsv_stack = np.stack((direction / np.pi,
                                 np.ones(direction.shape),
                                 1.0 - (2 * inclination / np.pi)))
        hsv_stack = np.moveaxis(hsv_stack, 0, -1)
        return np.clip(hsv_to_rgb(hsv_stack), 0, 1)

def _merge_direction_images(direction_files):
    # Could be simplified with:
    #direction_images = [imread(file) for file in direction_files]
    #direction_image = [img[:, :, np.newaxis] for img in direction_images]
    #direction_image = np.concatenate(direction_image, axis=-1)

    direction_image = None
    for direction_file in direction_files:
        single_direction_image = imread(direction_file)
        if direction_image is None:
            direction_image = single_direction_image
        else:
            if len(direction_image.shape) == 2:
                direction_image = np.stack((direction_image,
                                               single_direction_image),
                                              axis=-1)
            else:
                direction_image = np.concatenate((direction_image,
                                                     single_direction_image
                                                     [:, :, np.newaxis]),
                                                    axis=-1)

    return direction_image

def _visualize_one_direction(direction, rgb_stack):
    output_image = rgb_stack
    output_image[direction == -1] = 0

    return output_image.astype('float32')


@njit(cache = True, parallel = True)
def _visualize_multiple_direction(direction, valid_directions, rgb_stack):
    output_image = np.zeros((direction.shape[0] * 2,
                                direction.shape[1] * 2,
                                3))

    r = rgb_stack[..., 0]
    g = rgb_stack[..., 1]
    b = rgb_stack[..., 2]

    # Now we need to place them in the right pixel on our output image
    for x in prange(direction.shape[0]):
        for y in prange(direction.shape[1]):
            if valid_directions[x, y] == 0:
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2] = 0
            elif valid_directions[x, y] == 1:
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2, 0] = r[x, y, 0]
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2, 1] = g[x, y, 0]
                output_image[x * 2:x * 2 + 2, y * 2:y * 2 + 2, 2] = b[x, y, 0]
            else:
                output_image[x * 2, y * 2, 0] = r[x, y, 0]
                output_image[x * 2, y * 2, 1] = g[x, y, 0]
                output_image[x * 2, y * 2, 2] = b[x, y, 0]

                output_image[x * 2 + 1, y * 2, 0] = r[x, y, 1]
                output_image[x * 2 + 1, y * 2, 1] = g[x, y, 1]
                output_image[x * 2 + 1, y * 2, 2] = b[x, y, 1]

                if valid_directions[x, y] == 2:
                    output_image[x * 2, y * 2 + 1, 0] = r[x, y, 1]
                    output_image[x * 2, y * 2 + 1, 1] = g[x, y, 1]
                    output_image[x * 2, y * 2 + 1, 2] = b[x, y, 1]

                    output_image[x * 2 + 1, y * 2 + 1, 0] = r[x, y, 0]
                    output_image[x * 2 + 1, y * 2 + 1, 1] = g[x, y, 0]
                    output_image[x * 2 + 1, y * 2 + 1, 2] = b[x, y, 0]
                else:
                    output_image[x * 2, y * 2 + 1, 0] = r[x, y, 2]
                    output_image[x * 2, y * 2 + 1, 1] = g[x, y, 2]
                    output_image[x * 2, y * 2 + 1, 2] = b[x, y, 2]

                    if valid_directions[x, y] == 3:
                        output_image[x * 2 + 1, y * 2 + 1, 0] = 0
                        output_image[x * 2 + 1, y * 2 + 1, 1] = 0
                        output_image[x * 2 + 1, y * 2 + 1, 2] = 0
                    if valid_directions[x, y] == 4:
                        output_image[x * 2 + 1, y * 2 + 1, 0] = r[x, y, 3]
                        output_image[x * 2 + 1, y * 2 + 1, 1] = g[x, y, 3]
                        output_image[x * 2 + 1, y * 2 + 1, 2] = b[x, y, 3]

    return output_image

def create_fom(direction, inclination=None, saturation=None, value=None,
              colormap=Colormap.hsv_black, direction_offset=0):
    """
    Copied from SLIX:

    Generate a 2D colorized direction image in the HSV color space based on
    the original direction. Value and saturation of the color will always be
    one. The hue is determined by the direction.

    If the direction parameter is only a 2D np array, the result will be
    a simple orientation map where each pixel contains the HSV value
    corresponding to the direction angle.

    When a 3D stack with max. three directions is used, the result will be
    different. The resulting image will have two times the width and height.
    Each 2x2 square will show the direction angle of up to three directions.
    Depending on the number of directions, the following pattern is used to
    show the different direction angles.

    1 direction:

        1 1
        1 1

    2 directions:

        1 2
        2 1

    3 directions:

        1 2
        3 0

    Args:

        direction: 2D or 3D np array containing the direction of the image
                   stack

        inclination: Optional inclination of the image in degrees. If none is set, an inclination of 0° is assumed.

        saturation: Weight image by using the saturation value. Use either a 2D image
                    or a 3D image with the same shape as the direction. If no image
                    is used, the saturation for all image pixels will be set to 1

        value:  Weight image by using the value. Use either a 2D image
                or a 3D image with the same shape as the direction. If no image
                is used, the value for all image pixels will be set to 1

        colormap: The colormap to use. Default is HSV black. The available color maps
                  can be found in the colormap class.

        direction_offset: Direction offset in degree (default: 0). This will
        change the coloring of the vectors.

    Returns:

        np.ndarray: 2D image containing the resulting HSV orientation map

    """
    direction = np.array(direction)
    direction_shape = direction.shape
    if inclination is None:
        inclination = np.zeros_like(direction)

    colors = colormap(direction, inclination, direction_offset)
    hsv_colors = rgb_to_hsv(colors)

    # If no saturation is given, create an "empty" saturation image that will be used
    if saturation is None:
        saturation = np.ones(direction.shape)
    # Normalize saturation image
    saturation = saturation / saturation.max(axis=None)
    # If we have a saturation image, check if the shape matches (3D) and correct accordingly
    while len(saturation.shape) < len(direction.shape):
        saturation = saturation[..., np.newaxis]
    if not saturation.shape[-1] == direction_shape[-1]:
        saturation = np.repeat(saturation, direction_shape[-1], axis=-1)

    # If no value is given, create an "empty" value image that will be used
    if value is None:
        value = np.ones(direction.shape)
    # Normalize value image
    value = value / value.max(axis=None)
    # If we have a value image, check if the shape matches (3D) and correct accordingly
    while len(value.shape) < len(direction.shape):
        value = value[..., np.newaxis]
    if not value.shape[-1] == direction_shape[-1]:
        value = np.repeat(value, direction_shape[-1], axis=-1)

    hsv_colors[..., 1] *= saturation
    hsv_colors[..., 2] *= value
    colors = hsv_to_rgb(hsv_colors)

    if len(direction_shape) > 2:
        # count valid directions
        valid_directions = np.count_nonzero(direction > -1, axis=-1)
        return (255.0 * _visualize_multiple_direction(direction, valid_directions, colors)).astype(np.uint8)
    else:
        return (255.0 * _visualize_one_direction(direction, colors)).astype(np.uint8)


def color_bubble(colormap, offset=0, shape=(1000, 1000, 3)):
    """
    Copied from SLIX:

    Based on the chosen colormap in methods like unit_vectors or
    direction, the user might want to see the actual color bubble to understand
    the shown orientations. This method creates an empty np array and fills
    it with values based on the circular orientation from the middle point.
    The color can be directed from the colormap argument

    Args:
        colormap: Colormap function which will be used to create the color bubble
        offset: Direction offset in degree (default: 0)
        shape: Shape of the resulting color bubble.

    Returns: np array containing the color bubble

    """

    # create a meshgrid of the shape with the position of each pixel
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    # center of our color_bubble
    center = np.array([shape[0]/2, shape[1]/2])
    # radius where a full circle is still visible
    radius = np.minimum(np.minimum(center[0], center[1]),
                           np.minimum(shape[0] - center[0], shape[1] - center[1]))
    # calculate the direction as the angle between the center and the pixel
    direction = np.pi - np.arctan2(y - center[0], x - center[1]) % np.pi

    # calculate the inclination as the distance between the center and the pixel
    inclination = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    # normalize the inclination to a range of 0 to 90 degrees where 0 degree is at a distance of radius
    # and 90 degree is at a distance of 0
    inclination = 90 - inclination / radius * 90

    # create the color bubble
    color_bubble = colormap(direction, inclination, offset)
    color_bubble[inclination < 0] = 0

    return (255.0 * color_bubble).astype('uint8')

class H5FileReader:
    """
    This class allows to read HDF5 files from your file system.
    It supports reading datasets but not reading attributes.
    """

    def __init__(self):
        self.path = None
        self.file = None
        self.content = None

    def open(self, path):
        """

        Args:

            path: Path on the filesystem to the HDF5 file which will be read

        Returns:

            None
        """
        if not path == self.path:
            self.close()
            self.path = path
            self.file = h5py.File(path, 'r')

    def close(self):
        """
        Close the currently opened file, if any is open.

        Returns:

            None
        """
        if self.file is not None:
            self.file.close()
            self.file = None
            self.path = None
            self.content = None

    def read(self, dataset):
        """
        Read a dataset from the currently opened file, if any is open.
        The content of the dataset will be stored for future use.

        Args:

            dataset: Path to the dataset within the HDF5

        Returns:

            The opened dataset.

        """
        if self.content is None:
            self.content = {}
        if dataset not in self.content.keys():
            self.content[dataset] = numpy.squeeze(self.file[dataset][:])
        return self.content[dataset]


def imwrite_rgb(filepath, data):
    """
        Write generated RGB image to given filepath.
        Supported file formats: HDF5, Tiff.
        Other file formats are only indirectly supported and might result in
        errors.

        Args:

            filepath: Path to image

            data: Data which will be written to the disk

        Returns:

            None
        """

    save_data = data.copy()
    axis = np.argwhere(np.array(save_data.shape) == 3).flatten()

    if filepath.endswith('.tiff') or filepath.endswith('.tif'):
        save_data = np.moveaxis(save_data, axis[0], 0)
        tifffile.imwrite(filepath, save_data, photometric='rgb',
                         compression=8)

def write_fom(image_directions = None, direction_files = None, output_path = None):
    """
    Creates and writes the fiber orientation map (fom) from given direction (files) to a file.

    Parameters:
    - image_directions: np.ndarray (n, m, p)
        Directions for every pixel in the image. "p" is the number of directions per pixel.
        If None, direction_files should be defined instead.
    - direction_files: list (of strings)
        List of the paths to the direction files that should be used to create the fom.
        If None, image_directions should be used as input instead.
    - output_path: string
        Path to the output directory.

    Returns:
    - rgb_fom: np.ndarray (2*n, 2*m)
        Fiber orientation map (fom) from the directions of the image.
    """
    
    if not isinstance(image_directions, np.ndarray):
        if isinstance(direction_files, list):
            image_directions = _merge_direction_images(direction_files)
            rgb_fom = create_fom(image_directions)
        else:
            raise Exception("You have to input image_directions array or direction_files list.")
    else:
        rgb_fom = create_fom(np.swapaxes(image_directions, 0, 1))
        
    imwrite_rgb(f"{output_path}/fom.tiff", rgb_fom)
    imwrite_rgb(f"{output_path}/color_bubble.tiff", color_bubble(Colormap.hsv_black))

    return rgb_fom