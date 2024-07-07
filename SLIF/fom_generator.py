import numpy as np
import imageio as io

# Generate fom in same manner as SLIX

class Colormap:
    @staticmethod
    def hsv_black(direction: np.ndarray, inclination: np.ndarray, offset: float = 0) -> np.ndarray:
        direction, inclination = Colormap.prepare(direction, inclination, offset)

        hsv_stack = np.stack((direction / np.pi,
                                 np.ones(direction.shape),
                                 1.0 - (2 * inclination / np.pi)))
        hsv_stack = np.moveaxis(hsv_stack, 0, -1)
        return np.clip(hsv_to_rgb(hsv_stack), 0, 1)

def merge_direction_images(direction_files):
    direction_image = None
    for direction_file in direction_files:
        single_direction_image = io.imread(direction_file)
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

def write_fom(args, direction_files, output_path):
    
    direction_image = merge_direction_images(direction_files)

    rgb_fom = create_fom(direction_image)
    io.imwrite_rgb(f"{output_path}/fom.tiff", rgb_fom)
    io.imwrite_rgb(f"{output_path}/color_bubble.tiff", color_bubble(Colormap.hsv_black))