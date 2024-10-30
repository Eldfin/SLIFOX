# SLIFOX
Scattered Light Imaging Fitting and Orientation ToolboX.  
Finds the peaks in Scattered Light Imaging measurement data, fits a given distribution to this peaks and calculates the nerve fiber directions.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Functions](#functions)
- [Examples](#examples)

## Installation

Installed Python version >= 3.11.3 is necessary.  
Operating system should be Linux.

### Clone the repository

```bash
git clone https://github.com/Eldfin/SLIFOX.git
```

### Create a virtual environment
```bash
cd SLIFOX
python -m venv .venv
```

### Activate the virtual environment
* On Windows:
```bash
.venv\Scripts\activate
```
* On macOS and Linux:
```bash
source .venv/bin/activate
```

### Install the SLIF package
```bash
pip install .
```


## Features
- Finds the number of peaks and the peak positions in a SLI intensity profile.
- Fit a distribution to the SLI intensity profile to refine the found peak positions.
- Calculates the nerve fiber directions of a SLI intensity profile (in particular for overlapping nerve fibers) by searching for similar directions in neighbouring pixels.


## Usage

### Functions
Below the most important functions are listed.   
Very large data that does not fit into memory can be splitted and processed in chunks with the "process_image_in_chunks" function.

#### Function: `fit_pixel_stack`
##### Description
Fits the data of one pixel.

##### Parameters
- `angles`: np.ndarray (n, )  
    Array that stores the angles at which the intensities are measured.  
- `intensities`: np.ndarray (n, )  
    The measured intensities of the pixel.  
- `intensities_err`: np.ndarray (n, )  
    The error (standard deviation) of the corresponding measured intensities of the pixel.  
- `distribution`: "wrapped_cauchy", "von_mises", or "wrapped_laplace"  
    The name of the distribution.  
- `n_steps_height`: int  
    Number of variations in height to search for best initial guess.  
- `n_steps_mu`: int  
    Number of variations in mu to search for best initial guess.  
- `n_steps_scale`: int  
    Number of variations in scale to search for best initial guess.  
- `n_steps_fit`: int  
    Number of initial guesses to pick for fitting (starting from best inital guesses).  
- `min_steps_diff`: int
    Number of the minimum difference between any step (height, mu, scale) of the initial guesses.  
- `fit_height_nonlinear`: boolean  
    Whether to include the heights in the nonlinear fitting or not.  
- `refit_steps`: int  
    Number that defines how often the fitting process should be repeated with the result  
    as new initial guess.  
- `init_fit_filter`: None or list  
    List that defines which filter to apply before the first fit.   
    This filter will be applied on the intensities before doing anything and  
    will be remove after one fit is done. Then the normal fitting process starts with  
    this result as initial guess.  
    First value of the list is a string with:  
    "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".  
    The following one to two values are the params for this filter (scipy docs).  
- `method`: string  
    Defines which fitting method to use. Can be anything from the methods in lmfit.minimize  
    and additionally "biteopt" (Derivative-Free Global Optimization Method).  
- `only_peaks_count`: int  
    Defines a filter for found peaks, so that if the count of found peaks is not equal that number,  
    the function return the same as if no peaks are found.  
- `max_fit_peaks`: int
    Defines the maximum number of peaks that should be fitted.
- `max_find_peaks`: int
    Defines the maximum number of peaks that should be found.  
    More peaks then this will be cut off (so no performeance improvements by less max found peaks).
- `max_peak_hwhm`: float  
    Estimated maximum peak half width at half maximum.  
- `min_peak_hwhm`: float  
    Estimated minimum peak half width at half maximum.  
- `mu_range`: float  
    Range of mu (regarding estimated maximum and minimum bounds around true mu).  
- `scale_range`: float  
    Range of scale (regarding estimated maximum and minimum bounds around true scale).  
- `return_result_errors`: bool
    Whether to also return the error (standard deviation) of the fitted parameters.  
    Default is False.

##### Returns
- `best_parameters`: np.ndarray (m, )  
    Array which stores the best found parameters.  
- `peaks_mask`: np.ndarray (n_peaks, n)  
    Array that stores the indices of the measurements that corresponds (mainly) to a peak.  
- `params_err`: np.ndarray (m, ), optional
    Array which stores the errors of the best found parameters.  
    Only returned if return_result_errors is True.

#### Function: `fit_image_stack`
##### Description
Fits the data of a full image stack.

##### Parameters
- `image_stack`: np.ndarray (n, m, p)  
    Array that stores the p intensity measurements for every pixel in the (n*m sized) image  
- `distribution`: "wrapped_cauchy", "von_mises", or "wrapped_laplace"  
    The name of the distribution.  
- `fit_height_nonlinear`: boolean  
    Whether to include the heights in the nonlinear fitting or not.  
- `threshold`: int  
    Threshold value. If the mean intensity of one pixel is lower than that threshold value,  
    the pixel will not be evaluated.  
- `image_stack_err`: np.ndarray (n, m, p)
    The standard deviation (error) of the measured intensities in the image stack.
    Default options is the sqrt of the intensities.  
- `n_steps_height`: int  
    Number of variations in height to search for best initial guess.  
- `n_steps_mu`: int  
    Number of variations in mu to search for best initial guess.  
- `n_steps_scale`: int  
    Number of variations in scale to search for best initial guess.  
- `n_steps_fit`: int  
    Number of initial guesses to pick for fitting (starting from best inital guesses).  
- `min_steps_diff`: int
    Number of the minimum difference between any step (height, mu, scale) of the initial guesses.  
- `refit_steps`: int  
    Number that defines how often the fitting process should be repeated with the result  
    as new initial guess.  
- `init_fit_filter`: None or list  
    List that defines which filter to apply before the first fit.   
    This filter will be applied on the intensities before doing anything and  
    will be removed after one fit is done. Then the normal fitting process starts with  
    this result as initial guess.  
    First value of the list is a string with:  
    "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".  
    The following one to two values are the params for this filter (scipy docs).  
- `method`: string  
    Defines which fitting method to use. Can be anything from the methods in lmfit.minimize  
    and additionally "biteopt (Derivative-Free Global Optimization Method).  
- `only_peaks_count`: int  
    Defines a filter for found peaks, so that if the count of found peaks is not equal that number,  
    the function returns the same as if no peaks are found.  
- `max_fit_peaks`: int
    Defines the maximum number of peaks that should be fitted.
- `max_find_peaks`: int
    Defines the maximum number of peaks that should be found.  
    More peaks then this will be cut off (so no performeance improvements by less max found peaks). 
- `max_peak_hwhm`: float  
    Estimated maximum peak half width at half maximum.  
- `min_peak_hwhm`: float  
    Estimated minimum peak half width at half maximum.  
- `mu_range`: float  
    Range of mu (regarding estimated maximum and minimum bounds around true mu).  
- `scale_range`: float  
    Range of scale (regarding estimated maximum and minimum bounds around true scale).  
- `num_processes`: int  
    Number that defines in how many sub-processes the fitting process should be split into.  
- `return_result_errors`: bool
    Whether to also return the error (standard deviation) of the fitted parameters.
    Default is False.
      
##### Returns
- `image_params`: np.ndarray (n, m, q)  
    Array which stores the best found parameters for every pixel (of n*m pixels).  
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)  
    Array that stores the indices of the measurements that corresponds (mainly) to a peak,  
    for every pixel (of n*m pixels).  
- `image_params_errors`: np.ndarray (n, m, q), optional
    Array which stores the errors of the best found parameters for every pixel (of n*m pixels).
    Only returned if return_result_errors is True.

#### Function: `find_image_peaks`
##### Description
Finds the peaks of an image stack using only the peak finder (no fitting).

##### Parameters
- `image_stack`: np.ndarray (n, m, p)  
    Array that stores the p intensity measurements for every pixel in the (n*m sized) image  
- `threshold`: int  
    Threshold value. If the mean intensity of one pixel is lower than that threshold value,  
    the pixel will not be evaluated.  
- `image_stack_err`: np.ndarray (n, m, p)
    The standard deviation (error) of the measured intensities in the image stack.
    Default options is the sqrt of the intensities. 
- `pre_filter`: None or list  
    List that defines which filter to apply before the peak finding.   
    First value of the list is a string with:  
    "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".  
    The following one to two values are the params for this filter (scipy docs).  
- `only_peaks_count`: int  
    Defines a filter for found peaks, so that if the count of found peaks is not equal that number,  
    the function return the same as if no peaks are found.  
- `max_find_peaks`: int  
    Defines the maximum number of peaks that should be found (from highest to lowest).  
    More peaks then this will be cut off (so no performeance improvements by less max found peaks).
- `max_peak_hwhm`: float  
    Estimated maximum peak half width at half maximum.  
- `min_peak_hwhm`: float  
    Estimated minimum peak half width at half maximum.  
- `mu_range`: float  
    Range of mu (regarding estimated maximum and minimum bounds around true mu).  
- `scale_range`: float  
    Range of scale (regarding estimated maximum and minimum bounds around true scale).  
- `num_processes`: int  
    Number that defines in how many sub-processes the fitting process should be split into.  
      
##### Returns
- `image_peaks_mus`: np.ndarray (n, m, max_find_peaks)  
        Array which stores the best found parameters for every pixel (of n*m pixels).  
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)  
    Array that stores the indices of the measurements that corresponds (mainly) to a peak,  
    for every pixel (of n*m pixels).  


#### Function: `plot_data_pixels`
##### Description
Plots all the intensity profiles of the pixels of given data.

##### Parameters
- `data`: np.ndarray (n, m, p)  
    The image stack containing the measured intensities.  
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.  
- `image_params`: np.ndarray (n, m, q)  
    The output of fitting the image stack, which stores the parameters of the full fitfunction.  
    q = 3 * n_peaks + 1, is the number of parameters (max 19 for 6 peaks).  
- `image_peaks_mask`: np.ndarray (n, m, n_peaks, p)  
    The mask defining which of the p-measurements corresponds to one of the peaks.  
    The first two dimensions are the image dimensions.  
- `image_peak_pairs`: np.ndarray (n, m, 3, 2)  
    The peak pairs for every pixel, where the fourth dimension contains both peak numbers of  
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension  
    is the number of the peak pair (up to 3 peak-pairs for 6 peaks).  
    The first two dimensions are the image dimensions.  
- `only_mus`: bool  
    Defines if only the mus (for every pixel) are given in the image_params.  
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")  
    The name of the distribution.  
- `indices`: np.ndarray (n, m, 2)  
    The array storing the x- `and` y-coordinate of one pixel (if plotted data != full data).  
- `directory`: string  
    The directory path where the plots should be written to.  
        
##### Returns
- `None`


#### Function: `get_image_peak_pairs`
##### Description
Finds all the peak_pairs for a whole image stack and sorts them by comparing with neighbour pixels.

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `image_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
    Can also store only mus, when only_mus = True.
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `method`: string or list of strings
    Method that is used to sort the possible peak pair combinations.  
    Can be "single", "neighbor", "pli", "significance" or "random".  
    "single" will only return a combination if there is only one possible (no sorting).
    "neighbor" will sort the possible peak pair combinations by neighbouring peak pairs.
    "pli" will sort the possible peak pair combinations by given 3d-pli measurement data.
    "significance" will sort the peak pair combinations by direction significance.
    "random" will sort the peak pair combinations randomly.
    Can also be a list containing multiple methods that are used in order.
    E.g. ["neighbor", "significance"] will sort remaining combinations of a pixel by significance 
    if sorting by neighbor was not sucessfull.
- `min_distance`: float
    Defines the minimum (180 degree periodic) distance between two paired peaks in degree.
- `max_distance`: float
    Defines the maximum (180 degree periodic) distance between two paired peaks in degree.
    Default is 180 degree (no limit).
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution used for calculation of the goodness-of-fit for gof thresholding.
- `only_mus`: bool
    Defines if only the mus (for every pixel) are given in the image_params.
- `num_processes`: int
    Defines the number of processes to split the task into.
- `amplitude_threshold`: float
    Peaks with a amplitude below this threshold will not be evaluated.
- `rel_amplitude_threshold`: float
    Value between 0 and 1.
    Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below
    this threshold will not be evaluated.
- `gof_threshold`: float
    Value between 0 and 1. If greater than 0, only fitted peaks can be paired.
    Peaks with a goodness-of-fit value below this threshold will not be evaluated.
- `significance_threshold`: float
    Value between 0 and 1. Peak Pairs with peaks that have a significance
    lower than this threshold are not considered for possible pairs.
    This Value should stay low, so that the number of possible pairs is not reduced too much.
    A too high value can lead to wrong pairs, because of pairing only "good" peaks.
    See also "direction_significance" function for more info.
- `nb_significance_threshold`: float
    Value between 0 and 1. Neighboring directions with a lower significance are not considered
    as match for the neighboring method. This threshold should be high, when using
    the neighbor method. Lower value will lead to faster computing times, but increased
    probability of wrong pairs.
- `significance_weights`: list (2, )
    The weights for the amplitude and for the goodnes-of-fit, when calculating the significance.
    See also "direction_significance" function for more info.
- `significance_sens`: list (2, )
    The sensitivity values for the amplitude (first value) and for the goodness-of-fit (second value),
    when calculating the significance.
- `max_paired_peaks`: int
    Defines the maximum number of peaks that are paired.
    Value has to be smaller or equal the number of peaks in image_params (and max 6)
    (max_paired_peaks <= max_fit_peaks or max_find_peaks)
- `nb_diff_threshold`: float
    Threshold in degrees defining when a neighboring pixel direction is used to pair peaks
    with the "neighbor" method.
- `pli_diff_threshold`: float
    If the difference between the measured PLI direction and the calculated PLI direction from SLI
    is larger than this threshold, the "pli" method will not return any peak pair combination.
- `max_attempts`: int
    Number defining how many times it should be attempted to find a neighboring pixel 
    with a direction difference lower than the given "nb_diff_threshold".
- `search_radius`: int
    The radius within which to search for the closest pixel with a defined direction.
- `min_directions_diff`: float
    Value between 0 and 180.
    If any difference between directions of a peak pair is lower than this value,
    then this peak pair combination is not considered.
- `exclude_lone_peaks`: bool
    Whether to exclude lone peaks when calculating the directions for comparison.
    Since lone peak directions have a high probability to be incorrect, due to an 
    unfound peak, this value should normally stay True. This is just for the 
    comparing process, so lone peaks will still be visible in the returned peak pairs 
    with with a pair like e.g. [2, -1] for the second peak index.
- `image_num_peaks`: np.ndarray (n, m)
    If the number of peaks are already calculated, they can be inserted here to speed up the process.
- `image_sig_peaks_mask`: np.ndarray (n, m, max_find_peaks)
    If the significant peaks mask is already calculated, 
    it can be inserted here to speed up the process.
- `image_pli_directions`: np.ndarray (n, m)
    The directions in radians (0 to pi) from a pli measurement used for the method "pli".

##### Returns
- `image_peak_pairs`: np.ndarray (n, m, p, np.ceil(max_paired_peaks / 2), 2)
    The peak pairs for every pixel, where the fifth dimension contains both peak numbers of
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the fourth dimension
    is the number of the peak pair (dimension has length np.ceil(num_peaks / 2)).
    The third dimension contains the different possible combinations of peak pairs,
    which has the length:
    p = math.factorial(num_peaks) // ((2 ** (num_peaks // 2)) * math.factorial(num_peaks // 2))
            so n = 3 for num_peaks = 4 and n = 15 for num_peaks = 6.
            Odd numbers of num_peaks have the same dimension size as num_peaks + 1.
    The first two dimensions are the image dimensions.

#### Function: `map_directions`
##### Description
Calculates the directions from given peak_pairs.

##### Parameters
- `image_peak_pairs`: np.ndarray (n, m, np.ceil(max_paired_peaks / 2), 2)  
    The peak pairs for every pixel, where the fourth dimension contains both peak numbers of  
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension  
    is the number of the peak pair (up to 3 peak-pairs for 6 peaks).  
    The first two dimensions are the image dimensions.  
- `image_mus`: np.ndarray (n, m, p)  
    The mus (centers) of the found (p) peaks for everyone of the (n * m) pixels.  
- `only_peaks_count`: int or list of ints   
    Only use pixels where the number of peaks equals this number. -1 uses every number of peaks.
- `exclude_lone_peaks`: bool
    Whether to exclude the directions for lone peaks 
    (for peak pairs with only one number unequal -1 e.g. [2, -1]).
- `image_direction_sig`: np.ndarray (n, m, 3)
    Image containing the significance of every direction for every pixel.  
    Can be created with "map_direction_significances" or "get_image_direction_significances".  
    Used for threshold filtering with significance_threshold.
- `significance_threshold`: float
    Value between 0 and 1.  
    Directions with a significance below this threshold will not be mapped.
- `directory`: string  
    The directory path defining where direction images should be writen to.  
    If None, no images will be writen.  
- `normalize`: bool
    Whether the created image should be normalized (amd displayed with colors)
- `normalize_to`: list
    List of min and max value that defines the range the image is normalized to.  
    If min (or max) is None, the minimum (or maximum) of the image will be used.
- `image_directions`: np.ndarray (n, m, 3)
    If the directions have already been calculated, they can be inserted here.

##### Returns
- `directions`: (n, m, np.ceil(max_paired_peaks / 2))  
    The calculated directions for everyoe of the (n * m) pixels.  
    Max 3 directions (for 6 peaks).  

#### Function: `map_direction_significances`
##### Description
Maps the significances of all found directions from given "image_peak_pairs".

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `image_peak_pairs`: np.ndarray (n, m, np.ceil(max_paired_peaks / 2), 2)
    The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
    is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    The first two dimensions are the image dimensions.
- `image_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution.
- `amplitude_threshold`: float
    Peaks with a amplitude below this threshold will not be evaluated.  
- `rel_amplitude_threshold`: float
    Value between 0 and 1.  
    Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below  
    this threshold will not be evaluated.  
- `gof_threshold`: float
    Value between 0 and 1.  
    Peaks with a goodness-of-fit value below this threshold will not be evaluated.  
- `weights`: list (2, )
    The weights for the amplitude and for the goodnes-of-fit, when calculating the significance
- `sens`: list (2, )
    The sensitivity values for the amplitude (first value) and for the goodness-of-fit (second value),
    when calculating the significance.
- `only_mus`: boolean
    Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.
- `directory`: string
    The directory path defining where the significance image should be writen to.
    If None, no image will be writen.
- `normalize`: bool
    Whether the created image should be normalized (amd displayed with colors)
- `normalize_to`: list
    List of min and max value that defines the range the image is normalized to.  
    If min (or max) is None, the minimum (or maximum) of the image will be used.
- `num_processes`: int
    Defines the number of processes to split the task into.
- `image_direction_sig:` np.ndarray (n, m, 3)
    If the direction significances have already been calculated, they can be inserted here.

##### Returns
- `image_direction_sig`: (n, m, np.ceil(max_paired_peaks / 2))
        The calculated significances (ranging from 0 to 1) for everyoe of the (n * m) pixels.
        Max 3 significances (shape like directions). 

#### Function: `map_fom`
##### Description
    Creates and writes the fiber orientation map (fom) from given direction (files) to a file.

##### Parameters
- `image_directions`: np.ndarray (n, m, p)
    Directions for every pixel in the image. "p" is the number of directions per pixel.  
    If None, direction_files should be defined instead.
- `direction_files`: list (of strings)
    List of the paths to the direction files that should be used to create the fom.  
    If None, image_directions should be used as input instead.
- `output_path`: string
    Path to the output directory.
- `direction_offset`: float
    The direction offset in degree. Default is zero.
- `image_direction_sig`: np.ndarray (n, m, p)
    Significances of the directions that will be multiplied with the fom image,
    to darken directions with low significance. 

##### Returns
- `rgb_fom`: np.ndarray (2*n, 2*m, 3)
    Fiber orientation map (fom) from the directions of the image.

#### Function: `map_number_of_peaks`
##### Description
Maps the number of peaks for every pixel.

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `image_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution.
- `amplitude_threshold`: float
    Peaks with a amplitude below this threshold will not be evaluated.
- `rel_amplitude_threshold`: float
    Value between 0 and 1.  
    Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below  
    this threshold will not be evaluated.
- `gof_threshold`: float
    Value between 0 and 1.  
    Peaks with a goodness-of-fit value below this threshold will not be evaluated.
- `directory`: string
    The directory path defining where the significance image should be writen to.
    If None, no image will be writen.
- `colormap`: np.ndarray (6, 3)
    Colormap used for the image generation.
- `image_sig_peaks_mask`: np.ndarray (n, m, max_find_peaks)
    When significant peaks mask is already calculated,   
    it can be provided here to speed up the process.
- `image_num_peaks`: np.ndarray (n, m)
    If the number of peaks have already been calculated, they can be inserted here.

##### Returns
- `image_num_peaks`: (n, m)
    The number of peaks for every pixel.

#### Function: `map_peak_distances`
##### Description
Maps the distance between two paired peaks for every pixel.

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `image_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `image_peak_pairs`: np.ndarray (n, m, np.ceil(max_paired_peaks / 2), 2)
    The peak pairs for every pixel, where the fourth dimension contains both peak numbers of
    a pair (e.g. [1, 3], which means peak 1 and peak 3 is paired), and the third dimension
    is the number of the peak pair (up to 3 peak-pairs for 6 peaks).
    The first two dimensions are the image dimensions.  
    If image_peak_pairs is None, only_peaks_count is set to 2.  
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution.
- `amplitude_threshold`: float
    Peaks with a amplitude below this threshold will not be evaluated.
- `rel_amplitude_threshold`: float
    Value between 0 and 1.  
    Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below  
    this threshold will not be evaluated.
- `gof_threshold`: float
    Value between 0 and 1.  
    Peaks with a goodness-of-fit value below this threshold will not be evaluated.
- `only_mus`: boolean
    Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.
- `deviation`: boolean
    If true, the distance deviation to 180 degrees will be mapped, so that values of 0  
    represent peak distances of 180 degrees.
- `only_peaks_count`: int
    Only use pixels where the number of peaks equals this number. -1 for use of every number of peaks.
- `directory`: string
    The directory path defining where the significance image should be writen to.
    If None, no image will be writen.
- `normalize`: bool
    Whether the created image should be normalized (amd displayed with colors)
- `normalize_to`: list
    List of min and max value that defines the range the image is normalized to.  
    If min (or max) is None, the minimum (or maximum) of the image will be used.
- `num_processes`: int
    Defines the number of processes to split the task into.
- `image_sig_peaks_mask`: np.ndarray (n, m, max_find_peaks)
    If significant peaks mask is already calculated, 
    it can be provided here to speed up the process.
- `image_num_peaks`: np.ndarray (n, m)
    If number of peaks are already calculated,
    they can be provided here to speed up the process.
- `image_distances`: np.ndarray (n, m)
    If the distances have already been calculated, they can be inserted here.


##### Returns
- `image_distances`: (n, m)
    The distances between paired peaks for every pixel.

#### Function: `map_mean_peak_amplitudes`
##### Description
Maps the mean peak amplitude for every pixel.

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `image_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution.
- `amplitude_threshold`: float
    Peaks with a amplitude below this threshold will not be evaluated.
- `rel_amplitude_threshold`: float
    Value between 0 and 1.  
    Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below  
    this threshold will not be evaluated.  
- `gof_threshold`: float
    Value between 0 and 1.  
    Peaks with a goodness-of-fit value below this threshold will not be evaluated.
- `only_mus`: boolean
    Whether only the mus are provided in image_params. If so, only amplitude_threshold is used.
- `directory`: string
    The directory path defining where the significance image should be writen to.
    If None, no image will be writen.
- `normalize`: bool
    Whether the created image should be normalized (amd displayed with colors)
- `normalize_to`: list
    List of min and max value that defines the range the image is normalized to.  
    If min (or max) is None, the minimum (or maximum) of the image will be used.
- `image_sig_peaks_mask`: np.ndarray (n, m)
    If significant peaks mask is already calculated, 
    it can be provided here to speed up the process.
- `image_mean_amplitudes`: np.ndarray (n, m)
    If the mean amplitudes have already been calculated, they can be inserted here.

##### Returns
- `image_mean_amplitudes`: (n, m)
    The mean amplitude for every pixel.

#### Function: `map_mean_peak_widths`
##### Description
Maps the mean peak width for every pixel.

##### Parameters
- `image_stack`: np.ndarray (n, m, p)
    The image stack containing the measured intensities.
    n and m are the lengths of the image dimensions, p is the number of measurements per pixel.
- `image_params`: np.ndarray (n, m, q)
    The output of fitting the image stack, which stores the parameters of the full fitfunction.
- `image_peaks_mask`: np.ndarray (n, m, max_find_peaks, p)
    The mask defining which of the p-measurements corresponds to one of the peaks.
    The first two dimensions are the image dimensions.
- `distribution`: string ("wrapped_cauchy", "von_mises", or "wrapped_laplace")
    The name of the distribution.
- `amplitude_threshold`: float
    Peaks with a amplitude below this threshold will not be evaluated.
- `rel_amplitude_threshold`: float
    Value between 0 and 1.  
    Peaks with a relative amplitude (to maximum - minimum intensity of the pixel) below  
    this threshold will not be evaluated.
- `gof_threshold`: float
    Value between 0 and 1.  
    Peaks with a goodness-of-fit value below this threshold will not be evaluated.
- `directory`: string
    The directory path defining where the significance image should be writen to.
    If None, no image will be writen.
- `normalize`: bool
    Whether the created image should be normalized (amd displayed with colors)
- `normalize_to`: list
    List of min and max value that defines the range the image is normalized to.  
    If min (or max) is None, the minimum (or maximum) of the image will be used.
- `image_sig_peaks_mask`: np.ndarray (n, m)
    If significant peaks mask is already calculated, 
    it can be provided here to speed up the process.
- `image_mean_widths`: np.ndarray (n, m)
    If the mean widths have already been calculated, they can be inserted here.

##### Returns
- `image_mean_widths`: (n, m)
    The mean amplitude for every pixel.

### Examples

#### Fit a single pixel
```python
import numpy as np
import h5py
from SLIFOX import fit_pixel_stack, show_pixel
from SLIFOX.filters import apply_filter

# Settings:
data_file_path = "/home/user/workspace/SLI_Data.h5"
dataset_path = "pyramid/00"
pixel = [6311, 3300]
distribution = "wrapped_cauchy"
#pre_filter = ["gauss", 2]


# Pick the pixel data
with h5py.File(data_file_path, "r") as h5f:

    intensities = h5f[dataset_path][pixel[0], pixel[1]]
    intensities_err = np.sqrt(intensities)

    # Apply filter
    #intensities = apply_filter(intensities, pre_filter)
    #intensities_err = apply_filter(intensities_err, pre_filter)


# Calculate the angles
angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)

# Fit the pixel
best_parameters, peaks_mask = fit_pixel_stack(angles, intensities, intensities_err, 
                                                fit_height_nonlinear = True,
                                                n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 10, 
                                                n_steps_fit = 3, min_steps_diff = 5,
                                                refit_steps = 0, init_fit_filter = None, max_fit_peaks = 4,
                                                method="leastsq", distribution = distribution)

print("Optimized parameters:", best_parameters)

# Show the pixel
show_pixel(intensities, intensities_err, best_parameters, peaks_mask, distribution)

```

#### Fit image stack
```python
import h5py
from SLIFOX import fit_image_stack, pick_data
import os

# Settings
dataset_path = "pyramid/00"
data_file_path = "/home/user/workspace/SLI_Data.h5"
output_directory = "output"
output_filename = "output.h5"
area = None  # [x_left, x_right, y_top, y_bot]
randoms = 0 # number of random pixels to pick from data (0 equals full data)
distribution = "wrapped_cauchy"

# Pick the SLI measurement data
data, indices = pick_data(data_file_path, dataset_path, area = area, randoms = randoms)

# Optional: Write the picked data array to a HDF5 file
#with h5py.File("input.h5", "w") as h5f:
#    group = h5f.create_group(dataset_path)
#    group.create_dataset("data", data=data)
#    group.create_dataset("indices", data=indices)

# Fit the picked data
image_params, image_peaks_mask = fit_image_stack(data, fit_height_nonlinear = True, 
                                threshold = 1000, distribution = distribution,
                                n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 10, 
                                n_steps_fit = 3, min_steps_diff = 5,
                                refit_steps = 0, init_fit_filter = None, max_fit_peaks = 4,
                                method="leastsq", num_processes = 2)

# Write the output to a HDF5 file
if output_directory != "" and not os.path.exists(output_directory):
    os.makedirs(output_directory)

with h5py.File(output_directory + output_filename, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("params", data=image_params)
    group.create_dataset("peaks_mask", data=image_peaks_mask)
    group.create_dataset("indices", data=indices)
```

#### Only find peaks (without fitting)
```python
import h5py
import matplotlib.pyplot as plt
from SLIFOX import find_image_peaks, pick_data
import os

# Settings
dataset_path = "pyramid/00"
data_file_path = "/home/user/workspace/SLI_Data.h5"
output_file_path = "output/output.h5"
area = None  # [x_left, x_right, y_top, y_bot]
randoms = 0 # number of random pixels to pick from data (0 equals full data)
distribution = "wrapped_cauchy"

# Pick the SLI measurement data
data, indices = pick_data(data_file_path, dataset_path, area = area, randoms = randoms)

# Optional: Write the picked data array to a HDF5 file
#with h5py.File("input.h5", "w") as h5f:
#    group = h5f.create_group(dataset_path)
#    group.create_dataset("data", data=data)
#    group.create_dataset("indices", data=indices)

# Find the peaks from the picked data
image_mus, image_peaks_mask = find_image_peaks(data, threshold = 1000, pre_filter = None, 
                        only_peaks_count = -1, num_processes = 2)


# Optional: Write the output to a HDF5 file
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
with h5py.File(output_file_path, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("mus", data=image_mus)
    group.create_dataset("peaks_mask", data=image_peaks_mask)
    group.create_dataset("indices", data=indices)
```

#### Fit data and create (direction, ...) maps
```python
import h5py
import matplotlib.pyplot as plt
from SLIFOX import fit_image_stack, get_image_peak_pairs, pick_data, plot_data_pixels,\
                    map_number_of_peaks, map_peak_distances, map_mean_peak_amplitudes, \
                    map_mean_peak_widths, map_directions, map_direction_significances, map_fom, \
                    get_sig_peaks_mask
from SLIFOX.filters import apply_filter
import os

# Settings
dataset_path = "pyramid/00"
data_file_path = "/home/user/workspace/SLI_Data.h5"
output_file_path = "output/output.h5"
area = None  # [x_left, x_right, y_top, y_bot]
randoms = 0 # number of random pixels to pick from data (0 equals full data)
distribution = "wrapped_cauchy"
num_processes = 2
#pre_filter = ["gauss", 1]

# Pick the SLI measurement data
data, indices = pick_data(data_file_path, dataset_path, area = area, randoms = randoms)

# Optional: Filter the data before processing
#data = apply_filter(data, pre_filter)

# Optional: Write the picked data array to a HDF5 file
#with h5py.File("input.h5", "w") as h5f:
#    group = h5f.create_group(dataset_path)
#    group.create_dataset("data", data=data)
#    group.create_dataset("indices", data=indices)

# Fit the picked data
image_params, image_peaks_mask = fit_image_stack(data, fit_height_nonlinear = True, 
                                threshold = 1000, distribution = distribution,
                                n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 10, 
                                n_steps_fit = 3, min_steps_diff = 5,
                                refit_steps = 0, init_fit_filter = None, max_fit_peaks = 4,
                                method="leastsq", num_processes = num_processes)

# Optional: Write the output to a HDF5 file
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
with h5py.File(output_file_path, "w") as h5f:
    group = h5f.create_group(dataset_path)
    group.create_dataset("params", data=image_params)
    group.create_dataset("peaks_mask", data=image_peaks_mask)
    group.create_dataset("indices", data=indices)

# Optional: Pick SLIF output data (if already fitted)
#image_params, _ = pick_data(output_file_path, dataset_path + "/params")
#image_peaks_mask, _ = pick_data(output_file_path, dataset_path + "/peaks_mask")
    
# Find the peak pairs (directions)
image_peak_pairs = get_image_peak_pairs(data, image_params, image_peaks_mask, method = "neighbor",
                            min_distance = 20, distribution = distribution, 
                            only_mus = False, num_processes = num_processes,
                            amplitude_threshold = 1000, rel_amplitude_threshold = 0.1, 
                            gof_threshold = 0.2, significance_threshold = 0, 
                            significance_weights = [1, 1], significance_sens = [1, 1],
                            nb_angle_threshold = 5, 
                            max_attempts = 50, search_radius = 50, max_paired_peaks = 6)

# Use best pairs of all possible pairs
best_image_peak_pairs = image_peak_pairs[:, :, 0, :, :]

# Optional: Plot the picked data (only recommended for small image areas)
#plot_data_pixels(data, image_params, image_peaks_mask, best_image_peak_pairs, 
#                        distribution = distribution, indices = indices, directory = "plots")

# Calculate the nerve fiber directions and save direction map in directory
image_mus = image_params[:, :, 1::3]
image_directions = map_directions(best_image_peak_pairs, image_mus, directory = "maps",
                                    exclude_lone_peaks = True)

# Map the significance of the directions
image_direction_sig = map_direction_significances(data, best_image_peak_pairs, image_params, 
                                image_peaks_mask, distribution = distribution, 
                                weights = [1, 1], sens = [1, 1], num_processes = num_processes)

# Optional: Map the threshold filtered direction images 
# (when pairs filtered with same threshold redundant)
# map_direction_significance can also be called with specific amplitude / gof thresholds beforehand
#map_directions(best_image_peak_pairs, image_mus, directiory = "maps",  exclude_lone_peaks = True,
#                image_direction_sig = image_direction_sig, significance_threshold = 0.8)

# Create the fiber orientation map (fom) using the two direction files (for max 4 peaks)
map_fom(image_directions, output_path = "maps")

# Create a mask for the significant peaks
image_sig_peaks_mask = get_sig_peaks_mask(image_stack = image_stack, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask,
                            distribution = distribution, 
                            amplitude_threshold = amplitude_threshold, 
                            rel_amplitude_threshold = rel_amplitude_threshold, 
                            gof_threshold = gof_threshold, only_mus = only_mus)

# Create map for the number of peaks (will be saved as .tiff in "maps" directory)
image_num_peaks = map_number_of_peaks(image_sig_peaks_mask = image_sig_peaks_mask, directory = "maps")

# Create map for the distance between two paired peaks
map_peak_distances(image_params = image_params, only_mus = False, deviation = False,
                            image_peak_pairs = best_image_peak_pairs, directory = "maps",
                            num_processes = num_processes, only_peaks_count = -1,
                            image_sig_peaks_mask = image_sig_peaks_mask,
                            image_num_peaks = image_num_peaks)

# Create map for the mean amplitudes
map_mean_peak_amplitudes(image_stack = data, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, distribution = "wrapped_cauchy", 
                            only_mus = False, directory = "maps",
                            image_sig_peaks_mask = image_sig_peaks_mask)

# Create map for the mean peak widths
map_mean_peak_widths(image_stack = data, image_params = image_params, 
                            image_peaks_mask = image_peaks_mask, distribution = "wrapped_cauchy", 
                            directory = "maps", image_sig_peaks_mask = image_sig_peaks_mask)   
```








