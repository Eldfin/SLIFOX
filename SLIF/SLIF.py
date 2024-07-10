import numpy as np
from scipy.optimize import lsq_linear, nnls, Bounds
from lmfit import Model
import pymp
from numba import njit
from fcmaes import bitecpp
from time import time
from .signal_filters import apply_filter
from .wrapped_distributions import distribution_pdf
from .utils import angle_distance, calculate_chi2
from .SLI_peak_finder import find_peaks

@njit(cache = True, fastmath = True)
def _objective(params, x, y, y_err, distribution):
    """
    The objective function to minimize (for the biteopt method)

    Parameters:
    - params: np.ndarray (m, )
        The params for the fitfunction
    - x: np.ndarray (n, )
        The x-data (angles in radians) for the fitfunction.
    - y: np.ndarray (n, )
        The (measured) y-data.
    - y_err: np.ndarray (n, )
        The error (standard deviation) of the y-data.
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.
    
    Returns:
    - res: float
        The sum of the squared residuals
    """
    model = full_fitfunction(x, params, distribution)
    residuals = (y - model) / y_err
    return np.sum(residuals**2)

@njit(cache = True, fastmath = True)
def full_fitfunction(x, params, distribution = "wrapped_cauchy"):
    """
    Calculates the predicted light intensities of the given distribution model for array x 

    Parameters
    ----------
    - x: np.ndarray (n, )
        The angles (in radians)
    - params: np.ndarray (3 * n_peaks + 1, )
        The array of params: (I, mu, scale) per peak + A (underground)

    Returns
    -------
    - res: np.ndarray (n, )
        Intensities of the model at all positions of x
    """

    y = np.full_like(x, params[-1])
    heights = params[0:-1:3]
    for i, height in enumerate(heights):
        if height == 0:
            continue
        y += params[i * 3] * distribution_pdf(x, params[i * 3 + 1], params[i * 3 + 2], distribution)
                                           
    return y

def _lmfit_fitfunction(x, **params):
    # wrapper of the fitfunction for lmfit

    # First parameter name is the distribution name
    distribution = next(iter(params))
    
    lmparams = np.array([params[key] for key in params])

    return full_fitfunction(x, lmparams, distribution)

@njit(cache = True, fastmath = True)
def _calculate_scale_bounds(distribution, min_peak_hwhm, max_peak_hwhm, hwhm, scale_range):
    """
    Calculates the bounds for the scale parameter. 

    Parameters
    ----------
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.
    - min_peak_hwhm: float
        Estimated minimum peak half width at half maximum.
    - max_peak_hwhm: float
        Estimated maximum peak half width at half maximum.
    - hwhm: float
        The measured hwhm.
    - scale_range: float
        Range of scale (regarding estimated maximum and minimum bounds around true scale).

    Returns
    -------
    - min_scale: float
        Minimum bound for the scale parameter.
    - max_scale: float
        Maximum bound for the scale parameter.
    """

    if distribution == "wrapped_cauchy":
        # hwhm = scale (=: gamma)
        # scale = 95_ci_width / (2 * tan(pi/40))

        min_scale = max(0.0, min_peak_hwhm, hwhm - scale_range / 2)
        max_scale = min(1.0, max_peak_hwhm, hwhm + scale_range / 2)

    elif distribution == "wrapped_normal":
        # scale =: sigma (standard deviation)
        # hwhm = sqrt(2*ln(2)) * scale
        # scale =  95_ci_width / 3.92

        min_scale_defined = min_peak_hwhm / np.sqrt(2*np.log(2))
        max_scale_defined = max_peak_hwhm / np.sqrt(2*np.log(2))
        scale_calculated = hwhm / np.sqrt(2*np.log(2))
        min_scale = max(0.0, min_scale_defined, scale_calculated - scale_range / 2)
        max_scale = min(3.0, max_scale_defined, scale_calculated + scale_range / 2)

    elif distribution == "von_mises":
        # scale ~=~ 2 * ln(2) / hwhm**2
        
        min_scale_defined = 2 * np.log(2) / max_peak_hwhm**2
        max_scale_defined = 2 * np.log(2) / min_peak_hwhm**2
        scale_calculated = 2 * np.log(2) / hwhm**2
        min_scale = max(0.0, min_scale_defined, scale_calculated - scale_range / 2)
        max_scale = min(5.0, max_scale_defined, scale_calculated + scale_range / 2)

    elif distribution == "wrapped_laplace":
        # HWHM = ln(2) * b
        min_scale_defined = min_peak_hwhm / np.log(2)
        max_scale_defined = max_peak_hwhm / np.log(2)
        scale_calculated = hwhm / np.log(2)
        min_scale = max(0.0, min_scale_defined, scale_calculated - scale_range / 2)
        max_scale = min(5.0, max_scale_defined, scale_calculated + scale_range / 2)

    return min_scale, max_scale

@njit(cache = True, fastmath = True)
def create_bounds(angles, intensities, intensities_err, distribution, 
                    peaks_mask, peaks_mus, mu_range, scale_range,
                    min_peak_hwhm, max_peak_hwhm, global_amplitude, min_int):
    """
    Create the bounds for the fitting parameters. 

    Parameters
    ----------
    - angles: np.ndarray (n, )
        Array that stores the angles at which the intensities are measured.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The error (standard deviation) of the corresponding measured intensities of the pixel.
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.
    - peaks_mask: np.ndarray (n_peaks, n)
        Array that stores the indices of the measurements that are local minima.
    - peaks_mus: np.ndarray (n_peaks, )
        Array that stores the angles of the centers (mus) of the peaks.
    - mu_range: float
        Range of mu (regarding estimated maximum and minimum bounds around true mu).
    - scale_range: float
        Range of scale (regarding estimated maximum and minimum bounds around true scale).
    - min_peak_hwhm: float
        Estimated minimum peak half width at half maximum.
    - max_peak_hwhm: float
        Estimated maximum peak half width at half maximum.
    - global_amplitude: float
        The difference between the maximum and minimum intensity of the pixel.
    - min_int: float
        The minimum of the measured intensities.
    
    Returns
    -------
    - bounds_min: np.ndarray (m, )
        Minimum bounds for each parameter.
    - bounds_max: np.ndarray (m, )
        Maximum bounds for each parameter.
    """

    # Bounds for underground
    min_A = 0
    max_A = min_int + 0.05 * global_amplitude

    # if no peaks found return zeros
    if (len(peaks_mask) == 1 and np.all(peaks_mask[0] == 0)) or len(peaks_mus) == 0:
        bounds_min = np.zeros(4)
        bounds_max = np.zeros(4)
        return bounds_min, bounds_max

    num_peaks = len(peaks_mus)
    bounds_min = np.empty(num_peaks*3 + 1)
    bounds_max = np.empty(num_peaks*3 + 1)

    for i in range(num_peaks):
        index = 3 * i

        peak_angles = angles[peaks_mask[i]]
        peak_intensities = intensities[peaks_mask[i]]

        relative_angles = angle_distance(peaks_mus[i], peak_angles)
        angles_left = peak_angles[relative_angles < 0]
        angles_right = peak_angles[relative_angles > 0]
        intensities_left = peak_intensities[relative_angles < 0]
        intensities_right = peak_intensities[relative_angles > 0]

        if len(intensities_left) == 0:
            angles_left = peak_angles[relative_angles <= 0]
            intensities_left = peak_intensities[relative_angles <= 0]
        elif len(intensities_right) == 0:
            intensities_right = peak_intensities[relative_angles >= 0]
            angles_right = peak_angles[relative_angles >= 0]

        local_max_int = peak_intensities[np.argmin(np.abs(relative_angles))]
        amplitude = local_max_int - min_int

        # Calculate estimate of scale parameter with Half Width at Half Maximum (HWHM)
        half_max = local_max_int - amplitude / 2
        # Find points where intensity crosses half-maximum
        left_half_max_index = np.argmin(np.abs(intensities_left - half_max))
        right_half_max_index = np.argmin(np.abs(intensities_right - half_max))

        hwhm = np.abs(angle_distance(angles_left[left_half_max_index], angles_right[right_half_max_index])) / 2

        min_scale, max_scale = _calculate_scale_bounds(distribution, 
                                min_peak_hwhm, max_peak_hwhm, hwhm, scale_range)

        # Set range of Peak height according to the range of the pdf for the scale range
        # Amplitude = I * pdf(0, 0, scale)
        # For cauchy: Amplitude = I / (pi * scale)
        # Note: pdf(0, 0, scale) = full_fitfunction(0, [1, 0, scale], distribution)
        # however here distribution cases are handled explicit
        
        pdf_value1 = distribution_pdf(0, 0, min_scale, distribution)
        pdf_value2 = distribution_pdf(0, 0, max_scale, distribution)

        min_I = 0.8 * amplitude / max(pdf_value1, pdf_value2)
        max_I = 1.2 * amplitude / min(pdf_value1, pdf_value2)

        # Set bounds of mu around current maximum
        min_mu = peaks_mus[i] - mu_range / 2
        max_mu = peaks_mus[i] + mu_range / 2

        bounds_min[index:index+3] = [min_I, min_mu, min_scale]
        bounds_max[index:index+3] = [max_I, max_mu, max_scale]

    # Underground bounds
    bounds_min[-1] = min_A
    bounds_max[-1] = max_A

    return bounds_min, bounds_max

    
@njit(cache = True, fastmath = True)
def create_init_guesses(angles, intensities, intensities_err, bounds_min, bounds_max, distribution,
                    n_steps_height = 10,  n_steps_mu = 10, n_steps_scale = 5, n_steps_fit = 10):
    """
    Create the (best) initial guesses for the fitting process. 

    Parameters
    ----------
    - angles: np.ndarray (n, )
        Array that stores the angles at which the intensities are measured.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The error (standard deviation) of the corresponding measured intensities of the pixel.
    - bounds_min: np.ndarray (m, )
        Minimum bounds for each parameter.
    - bounds_max: np.ndarray (m, )
        Maximum bounds for each parameter.
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.
    - n_steps_height: int
        Number of variations in height to search for best initial guess.
    - n_steps_mu: int
        Number of variations in mu to search for best initial guess.
    - n_steps_scale: int
        Number of variations in scale to search for best initial guess.
    - n_steps_fit: int
        Number of initial guesses to pick for fitting (starting from best inital guesses).

    Returns
    -------
    - initial_guesses: np.ndarray (n_steps_fit, m + 1)
        Array which stores the params and chi2 for every initial guess.
        First entry of rows are the calculated chi2 and the following entries are the params.
    """

    n_peaks = len(bounds_min) // 3

    height_tests = np.empty((n_peaks, n_steps_height))
    mu_tests = np.empty((n_peaks, n_steps_mu))
    scale_tests = np.empty((n_peaks, n_steps_scale))
    if n_steps_height == 1: 
        height_tests[0] = np.array([bounds_min[0], bounds_max[0]]).mean()
    else: 
        height_tests[0] = np.linspace(bounds_min[0], bounds_max[0], n_steps_height)
        # Sort the test arrays from the distance of the mean
        # height_tests[0] = height_tests[0][np.argsort(np.abs(height_tests[0] - height_tests[0].mean()))]
    if n_steps_mu == 1: 
        mu_tests[0] = np.array([bounds_min[1], bounds_max[1]]).mean()
    else: 
        mu_tests[0] = np.linspace(bounds_min[1], bounds_max[1], n_steps_mu)
    if n_steps_scale == 1: 
        scale_tests[0] = np.array([bounds_min[2], bounds_max[2]]).mean()
    else: 
        scale_tests[0] = np.linspace(bounds_min[2], bounds_max[2], n_steps_scale)

    # Calculate the relative height between init height and bounds
    relative_height = (height_tests[0] - bounds_min[0]) / (bounds_max[0] - bounds_min[0])
    # Calculate the relative mu between init mu and bounds
    relative_mu = (mu_tests[0] - bounds_min[1]) / (bounds_max[1] - bounds_min[1])

    relative_scale = (scale_tests[0] - bounds_min[2]) / (bounds_max[2] - bounds_min[2])

    # Set other inits (height and mu) regarding the first peak relatives (0.99 to take care of bounds)
    for i in range(1, n_peaks):
        height_tests[i] = bounds_min[i*3] + 0.99 * relative_height * (bounds_max[i*3] - bounds_min[i*3])
        mu_tests[i] = bounds_min[i*3+1] + 0.99 * relative_mu * (bounds_max[i*3+1] - bounds_min[i*3+1])
        scale_tests[i] = bounds_min[i*3+2] + 0.99 * relative_scale * (bounds_max[i*3+2] - bounds_min[i*3+2])

    num_parameters = len(bounds_min)

    # Create a numpy nd-array
    # First entry of rows are the redchis and the following entries are the params
    initial_guesses = np.empty((len(height_tests[0]) * len(mu_tests[0]) * len(scale_tests[0]),
                                     num_parameters + 1))

    # first index of tests arrays equals the peak index: 0 is the first peak
    # second index of tests arrays indexes the specific init guess
    index = 0
    for index_height, init_height in enumerate(height_tests[0]):
        for index_mu, init_mu in enumerate(mu_tests[0]):
            for index_scale, init_scale in enumerate(scale_tests[0]):
                initial_guess = np.empty(num_parameters)
                for i in range(n_peaks):
                    initial_guess[i*3:(i*3+3)] = [height_tests[i][index_height],
                     mu_tests[i][index_mu], scale_tests[i][index_scale]]
                initial_guess[num_parameters - 1] = np.min(intensities)

                model_y = full_fitfunction(angles, initial_guess, distribution)
                initial_guesses[index][0] = calculate_chi2(model_y, intensities, angles, 
                                intensities_err, len(initial_guess))
                initial_guesses[index][1:] = initial_guess
                index += 1

    # Sort the initial guesses starting with the lowest chi2
    initial_guesses = initial_guesses[initial_guesses[:, 0].argsort()]

    # Pick the best n_steps_fit initial_guesses
    initial_guesses = initial_guesses[:n_steps_fit]

    return initial_guesses

@njit(cache = True, fastmath = True)
def _build_design_matrix(angles, intensities, intensities_err, best_parameters, distribution):
    """
    Build the design matrix for non negative least square fitting (linear regression).

    Parameters
    ----------
    - angles: np.ndarray (n, )
        Array that stores the angles at which the intensities are measured.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The error (standard deviation) of the corresponding measured intensities of the pixel.
    - best_parameters: np.ndarray (m, )
        Array which stores the found parameters of nonlinear fitting.
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.

    Returns
    -------
    - weighted_design_matrix: np.ndarray (n, n_peaks + 1)
    - weighted_y_data: np.ndarray (n, )

    """
    # Calculate weights
    weights = 1 / intensities_err**2

    n_peaks = len(best_parameters) // 3

    design_matrix = np.zeros((len(angles),n_peaks + 1))

    for i in range(n_peaks):
            design_matrix[:, i] = distribution_pdf(angles, 
                                    best_parameters[i * 3 + 1], best_parameters[i * 3 + 2], 
                                    distribution)
        #weighted_design_matrix[:, i] = design_matrix[:, i] / np.sqrt(intensities)

    # Last column for offset
    design_matrix[:, -1] = np.ones(len(angles))

    # Apply weights to the design matrix and the y-data
    weighted_design_matrix = design_matrix * np.sqrt(weights[:, np.newaxis])
    weighted_y_data = intensities * np.sqrt(weights)

    return weighted_design_matrix, weighted_y_data
    
def fit_heights_linear(angles, intensities, intensities_err, best_parameters, distribution):
    """
    Build the design matrix for non negative least square fitting (linear regression).

    Parameters
    ----------
    - angles: np.ndarray (n, )
        Array that stores the angles at which the intensities are measured.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The error (standard deviation) of the corresponding measured intensities of the pixel.
    - best_parameters: np.ndarray (m, )
        Array which stores the found parameters of nonlinear fitting.
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.

    Returns
    -------
    - corrected_heights: np.ndarray (p, )
        The result of nonlinear least square fitting the heights.
    
    """
    
    # Apply bounded (and weighted) linear regression (linear least squares)
    weighted_design_matrix, weighted_y_data = _build_design_matrix(angles, intensities, 
                        intensities_err, best_parameters, distribution)

    corrected_heights,_ = nnls(weighted_design_matrix, weighted_y_data)
    #result = lsq_linear(weighted_design_matrix, weighted_y_data, bounds)
    #corrected_heights = result.x

    return corrected_heights

def fit_pixel_stack(angles, intensities, intensities_err, distribution = "wrapped_cauchy", 
                    n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 5, 
                    n_steps_fit = 10, fit_height_nonlinear = True, 
                    refit_steps = 1, init_fit_filter = None,
                    method = "leastsq", only_peaks_count = -1, max_peaks = 4,
                    max_peak_hwhm = 50 * np.pi/180, min_peak_hwhm = 10 * np.pi/180, 
                    mu_range = 40 * np.pi/180, scale_range = 0.4):
    """
    Fits the data of one pixel.

    Parameters
    ----------
    - angles: np.ndarray (n, )
        Array that stores the angles at which the intensities are measured.
    - intensities: np.ndarray (n, )
        The measured intensities of the pixel.
    - intensities_err: np.ndarray (n, )
        The error (standard deviation) of the corresponding measured intensities of the pixel.
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.
    - n_steps_height: int
        Number of variations in height to search for best initial guess.
    - n_steps_mu: int
        Number of variations in mu to search for best initial guess.
    - n_steps_scale: int
        Number of variations in scale to search for best initial guess.
    - n_steps_fit: int
        Number of initial guesses to pick for fitting (starting from best inital guesses).
    - fit_height_nonlinear: boolean
        Whether to include the heights in the nonlinear fitting or not.
    - refit_steps: int
        Number that defines how often the fitting process should be repeated with the result
        as new initial guess.
    - init_fit_filter: None or list
        List that defines which filter to apply before the first fit. 
        This filter will be applied on the intensities before doing anything and
        will be remove after one fit is done. Then the normal fitting process starts with
        this result as initial guess.
        First value of the list is a string with:
        "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
        The following one to two values are the params for this filter (scipy docs).
    - method: string
        Defines which fitting method to use. Can be anything from the methods in lmfit.minimize
        and additionally "biteopt".
        Full list: 
            leastsq: Levenberg-Marquardt (default)
            least_squares: Least-Squares minimization, using Trust Region Reflective method
            differential_evolution: differential evolution
            brute: brute force method
            basinhopping: basinhopping
            ampgo: Adaptive Memory Programming for Global Optimization
            nelder: Nelder-Mead
            lbfgsb: L-BFGS-B
            powell: Powell
            cg: Conjugate-Gradient
            newton: Newton-CG
            cobyla: Cobyla
            bfgs: BFGS
            tnc: Truncated Newton
            trust-ncg: Newton-CG trust-region
            trust-exact: nearly exact trust-region
            trust-krylov: Newton GLTR trust-region
            trust-constr: trust-region for constrained optimization
            dogleg: Dog-leg trust-region
            slsqp: Sequential Linear Squares Programming
            emcee: Maximum likelihood via Monte-Carlo Markov Chain
            shgo: Simplicial Homology Global Optimization
            dual_annealing: Dual Annealing optimization
            biteopt: Derivative-Free Global Optimization Method
    - only_peaks_count: int
        Defines a filter for found peaks, so that if the count of found peaks is not equal that number,
        the function return the same as if no peaks are found.
    - max_peaks: int
        Defines the maximum number of peaks that should be returned from the 
        total found peaks (starting from highest peak).
    - max_peak_hwhm: float
        Estimated maximum peak half width at half maximum.
    - min_peak_hwhm: float
        Estimated minimum peak half width at half maximum.
    - mu_range: float
        Range of mu (regarding estimated maximum and minimum bounds around true mu).
    - scale_range: float
        Range of scale (regarding estimated maximum and minimum bounds around true scale).

    Returns
    -------
    - best_parameters: np.ndarray (m, )
        Array which stores the best found parameters.
    - best_redchi: float
        Calculated Chi2 of the model with the found parameters and given data.
    - peaks_mask: np.ndarray (n_peaks, n)
        Array that stores the indices of the measurements that corresponds (mainly) to a peak.
    
    """

    # Ensure no overflow in (subtract) operations happen:
    intensities = intensities.astype(np.int32)

    min_int = np.min(intensities)
    max_int = np.max(intensities)
    global_amplitude = max_int - min_int
    
    if init_fit_filter:
        original_intensities = intensities
        original_intensities_err = intensities_err
        intensities = apply_filter(intensities, init_fit_filter)
        intensities_err = apply_filter(intensities_err, init_fit_filter)
        refit_steps += 1

    peaks_mask, peaks_mus = find_peaks(angles, intensities, intensities_err, 
                    only_peaks_count = only_peaks_count, max_peaks = max_peaks,
                    max_peak_hwhm = max_peak_hwhm, min_peak_hwhm = min_peak_hwhm, 
                    mu_range = mu_range, scale_range = scale_range)

    bounds_min, bounds_max = create_bounds(angles, intensities, intensities_err, distribution,
                    peaks_mask, peaks_mus, mu_range, scale_range,
                    min_peak_hwhm, max_peak_hwhm, global_amplitude, min_int)

    # if no peaks found return params of zeros, np.nan as chi2, zeros as peaks_mask
    if np.all(bounds_max == 0):
        return np.zeros(4), np.nan, np.zeros((1, len(angles)), dtype = np.bool_)

    if method == "biteopt":
        if not fit_height_nonlinear:
            bounds = Bounds(np.delete(bounds_min, np.s_[0::3]), np.delete(bounds_max, np.s_[0::3]))
        else:
            bounds = Bounds(bounds_min, bounds_max)

    initial_guesses = create_init_guesses(angles, intensities, intensities_err, bounds_min, bounds_max,
                                            distribution, n_steps_height, n_steps_mu, 
                                            n_steps_scale, n_steps_fit)

    num_parameters = len(bounds_min)

    best_parameters = np.empty(num_parameters)
    best_redchi = None

    try:
        
        for init_guess in initial_guesses[:, 1:]:

            if method == "biteopt":
                if not fit_height_nonlinear:
                    heights = init_guess[0:-1:3]
                    offset = init_guess[-1]
                    init_guess = np.delete(init_guess, np.s_[0::3])
                def biteopt_objective(params):
                    if not fit_height_nonlinear:
                        params = np.insert(params, range(0, len(params), 2), heights)
                        params = np.append(params, offset)
                    
                    return _objective(params, angles, intensities, intensities_err, distribution)
                
                result = bitecpp.minimize(biteopt_objective, bounds = bounds, x0 = init_guess, max_evaluations = 10000)
                
                if not fit_height_nonlinear:
                    result.x = np.insert(result.x, range(0, len(result.x), 2), heights)
                    result.x = np.append(result.x, offset)

                model_y = full_fitfunction(angles, result.x, distribution)
                redchi = calculate_chi2(model_y, intensities, angles, intensities_err, len(result.x))
            else:

                wcm_model = Model(_lmfit_fitfunction)

                # Create params
                params = wcm_model.make_params()
                for i, guess in enumerate(init_guess):
                    vary = fit_height_nonlinear or not i in [0,3,6,9,12]
                    # Make first parameter name to distribution name
                    if i == 0:
                        name = distribution
                    else:
                        name = f'p{i}'
                    params.add(name = name, value = guess, 
                                    min = bounds_min[i], max = bounds_max[i], vary = vary)

                result = wcm_model.fit(intensities, params, x=angles, 
                    weights = 1 / intensities_err, method = method)
                redchi = result.redchi
                
            if not best_redchi or (redchi < best_redchi and redchi > 1):
                best_redchi = redchi
                if method == "biteopt":
                    best_parameters = result.x
                else:
                    best_parameters = np.array([result.params[key].value for key in result.params])
    except RuntimeError as err:
        print(err)
        if (index_height == len(height_tests[0]) - 1 and index_mu == len(mu_tests[0])
            and index_scale == len(scale_tests[0]) - 1) and not best_redchi:
            print("ATTENTION: Optimal parameters for pixel not found!")
        pass

    corrected_heights = fit_heights_linear(angles, intensities, intensities_err, 
                                            best_parameters, distribution)
    # Replace heights with corrected heights
    best_parameters[0::3] = corrected_heights

    # Maybe To-Do Idea: If chi2 of peak is bad, add one peak to that peak and fit again
    
    # Refit mu and scale with updated heights (not refitting height leads to better results)
    best_heights = best_parameters[0:-1:3]
    zero_heights = np.asarray(best_heights < 1).nonzero()[0]

    # if first fit was filtered, now fit original data
    if init_fit_filter:
        intensities = original_intensities
        intensities_err = original_intensities_err

    if method == "biteopt" and fit_height_nonlinear:
        bounds = Bounds(np.delete(bounds_min, np.s_[0::3]), np.delete(bounds_max, np.s_[0::3]))
    for r in range(refit_steps):
        if method == "biteopt":
            heights = best_parameters[0:-1:3]
            offset = best_parameters[-1]

            def biteopt_objective(params):
                params = np.insert(params, range(0, len(params), 2), heights)
                params = np.append(params, offset)
                
                return _objective(params, angles, intensities, intensities_err, distribution)
            init_guess = np.delete(best_parameters, np.s_[0::3])
            result = bitecpp.minimize(biteopt_objective, bounds, x0 = init_guess, max_evaluations = 10000)
            result.x = np.insert(result.x, range(0, len(result.x), 2), heights)
            result.x = np.append(result.x, offset)

            best_parameters = result.x
        else:
            for i, param in enumerate(best_parameters):
                vary = not i in [0,3,6,9,12]
                if i == 0:
                    name = distribution
                else:
                    name = f'p{i}'
                # If the height of the peak is zero, dont fit the peak
                if i in zero_heights or i-1 in zero_heights or i-2 in zero_heights:
                    vary = False
                if i < num_parameters:
                    result.params[name].value = param
                    result.params[name].vary = vary
                else:
                    result.params.add(name = name, value = param, 
                                    min = bounds_min[i], max = bounds_max[i], vary = vary)
            result = wcm_model.fit(intensities, result.params, x=angles, weights = 1 / intensities_err)
            
            best_parameters = np.array([result.params[key].value for key in result.params])

        corrected_heights = fit_heights_linear(angles, intensities, intensities_err, 
                                                best_parameters, distribution)
        best_parameters[0::3] = corrected_heights
    
    model_y = full_fitfunction(angles, best_parameters, distribution)
    best_redchi = calculate_chi2(model_y, intensities, angles, intensities_err, len(best_parameters))

    return best_parameters, best_redchi, peaks_mask

def _format_time(hours, minutes, seconds):
    """
    A function that formats time based on the input hours, minutes, and seconds.

    Parameters:
    - hours: int
        The number of hours.
    - minutes: int
        The number of minutes.
    - seconds: int
        The number of seconds.

    Returns:
    - str
        A formatted string representing the time in hours, minutes, and seconds.
    """
    parts = []
    if hours > 0:
        parts.append(f"{int(hours)} hours")
    if minutes > 0:
        parts.append(f"{int(minutes)} minutes")
    if seconds > 0:
        parts.append(f"{int(seconds)} seconds")
    return ", ".join(parts) if parts else "0 seconds"

def fit_image_stack(image_stack, distribution = "wrapped_cauchy", fit_height_nonlinear = True,
                    threshold = 1000,
                    n_steps_height = 10, n_steps_mu = 10, n_steps_scale = 5,
                        n_steps_fit = 10, refit_steps = 1,
                        init_fit_filter = None, method = "leastsq", 
                        only_peaks_count = -1, max_peaks = 4,
                        max_peak_hwhm = 50 * np.pi/180, min_peak_hwhm = 10 * np.pi/180, 
                        mu_range = 40 * np.pi/180, scale_range = 0.4,
                        num_processes = 2):
    """
    Fits the data (of a full image stack).

    Parameters
    ----------
    - image_stack: np.ndarray (n, m, p)
        Array that stores the p intensity measurements for every pixel in the (n*m sized) image
    - distribution: "wrapped_cauchy", "von_mises", or "wrapped_laplace"
        The name of the distribution.
    - fit_height_nonlinear: boolean
        Whether to include the heights in the nonlinear fitting or not.
    - threshold: int
        Threshold value. If the mean intensity of one pixel is lower than that threshold value,
        the pixel will not be evaluated.
    - n_steps_height: int
        Number of variations in height to search for best initial guess.
    - n_steps_mu: int
        Number of variations in mu to search for best initial guess.
    - n_steps_scale: int
        Number of variations in scale to search for best initial guess.
    - n_steps_fit: int
        Number of initial guesses to pick for fitting (starting from best inital guesses).
    - refit_steps: int
        Number that defines how often the fitting process should be repeated with the result
        as new initial guess.
    - init_fit_filter: None or list
        List that defines which filter to apply before the first fit. 
        This filter will be applied on the intensities before doing anything and
        will be removed after one fit is done. Then the normal fitting process starts with
        this result as initial guess.
        First value of the list is a string with:
        "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
        The following one to two values are the params for this filter (scipy docs).
    - method: string
        Defines which fitting method to use. Can be anything from the methods in lmfit.minimize
        and additionally "biteopt".
        Full list: 
            leastsq: Levenberg-Marquardt (default)
            least_squares: Least-Squares minimization, using Trust Region Reflective method
            differential_evolution: differential evolution
            brute: brute force method
            basinhopping: basinhopping
            ampgo: Adaptive Memory Programming for Global Optimization
            nelder: Nelder-Mead
            lbfgsb: L-BFGS-B
            powell: Powell
            cg: Conjugate-Gradient
            newton: Newton-CG
            cobyla: Cobyla
            bfgs: BFGS
            tnc: Truncated Newton
            trust-ncg: Newton-CG trust-region
            trust-exact: nearly exact trust-region
            trust-krylov: Newton GLTR trust-region
            trust-constr: trust-region for constrained optimization
            dogleg: Dog-leg trust-region
            slsqp: Sequential Linear Squares Programming
            emcee: Maximum likelihood via Monte-Carlo Markov Chain
            shgo: Simplicial Homology Global Optimization
            dual_annealing: Dual Annealing optimization
            biteopt: Derivative-Free Global Optimization Method
    - only_peaks_count: int
        Defines a filter for found peaks, so that if the count of found peaks is not equal that number,
        the function returns the same as if no peaks are found.
    - max_peaks: int
        Defines the maximum number of peaks that should be returned from the 
        total found peaks (starting from highest peak).
    - max_peak_hwhm: float
        Estimated maximum peak half width at half maximum.
    - min_peak_hwhm: float
        Estimated minimum peak half width at half maximum.
    - mu_range: float
        Range of mu (regarding estimated maximum and minimum bounds around true mu).
    - scale_range: float
        Range of scale (regarding estimated maximum and minimum bounds around true scale).
    - num_processes: int
        Number that defines in how many sub-processes the fitting process should be split into.

    Returns
    -------
    - deflattened_params: np.ndarray (n, m, q)
        Array which stores the best found parameters for every pixel (of n*m pixels).
    - deflattened_peaks_mask: np.ndarray (n, m, max_peaks, p)
        Array that stores the indices of the measurements that corresponds (mainly) to a peak,
        for every pixel (of n*m pixels).
    """

    total_pixels = image_stack.shape[0]*image_stack.shape[1]
    flattened_stack = image_stack.reshape((total_pixels, image_stack.shape[2]))
    mask = np.mean(flattened_stack, axis = 1) > threshold
    mask_pixels, = mask.nonzero()
    num_params = 3 * max_peaks + 1
    output_params = pymp.shared.array((flattened_stack.shape[0], num_params))
    output_peaks_mask = pymp.shared.array((flattened_stack.shape[0], max_peaks, image_stack.shape[2]), dtype=np.bool_)
    goals = np.array([len(mask_pixels) * (i+1)/num_processes for i in range(num_processes)])
    statuses = pymp.shared.array(num_processes)
    start_time = pymp.shared.array((1,))
    process_started = pymp.shared.array((1,), dtype=np.bool_)
    elapsed_time = pymp.shared.array((1,))
    start_time[0] = 0
    process_started[0] = False
    elapsed_time[0] = 0

    with pymp.Parallel(num_processes) as p:
        for i in p.range(len(mask_pixels)):
            pixel = mask_pixels[i]
            if p.thread_num == 0:
                current_index = i
            else:
                current_index = int(i - goals[p.thread_num - 1])
            statuses[p.thread_num] = round(100 * current_index / int(len(mask_pixels)/num_processes), 2)
            overall_progress = np.mean(statuses)
            current_time = time() - start_time[0]
            if not process_started[0] and p.thread_num == 0:
                p.print("Compiling numba functions (can take up to 5 minutes)...")
            elif process_started[0] and current_time - elapsed_time[0] >= 5:
                # If last printed out status is over 5 seconds ago, print current status
                with pymp.shared.lock():
                    if current_time - elapsed_time[0] >= 5:
                        elapsed_time[0] = current_time
                        estimated_total_time = elapsed_time[0] * 100 / overall_progress if overall_progress > 0 else 0
                        remaining_time = estimated_total_time - elapsed_time[0]
                        remaining_hours, remaining_seconds = divmod(remaining_time, 3600)
                        remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
                        elapsed_hours, elapsed_seconds = divmod(elapsed_time[0], 3600)
                        elapsed_minutes, elapsed_seconds = divmod(elapsed_seconds, 60)
                        p.print("______________________________________")
                        p.print(f"Overall Progress: {overall_progress:.2f}%")
                        p.print(f"Elapsed Time: {_format_time(elapsed_hours, elapsed_minutes, elapsed_seconds)}")
                        p.print(f"Estimated Remaining Time: {_format_time(remaining_hours,
                                                                remaining_minutes, remaining_seconds)}")

            intensities = flattened_stack[pixel][0:image_stack.shape[2]]
            intensities_err = np.sqrt(intensities)
            angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)
            best_parameters, best_r_chi2, peaks_mask = fit_pixel_stack(angles, intensities, 
                                    intensities_err, 
                                    distribution, n_steps_height, n_steps_mu, n_steps_scale,
                                    fit_height_nonlinear = fit_height_nonlinear, refit_steps = refit_steps,
                                    n_steps_fit = n_steps_fit, init_fit_filter = init_fit_filter, 
                                    method = method, only_peaks_count = only_peaks_count, 
                                    max_peaks = max_peaks,
                                    max_peak_hwhm = max_peak_hwhm, min_peak_hwhm = min_peak_hwhm, 
                                    mu_range = mu_range, scale_range = scale_range)
            output_params[pixel][0:len(best_parameters)-1] = best_parameters[:-1]
            output_params[pixel][-1] = best_parameters[-1]
            output_peaks_mask[pixel][0:len(peaks_mask)] = peaks_mask

            # Start the timer after the first iteration
            if not process_started[0]:
                with pymp.shared.lock():
                    if not process_started[0]:  # Double-check inside the lock
                        start_time[0] = time()
                        process_started[0] = True
                        p.print("Process started")

    print("______________________________________")
    print("Process finished")
    elapsed_time[0] = time() - start_time[0]
    elapsed_hours, elapsed_seconds = divmod(elapsed_time[0], 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_seconds, 60)
    print(f"Processing Time: {_format_time(elapsed_hours, elapsed_minutes, elapsed_seconds)}")

    deflattened_params = output_params.reshape((image_stack.shape[0], 
                                            image_stack.shape[1], output_params.shape[1]))
    deflattened_peaks_mask = output_peaks_mask.reshape((image_stack.shape[0], 
                                            image_stack.shape[1], output_peaks_mask.shape[1], 
                                            output_peaks_mask.shape[2]))
    return deflattened_params, deflattened_peaks_mask

def find_image_peaks(image_stack, threshold = 1000, init_fit_filter = None, 
                        only_peaks_count = -1, max_peaks = 4,
                        max_peak_hwhm = 50 * np.pi/180, min_peak_hwhm = 10 * np.pi/180, 
                        mu_range = 40 * np.pi/180, scale_range = 0.4,
                        num_processes = 2):
    """
    Finds the peaks of an image stack using only the peak finder.

    Parameters
    ----------
    - image_stack: np.ndarray (n, m, p)
        Array that stores the p intensity measurements for every pixel in the (n*m sized) image
    - threshold: int
        Threshold value. If the mean intensity of one pixel is lower than that threshold value,
        the pixel will not be evaluated.
    - init_fit_filter: None or list
        List that defines which filter to apply before the first fit. 
        This filter will be applied on the intensities before doing anything and
        will be remove after one fit is done. Then the normal fitting process starts with
        this result as initial guess.
        First value of the list is a string with:
        "fourier", "gauss", "uniform", "median", "moving_average", or "savgol".
        The following one to two values are the params for this filter (scipy docs).
    - only_peaks_count: int
        Defines a filter for found peaks, so that if the count of found peaks is not equal that number,
        the function return the same as if no peaks are found.
    - max_peaks: int
        Defines the maximum number of peaks that should be returned from the 
        total found peaks (starting from highest peak).
    - max_peak_hwhm: float
        Estimated maximum peak half width at half maximum.
    - min_peak_hwhm: float
        Estimated minimum peak half width at half maximum.
    - mu_range: float
        Range of mu (regarding estimated maximum and minimum bounds around true mu).
    - scale_range: float
        Range of scale (regarding estimated maximum and minimum bounds around true scale).
    - num_processes: int
        Number that defines in how many sub-processes the fitting process should be split into.

    Returns
    -------
    - deflattened_params: np.ndarray (n, m, q)
        Array which stores the best found parameters for every pixel (of n*m pixels).
    - deflattened_peaks_mask: np.ndarray (n, m, max_peaks, p)
        Array that stores the indices of the measurements that corresponds (mainly) to a peak,
        for every pixel (of n*m pixels).
    """

    total_pixels = image_stack.shape[0]*image_stack.shape[1]
    flattened_stack = image_stack.reshape((total_pixels, image_stack.shape[2]))
    mask = np.mean(flattened_stack, axis = 1) > threshold
    mask_pixels, = mask.nonzero()
    output_peaks_mus = pymp.shared.array((flattened_stack.shape[0], max_peaks))
    output_peaks_mask = pymp.shared.array((flattened_stack.shape[0], max_peaks, image_stack.shape[2]), dtype=np.bool_)
    goals = np.array([len(mask_pixels) * (i+1)/num_processes for i in range(num_processes)])
    statuses = pymp.shared.array(num_processes)
    start_time = pymp.shared.array((1,))
    process_started = pymp.shared.array((1,), dtype=np.bool_)
    elapsed_time = pymp.shared.array((1,))
    start_time[0] = 0
    process_started[0] = False
    elapsed_time[0] = 0

    with pymp.Parallel(num_processes) as p:
        for i in p.range(len(mask_pixels)):
            pixel = mask_pixels[i]
            if p.thread_num == 0:
                current_index = i
            else:
                current_index = int(i - goals[p.thread_num - 1])

            statuses[p.thread_num] = round(100 * current_index / int(len(mask_pixels)/num_processes), 2)
            overall_progress = np.mean(statuses)
            current_time = time() - start_time[0]
            if not process_started[0] and p.thread_num == 0:
                p.print("Compiling numba functions (can take up to 5 minutes)...")
            elif process_started[0] and current_time - elapsed_time[0] >= 5:
                # If last printed out status is over 5 seconds ago, print current status
                with pymp.shared.lock():
                    if current_time - elapsed_time[0] >= 5:
                        elapsed_time[0] = current_time
                        estimated_total_time = elapsed_time[0] * 100 / overall_progress if overall_progress > 0 else 0
                        remaining_time = estimated_total_time - elapsed_time[0]
                        remaining_hours, remaining_seconds = divmod(remaining_time, 3600)
                        remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
                        elapsed_hours, elapsed_seconds = divmod(elapsed_time[0], 3600)
                        elapsed_minutes, elapsed_seconds = divmod(elapsed_seconds, 60)
                        p.print("______________________________________")
                        p.print(f"Overall Progress: {overall_progress:.2f}%")
                        p.print(f"Elapsed Time: {_format_time(elapsed_hours, elapsed_minutes, elapsed_seconds)}")
                        p.print(f"Estimated Remaining Time: {_format_time(remaining_hours,
                                                                remaining_minutes, remaining_seconds)}")

            intensities = flattened_stack[pixel][0:image_stack.shape[2]]
            intensities_err = np.sqrt(intensities)
            angles = np.linspace(0, 2*np.pi, num=len(intensities), endpoint=False)

            peaks_mask, peaks_mus = find_peaks(angles, intensities, intensities_err, 
                            only_peaks_count = only_peaks_count, max_peaks = max_peaks,
                            max_peak_hwhm = max_peak_hwhm, min_peak_hwhm = min_peak_hwhm, 
                            mu_range = mu_range, scale_range = scale_range)

            output_peaks_mask[pixel][0:len(peaks_mask)] = peaks_mask
            output_peaks_mus[pixel][0:len(peaks_mus)] = peaks_mus

            # Start the timer after the first iteration
            if not process_started[0]:
                with pymp.shared.lock():
                    if not process_started[0]:  # Double-check inside the lock
                        start_time[0] = time()
                        process_started[0] = True
                        p.print("Process started")

    print("______________________________________")
    print("Process finished")
    elapsed_time[0] = time() - start_time[0]
    elapsed_hours, elapsed_seconds = divmod(elapsed_time[0], 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_seconds, 60)
    print(f"Processing Time: {_format_time(elapsed_hours, elapsed_minutes, elapsed_seconds)}")

    deflattened_peaks_mask = output_peaks_mask.reshape((image_stack.shape[0], 
                                            image_stack.shape[1], output_peaks_mask.shape[1], 
                                            output_peaks_mask.shape[2]))
    deflattened_peaks_mus = output_peaks_mus.reshape((image_stack.shape[0], 
                                            image_stack.shape[1], output_peaks_mus.shape[1]))

    return deflattened_peaks_mask, deflattened_peaks_mus
