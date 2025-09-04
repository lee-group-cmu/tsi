import numpy as np
import pandas as pd
import os
import sys
from typing import List, Dict, Union, Optional
import warnings

import torch
import alphashape

from torch.distributions import Distribution, Independent, Uniform

from sbi.utils import get_kde, process_prior
from sbi.inference import FMPE, NPE, NPSE

from scipy.spatial import cKDTree

from gaiaxpy.core.config import load_xpmerge_from_xml, load_xpsampling_from_xml

from fmpes import CustomFMPE

# NOTE: VSI = Valid Scientific Inference (now TSI = Trustworthy Scientific Inference)

def setup_paths(notebook: bool = False) -> Dict[str, str]:
    """
    Create dictionary of file paths for data, output, and figures.

    Parameters
    ----------
    notebook : bool
        Set path for use in Jupyter notebook or script.

    Returns
    -------
     : dict
        Dictionary of file paths for data, out, assets, and figures
    """
    # requires "../milkyjay/folder/file_with_call.py" to work
    if notebook == False:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    elif notebook == True:
        base_path = os.path.dirname(os.getcwd())
    return {
        "data": os.path.join(base_path, "data"),
        "processed_data": os.path.join(base_path, "processed_data"),
        "assets": os.path.join(base_path, "assets"),
        "figures": os.path.join(base_path, "figures")
    }

def get_eval_grid(y_eval: Union[np.ndarray, torch.Tensor],
                  evaluation_grid_size: int,
                  method: str = "cartesian",
                  alpha: float = 0.01,
                  subsample_size: Optional[int] = None,
                  random_seed: int = 42) -> np.ndarray:
    """
    Generate an evaluation grid within the region defined by the input data.
    
    The grid is generated so that its points lie within the concave hull (alpha shape)
    of the provided evaluation data. Two methods are available:
    
        - "cartesian": A regular Cartesian grid is generated over the bounding box
                       of the data and then filtered using the alpha shape. If the
                       resulting number of points exceeds evaluation_grid_size, a random
                       subset is selected; if it is too few, additional random points are
                       added until the target is met.
        - "random": Points are sampled uniformly at random within the bounding box until
                    evaluation_grid_size points inside the alpha shape are obtained.
    
    Parameters
    ----------
    y_eval : Union[np.ndarray, torch.Tensor]
        2D array (or tensor) of evaluation data points (e.g., features) that define the region.
        Shape: (n_samples, n_features)
    evaluation_grid_size : int
        Desired number of grid points.
    method : str, optional
        Method to generate the grid. Options: "cartesian" or "random". Default is "cartesian".
    alpha : float, optional
        Alpha parameter for the alpha shape algorithm determining the concavity of the shape.
        Default is 0.01.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    np.ndarray
        Array of shape (evaluation_grid_size, n_features) containing the grid points.
    """
    # Ensure input is a numpy array
    if isinstance(y_eval, torch.Tensor):
        y_eval = y_eval.numpy()
    y_eval = np.asarray(y_eval)
    
    # assist with runtime
    if subsample_size is not None and y_eval.shape[0] > subsample_size:
        indices = np.random.choice(y_eval.shape[0], subsample_size, replace=False)
        y_eval = y_eval[indices]
    
    # Determine the number of parameters (dimensions)
    poi_dim = y_eval.shape[1]
    
    # Compute the alpha shape from the evaluation data
    alpha_shape_obj = alphashape.alphashape(y_eval, alpha)
    
    # Calculate the bounding box for the evaluation data
    mins = np.min(y_eval, axis=0)
    maxs = np.max(y_eval, axis=0)
    
    grid_points = []  # will store accepted points
    
    if method.lower() == "cartesian":
        # Determine grid resolution along each axis
        grid_size_per_axis = int(np.ceil(evaluation_grid_size ** (1/poi_dim)))
        # Create equally spaced values for each dimension
        grids = [np.linspace(mins[d], maxs[d], grid_size_per_axis) for d in range(poi_dim)]
        # Generate the Cartesian product via meshgrid
        mesh = np.meshgrid(*grids)
        # Flatten the meshgrid into a list of candidate points
        candidate_points = np.vstack([m.flatten() for m in mesh]).T
        
        # Filter candidates: accept points inside the alpha shape
        for pt in candidate_points:
            candidate = pt.reshape(1, -1)  # reshape to mimic the old code’s format
            if alpha_shape_obj.contains(candidate):
                grid_points.append(pt.tolist())
        grid_points = np.array(grid_points)
        
        # Adjust the grid size:
        np.random.seed(random_seed)
        if grid_points.shape[0] > evaluation_grid_size:
            # If too many, randomly select the required number of points
            indices = np.random.choice(grid_points.shape[0], evaluation_grid_size, replace=False)
            grid_points = grid_points[indices]
        elif grid_points.shape[0] < evaluation_grid_size:
            # If too few, supplement with additional random points
            additional_points = []
            while grid_points.shape[0] + len(additional_points) < evaluation_grid_size:
                rand_pt = np.array([np.random.uniform(mins[d], maxs[d]) for d in range(poi_dim)])
                candidate = rand_pt.reshape(1, -1)
                if alpha_shape_obj.contains(candidate):
                    additional_points.append(rand_pt.tolist())
            if additional_points:
                additional_points = np.array(additional_points)
                grid_points = np.concatenate([grid_points, additional_points], axis=0)
    
    elif method.lower() == "random":
        np.random.seed(random_seed)
        # Continuously sample random points until the desired count is reached
        while len(grid_points) < evaluation_grid_size:
            rand_pt = np.array([np.random.uniform(mins[d], maxs[d]) for d in range(poi_dim)])
            candidate = rand_pt.reshape(1, -1)
            if alpha_shape_obj.contains(candidate):
                grid_points.append(rand_pt.tolist())
        grid_points = np.array(grid_points)
    
    else:
        raise ValueError("Method must be either 'cartesian' or 'random'.")
    
    return grid_points

def get_prior(prior_samples: Union[np.ndarray, torch.Tensor] = None, 
              prior_method: Union[str, callable] = "kde", 
              prior_args: dict = None, 
              **kwargs) -> Distribution:
    """
    Construct a prior distribution using various methods.

    Parameters
    ----------
    prior_samples : np.ndarray or torch.Tensor, optional
        Samples to be used for constructing the prior (e.g., training label data).
        If a torch.Tensor is provided, it will be converted to a NumPy array.
    prior_method : str or callable, optional
        The method for constructing the prior. Supported options:
          - "kde": Uses a kernel density estimator.
          - "uniform": Constructs an independent uniform prior over a specified box.
          - callable: A custom function that takes (prior_samples, prior_args) and returns a Distribution.
    prior_args : dict, optional
        Additional keyword arguments for the chosen method.
        For "kde", you can pass e.g. {"bandwidth": "silvermann"}.
        For "uniform", you should provide "low" and "high" bounds (as arrays or scalars).
    **kwargs : dict
        Extra keyword arguments passed to the prior constructor (allows for future updates).

    Returns
    -------
    Distribution
        A torch Distribution object representing the prior.

    Raises
    ------
    ValueError
        If required parameters for the chosen method are missing.
    TypeError
        If prior_method is neither a string nor a callable.
    """
    
    if prior_args is None:
        prior_args = {}
        
    # Convert torch.Tensor input to a numpy array.
    if isinstance(prior_samples, np.ndarray):
        prior_samples = torch.from_numpy(prior_samples)
    
    # If a custom callable is provided, use it.
    if callable(prior_method):
        return prior_method(prior_samples, prior_args)
    
    if isinstance(prior_method, str):
        method = prior_method.lower()
        
        if method == "kde":
            bandwidth = prior_args.get("bandwidth", "scott") # Change default here
            if prior_samples is None:
                raise ValueError("prior_samples must be provided for KDE prior method.")
            density_estimator = get_kde(samples=prior_samples, bandwidth=bandwidth)
            return process_prior(density_estimator, custom_prior_wrapper_kwargs=prior_args)[0]
        
        elif method == "uniform":
            # Use provided bounds or infer from prior_samples if available.
            if "low" in prior_args and "high" in prior_args:
                low = np.array(prior_args["low"])
                high = np.array(prior_args["high"])
            elif prior_samples is not None:
                low = np.min(prior_samples, axis=0)
                high = np.max(prior_samples, axis=0)
            else:
                raise ValueError("For uniform prior, 'low' and 'high' must be provided in prior_args if no prior_samples is given.")
            
            # Convert bounds to tensors.
            low_tensor = torch.tensor(low, dtype=torch.float32)
            high_tensor = torch.tensor(high, dtype=torch.float32)
            
            base_uniform = Uniform(low_tensor, high_tensor)
            # Wrap in Independent distribution if bounds are provided as 1D arrays.
            if low_tensor.ndim == 1:
                prior = Independent(base_uniform, 1)
            else:
                prior = base_uniform
            return prior
        
        else:
            raise ValueError(f"Unsupported prior method: {prior_method}. Use 'kde', 'uniform', or provide a callable.")
    else:
        raise TypeError("prior_method must be either a string or a callable.")

def get_estimator(parameters: torch.Tensor, 
                  samples: torch.Tensor, 
                  prior: Distribution, 
                  estimator_type: str = "fmpe", 
                  **kwargs) -> Distribution:
    """
    Construct and train an estimator (posterior) based on the provided training data,
    prior distribution, and estimator type.

    Parameters
    ----------
    parameters : torch.Tensor
        The training parameters (e.g., labels) with shape (n_samples, n_parameters).
    samples : torch.Tensor
        The training input data (e.g., spectra) with shape (n_samples, n_features).
    prior : torch.distributions.Distribution
        A prior distribution over the parameters.
    estimator_type : str, optional
        The type of estimator to use. Supported options (case-insensitive):
          - "fmpe": Uses an SBI FMPE estimator.
          - "npe": Uses an SBI NPE estimator.
          - "npse": Uses an SBI NPSE estimator.
          - "custom_fmpe": Uses a custom flowmatching estimator.
          - "dingo_fmpe": Uses the DingoFMPE estimator (not implemented).
    **kwargs : dict
        Additional keyword arguments to configure the estimator training.
        For example, for "custom_fmpe" you might pass training_batch_size, learning_rate,
        max_num_epochs, validation_fraction, etc.

    Returns
    -------
    torch.distributions.Distribution
        A posterior distribution built by the chosen estimator.
    
    Raises
    ------
    ValueError
        If the provided estimator_type is not implemented.
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator_type = estimator_type.lower()

    if estimator_type == "fmpe":
        # Using SBI FMPE
        trainer = FMPE(prior=prior, device=DEVICE)
        trainer.append_simulations(parameters, samples)
        trainer.train(**kwargs)
        posterior = trainer.build_posterior()
        return posterior

    elif estimator_type == "npe":
        # Using SBI NPE
        inference = NPE()
        posterior_estimator = inference.append_simulations(parameters, samples).train(**kwargs)
        posterior = inference.build_posterior(posterior_estimator)
        return posterior

    elif estimator_type == "npse":
        # Using SBI NPSE with default sde_type "ve" (can be overridden via kwargs)
        inference = NPSE(sde_type=kwargs.pop("sde_type", "ve"))
        inference.append_simulations(parameters, samples)
        score_estimator = inference.train(**kwargs)
        posterior = inference.build_posterior(score_estimator)
        return posterior

    elif estimator_type == "custom_fmpe":
        # Using a custom FMPE implementation (e.g., credited to Luca)
        custom_fmpe = CustomFMPE(prior=prior, device=DEVICE)
        custom_fmpe.append_simulations(parameters, samples)
        custom_fmpe.train(**kwargs)
        posterior = custom_fmpe.build_posterior()
        return posterior

    elif estimator_type == "dingo_fmpe":
        # IMPLEMENT LATER
        raise NotImplementedError("dingo_fmpe estimator is not implemented yet.")
    
    else:
        raise ValueError("Estimator type not implemented. Choose from 'fmpe', 'npe', 'npse', 'custom_fmpe', or 'dingo_fmpe'.")
    
def create_test_statistic(test_statistic: str,
                          prior,
                          estimator_arg,
                          poi_dim: int,
                          evaluation_grid=None,
                          **kwargs):
    """
    Create and return an LF2I test statistic object based on the provided parameters.

    Parameters
    ----------
    test_statistic : str
        Type of test statistic to construct. Options:
          - "PosteriorPriorRatio"
          - "Posterior"
          - "Waldo"
          - "MAP"
    prior : torch.distributions.Distribution
        The prior distribution.
    estimator_arg : any
        Either a trained estimator or an estimator type string.
    poi_dim : int
        Dimension of the parameters of interest.
    evaluation_grid : array-like, optional
        Evaluation grid to be used (required for MAP).
    **kwargs : dict
        Additional keyword arguments for configuring the test statistic:
          - estimator_kwargs: dict of kwargs for the estimator.
          - norm_posterior_samples: normalization parameter (if applicable).
          - n_jobs: number of jobs (default: -2).
          - num_posterior_samples: number of posterior samples (default values differ by type).
          - estimation_method: for Waldo (default "posterior").
          - param_grid: for MAP (defaults to evaluation_grid).

    Returns
    -------
    Object
        The constructed LF2I test statistic.
    """
    # Pop common kwargs with defaults.
    n_jobs = kwargs.pop("n_jobs", -2)
    estimator_kwargs = kwargs.pop("estimator_kwargs", {})
    
    ts_type = test_statistic.lower()
    
    # BE CAREFUL: ISSUES WITH NORM_POSTERIOR_SAMPLES
    if ts_type == "posteriorpriorratio":
        from lf2i.test_statistics.posterior import PosteriorPriorRatio
        norm_posterior_samples = kwargs.pop("norm_posterior_samples", False)
        ts = PosteriorPriorRatio(
            prior=prior,
            estimator=estimator_arg,
            estimator_kwargs=estimator_kwargs,
            poi_dim=poi_dim,
            norm_posterior=norm_posterior_samples,
            n_jobs=n_jobs
        )
    elif ts_type == "posterior":
        from lf2i.test_statistics.posterior import Posterior
        ts = Posterior(
            estimator=estimator_arg,
            estimator_kwargs=estimator_kwargs,
            poi_dim=poi_dim,
            n_jobs=n_jobs
        )
    elif ts_type == "waldo":
        from lf2i.test_statistics.waldo import Waldo
        num_posterior_samples = kwargs.pop("num_posterior_samples", 50_000)
        estimation_method = kwargs.pop("estimation_method", "posterior")
        ts = Waldo(
            estimation_method=estimation_method,
            estimator=estimator_arg,
            estimator_kwargs=estimator_kwargs,
            poi_dim=poi_dim,
            num_posterior_samples=num_posterior_samples,
            n_jobs=n_jobs
        )
    elif ts_type == "map":
        from lf2i.test_statistics.map import MAP
        num_posterior_samples = kwargs.pop("num_posterior_samples", 20_000)
        param_grid = kwargs.pop("param_grid", evaluation_grid)
        ts = MAP(
            estimator=estimator_arg,
            poi_dim=poi_dim,
            num_posterior_samples=num_posterior_samples,
            param_grid=param_grid,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("Invalid test statistic type. Choose from 'PosteriorPriorRatio', 'Posterior', 'Waldo', or 'MAP'.")
    
    return ts

def load_vsi(path: Optional[str] = None,
             confidence_level: Optional[float] = None,
             evaluation_grid_size: Optional[int] = None,
             estimator_method: Optional[str] = None,
             test_statistic_method: Optional[str] = None,
             assets_dir: Optional[str] = None):
    """
    Load a VSI object from a saved dill file.
    
    You can either supply the full file path via the `path` argument, or provide key parameters
    (confidence_level, evaluation_grid_size, estimator_method, test_statistic_method) so that the function
    constructs the folder name under the assets directory and loads the VSI instance from that folder.
    
    Parameters
    ----------
    path : str, optional
        Full path to the saved VSI file.
    confidence_level : float, optional
        Confidence level used in the VSI instance.
    evaluation_grid_size : int, optional
        Evaluation grid size used in the VSI instance.
    estimator_method : str, optional
        Estimator method used.
    test_statistic_method : str, optional
        Test statistic method used.
    assets_dir : str, optional
        Path to the assets directory. Defaults to assets directory defined in your project.
    
    Returns
    -------
    VSI
        The loaded VSI object.
    
    Raises
    ------
    FileNotFoundError
        If the specified file or folder cannot be found.
    ValueError
        If key parameters are missing when a full path is not provided.
    """
    import os
    import dill
    
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            vsi_obj = dill.load(f)
        return vsi_obj
    
    # If no full path is provided, we require the key attributes.
    if assets_dir is None:
        # You can set a default assets directory here.
        assets_dir = os.path.join(os.getcwd(), "assets")
    if confidence_level is None or evaluation_grid_size is None or estimator_method is None or test_statistic_method is None:
        raise ValueError("When no path is provided, you must supply confidence_level, evaluation_grid_size, estimator_method, and test_statistic_method.")
    
    folder_name = f"VSI_conf{confidence_level}_grid{evaluation_grid_size}_est{estimator_method}_test{test_statistic_method}"
    folder_path = os.path.join(assets_dir, folder_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Look for the VSI instance file. We assume the main file starts with "VSI_instance"
    candidates = [f for f in os.listdir(folder_path) if f.startswith("VSI_instance") and f.endswith(".pkl")]
    if not candidates:
        raise FileNotFoundError("No VSI instance file found in the folder.")
    # Optionally sort and choose the latest version.
    candidates.sort()
    vsi_file = os.path.join(folder_path, candidates[-1])
    with open(vsi_file, "rb") as f:
        vsi_obj = dill.load(f)
    return vsi_obj

def select_spread_out_stars_near_tracks(catalog, tracks, target_feh, feh_tol=0.05, k=5):
    """
    Select a set of stars near a given metallicity track, spaced along the track in (Teff, logg) space.

    This function identifies `k` stars from the input catalog that are near a specified evolutionary
    track defined in (Teff, logg) space, for a fixed target metallicity ([Fe/H]). The selected stars
    are evenly spaced along the track, based on proximity to evenly sampled points on the track curve.

    Parameters
    ----------
    catalog : structured ndarray
        Structured NumPy array containing stellar parameters. Must include fields 'TEFF', 'LOGG', and 'FE_H'.
    tracks : list of ndarray
        List of arrays of shape (N, 2), where each array represents a track in (Teff, logg) space.
    target_feh : float
        Target [Fe/H] metallicity value used to filter stars in the catalog.
    feh_tol : float, optional
        Tolerance around `target_feh` used to select stars. Default is 0.05.
    k : int, optional
        Number of stars to select. Default is 5.

    Returns
    -------
    ndarray
        Array of integer indices into `catalog` corresponding to the selected stars.
    
    Raises
    ------
    ValueError
        If no stars are found within the specified [Fe/H] tolerance.
    """
    # Filter by Fe/H
    feh = catalog["FE_H"]
    feh_mask = np.abs(feh - target_feh) < feh_tol
    if np.sum(feh_mask) == 0:
        raise ValueError("No stars found within the [Fe/H] window.")
    
    teff = catalog["TEFF"][feh_mask]
    logg = catalog["LOGG"][feh_mask]
    star_coords = np.vstack([teff, logg]).T
    star_indices = np.where(feh_mask)[0]

    # Combine and downsample the track evenly
    track_points = np.vstack(tracks)
    
    # Sort by Teff (or along arc length, if you prefer)
    order = np.argsort(track_points[:, 0])
    track_sorted = track_points[order]

    # Choose k points evenly spaced along the sorted track
    selected_track_points = track_sorted[np.linspace(0, len(track_sorted)-1, k).astype(int)]

    # For each point, find closest star
    star_tree = cKDTree(star_coords)
    indices = []

    for pt in selected_track_points:
        dist, idx = star_tree.query(pt)
        indices.append(star_indices[idx])  # convert to index in original catalog

    return np.array(indices)
