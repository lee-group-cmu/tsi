import pickle
import numpy as np
import random
from brutus import seds as bseds
from brutus import utils as butils
from brutus import filters
from brutus import pdf
import numpy.lib.recfunctions as rfn


def fetch_grid(assets_dir):
    """
    This method gets the SEDmaker which generates three grids:
    - Labels, e.g. intrinsic parameters
    - Params, e.g. surface-level parameters
    - SEDs, e.g. spectral energy distributions, with reddening, by default at
        distance 1000
    Each of these grids are structured numpy arrays, meaning that their rows are
    tuples corresponding with individual stars. Their dtypes contain names of
    the fields that represent the stars on each corresponding level.
    """
    try:
        with open(f'{assets_dir}/mist.pkl', 'rb') as f:
            mist = pickle.load(f)
    except:
        mist = bseds.SEDmaker(nnfile=f'{assets_dir}/nn_c3k.h5',
                        mistfile=f'{assets_dir}/MIST_1.2_EEPtrk.h5')
        # build the SED grid
        mist.make_grid(smf_grid=np.array([0.]),  # no binaries
                    afe_grid=np.array([0.]))  # no afe

    # Get a mask for successful parameter inputs
    sel = mist.grid_sel

    # Get the labels and params and seds
    labels = mist.grid_label[sel]
    params = mist.grid_param[sel]
    seds = mist.grid_sed[sel]

    # Do some basic transforms on the labels and params
    params = rfn.append_fields(params, 't_eff', 10**params['logt'], usemask=False)
    params = rfn.append_fields(params, 'age', 10**params['loga'], usemask=False)
    params = rfn.merge_arrays((labels, params),
                            flatten=True,
                            usemask=False)

    # Subselect the filters
    filt = filters.ps[:-2] + filters.tmass
    stype = np.dtype([(f, '<f8', (3,)) for f in filt])
    seds = np.array(seds[filt].tolist(), dtype=stype)

    return params, seds


def transform_meas(params, seds):
    """
    This method transforms the SEDs into a format that can be operated on more
    readily. Steps involved are:
    - Subtracting off the default distance modulus
    - Adding back varying distance moduli

    Args:
        params (np.ndarray): A structured numpy array of parameters
        seds (np.ndarray): An unstructured numpy array of SEDs of shape (n_samples, n_filters)
    """
    assert seds.dtype == np.float64

    # Subtract off default distance modulus
    default_mu = 5. * np.log10(1000.) - 5.
    seds -= default_mu

    # Add back varying distance moduli
    mu = (5. * np.log10(params['dist']) - 5.).reshape(-1, 1).repeat(seds.shape[1], axis=1)
    seds += mu

    return seds


def sample_joint_distribution(params, seds, args: dict, n_samples: int, assets_dir: str):
    """
    This method samples the prior distribution where the shape of the prior is
    determined by the args input. The steps involved are:
    0. Processing.
    - Combine datasets (params and seds) into one big grid
    1. Sample the prior.
    - Sample `len(params)` distances from the distance prior and assign them to
        the distance field in the data array
    - Sample `n_samples` params from the params array in proportion to the 
        galactic prior
    - Sample `n_samples` of av and rv components and assign them to the 
        sampled params
    2. Evaluate the likelihood.
    - Redden the magnitudes of the seds
    - Correct the default distance modulus to the sampled distance modulus
    - Normalize measurements (while retaining one raw magnitude for comparison)
    Return a tuple of the prior samples and the measurements, each of which are
    structured numpy arrays.

    Args:
        args (dict): A dictionary of arguments that define the prior
            distribution, namely the age_feh_hyperparameter between -1 and 1
        n_samples (int): The number of samples to draw from the prior
    """
    # Define the input and output structures
    ptype = params.dtype
    stype = seds.dtype
    mtype = np.dtype([(name, dtype, (1,)) for name, dtype, shape in stype.descr] + [('norm', np.float64, (1,))])

    # Define the prior
    try:
        with open(f'{assets_dir}/prior.pkl', 'rb') as f:
            prior = pickle.load(f)
    except:
        prior = BrutusPrior(coords=(90, 20), assets_path=assets_dir, reddening=False)
        with open(f'{assets_dir}/prior.pkl', 'wb') as f:
            pickle.dump(prior, f)
    Ngrid = len(params)

    # 1. Sample the prior
    ## Sample `len(params)` distances from the distance prior and assign them to
    ## the distance field in the data array
    dist_grid = np.logspace(-2, 2, 1000)
    try:
        params = rfn.append_fields(params, 'dist', np.random.choice(dist_grid, size=Ngrid), usemask=False)
    except:
        params['dist'] = np.random.choice(dist_grid, size=Ngrid)

    # 0. Processing (after having added `dist` to `params` for use later)
    grid = rfn.merge_arrays((params, seds),
                            flatten=True,
                            usemask=False)

    ## Sample `n_samples` params from the params array in proportion to the
    ## galactic prior
    sample_indices, _ = prior.sample(n_samples,
                                     grid,
                                     age_feh_hyperparam=args['age_feh_hyperparam'] if 'age_feh_hyperparam' in args else 0.0,
                                        halo_hyperparam=args['halo_hyperparam'] if 'halo_hyperparam' in args else 0.0,
                                        imf_hyperparam=args['imf_hyperparam'] if 'imf_hyperparam' in args else 0.0)
    prior_samples, meas_samples = params[sample_indices], seds[sample_indices]

    ## Sample `n_samples` of av and rv components and assign them to the
    ## sampled params
    av_grid = np.arange(0., 1.5 + 1e-5, 0.05)
    rv_grid = np.arange(2.4, 4.2 + 1e-5, 0.05)
    try:
        prior_samples = rfn.append_fields(prior_samples, 'av', np.random.choice(av_grid, size=n_samples), usemask=False)
    except:
        prior_samples['av'] = np.random.choice(av_grid, size=n_samples)
    try:
        prior_samples = rfn.append_fields(prior_samples, 'rv', np.random.choice(rv_grid, size=n_samples), usemask=False)
    except:
        prior_samples['rv'] = np.random.choice(rv_grid, size=n_samples)

    # 2. Evaluate the likelihood
    ## Redden the magnitudes of the seds
    mag_coeffs = np.vstack([np.array(sed.tolist()).reshape(1, -1, 3) for sed in meas_samples]) # Unstructured (n_samples, n_filters, 3) np.ndarray
    reddened_seds = butils.get_seds(mag_coeffs, prior_samples['av'], prior_samples['rv'])

    ## Correct the default distance modulus to the sampled distance modulus
    reddened_seds = transform_meas(prior_samples, reddened_seds)

    ## Normalize measurements (while retaining one raw magnitude for comparison)
    magnitudes = np.linalg.norm(reddened_seds, axis=1).reshape(-1, 1)
    reddened_seds = np.hstack([reddened_seds / magnitudes, magnitudes])

    # Revert to structured array
    meas_samples = np.array([tuple(sed) for sed in reddened_seds], dtype=mtype) # Structured (n_samples,) np.ndarray

    return prior_samples, meas_samples


def sample_uniform_distribution(params, seds, n_samples: int):
    """
    This method samples the uniform distribution over the primary parameters of
    interest. The steps involved are:
    0. Processing.
    - Define an array with only the three parameters of primary interest
    1. Sample the uniform distribution.
    - Bin the grid into a 3D histogram
    - Sample from the bins until we have at least `n_samples` entries
    2. Assign a random distance to each sampled entry.
    - Sample `n_samples` of av and rv components and assign them to the 
        sampled params
    3. Evaluate the likelihood.
    - Redden the magnitudes of the seds
    - Correct the default distance modulus to the sampled distance modulus
    - Normalize measurements (while retaining one raw magnitude for comparison)
    Return a tuple of the prior samples and the measurements, each of which are
    structured numpy arrays.
    """
    # Define the input and output structures
    ptype = params.dtype
    stype = seds.dtype
    mtype = np.dtype([(name, dtype, (1,)) for name, dtype, shape in stype.descr] + [('norm', np.float64, (1,))])

    # 0. Processing
    evaluation_grid = params, seds
    grid = evaluation_grid[0]

    # Define the number of bins for each dimension
    num_bins = 30

    # Extract the parameters for the 3D histogram
    param1, param2, param3 = 'logg', 'logt', 'feh_surf'

    # Convert the structured array to a regular numpy array
    grid_array = np.array([grid[param1], grid[param2], grid[param3]]).T

    # 1. Sample the uniform distribution
    # Create the 3D histogram
    hist, edges = np.histogramdd(grid_array, bins=num_bins)

    # Get the indices of the grid rows that fall into each bin
    indices = [np.digitize(grid_array[:, i], edges[i]) - 1 for i in range(grid_array.shape[1])]
    indices = np.array(indices).T

    # Clip indices to ensure they are within valid range
    indices = np.clip(indices, 0, np.array(hist.shape) - 1)

    # Create a list to hold the sets of indices for each bin
    bin_indices = [[] for _ in range(hist.size)]

    # Iterate over the grid rows and assign them to the appropriate bin
    for i, idx in enumerate(indices):
        bin_idx = np.ravel_multi_index(idx, hist.shape)
        bin_indices[bin_idx].append(i)

    # Initialize an empty list to hold the sampled indices
    sampled_indices = []

    nonempty_sublists = [sublist for sublist in bin_indices if sublist]

    # Continue sampling until we have at least num_samples entries
    while len(sampled_indices) < n_samples:
        # Sample a random nonempty sublist
        random_sublist = random.choice(nonempty_sublists)

        # Pop an entry from the selected sublist at random
        random_entry = random_sublist.pop(np.random.randint(len(random_sublist)))

        if not random_sublist:
            nonempty_sublists.remove(random_sublist)

        # Add the popped entry to the sampled indices
        sampled_indices.append(random_entry)

    # Convert the sampled indices to a numpy array
    sampled_indices = np.array(sampled_indices)
    unif_samples, unif_samples_meas = evaluation_grid[0][sampled_indices], evaluation_grid[1][sampled_indices]

    # 2. Assign a random distance to each sampled entry
    dist_grid = np.logspace(-2, 2, 1000)
    try:
        unif_samples = rfn.append_fields(unif_samples, 'dist', np.random.choice(dist_grid, size=n_samples), usemask=False)
    except:
        unif_samples['dist'] = np.random.choice(unif_samples, size=n_samples)

    ## Sample `n_samples` of av and rv components and assign them to the
    ## sampled params
    av_grid = np.arange(0., 1.5 + 1e-5, 0.05)
    rv_grid = np.arange(2.4, 4.2 + 1e-5, 0.05)
    try:
        unif_samples = rfn.append_fields(unif_samples, 'av', np.random.choice(av_grid, size=n_samples), usemask=False)
    except:
        unif_samples['av'] = np.random.choice(av_grid, size=n_samples)
    try:
        unif_samples = rfn.append_fields(unif_samples, 'rv', np.random.choice(rv_grid, size=n_samples), usemask=False)
    except:
        unif_samples['rv'] = np.random.choice(rv_grid, size=n_samples)

    # 3. Evaluate the likelihood
    ## Redden the magnitudes of the seds
    mag_coeffs = np.vstack([np.array(sed.tolist()).reshape(1, -1, 3) for sed in unif_samples_meas]) # Unstructured (n_samples, n_filters, 3) np.ndarray
    reddened_seds = butils.get_seds(mag_coeffs, unif_samples['av'], unif_samples['rv'])

    ## Correct the default distance modulus to the sampled distance modulus
    reddened_seds = transform_meas(unif_samples, reddened_seds)

    ## Normalize measurements (while retaining one raw magnitude for comparison)
    magnitudes = np.linalg.norm(reddened_seds, axis=1).reshape(-1, 1)
    reddened_seds = np.hstack([reddened_seds / magnitudes, magnitudes])

    # Revert to structured array
    unif_samples_meas = np.array([tuple(sed) for sed in reddened_seds], dtype=mtype) # Structured (n_samples,) np.ndarray

    return unif_samples, unif_samples_meas


class BrutusPrior:
    def __init__(self, coords=(90, 20), assets_path='', reddening=False):
        self.coords = coords
        self.dustfile = f'{assets_path}/bayestar2019_v1.h5'
        self.reddening = reddening

    def __call__(self, theta, halo_hyperparam=0.0, age_feh_hyperparam=0.0, imf_hyperparam=0.0, return_components=False):
        if self.reddening:
            return pdf.gal_lnprior(theta['dist'], self.coords, theta, halo_hyperparam) + pdf.dust_lnprior(theta['dist'], self.coords, theta['av'], dustfile=self.dustfile)
        else:
            assert halo_hyperparam >= 0.0 and halo_hyperparam <= 1.0
            assert age_feh_hyperparam >= -2.0 and age_feh_hyperparam <= 2.0
            assert imf_hyperparam >= 0.0 and imf_hyperparam <= 1.0
            return pdf.gal_lnprior(dists=theta['dist'],
                                   coord=self.coords,
                                   labels=theta,
                                   feh_thin=-0.2,
                                   feh_thick=-0.7,
                                   feh_halo=-1.6 - age_feh_hyperparam - 1.0,
                                   feh_age_ctr=-0.5 - age_feh_hyperparam / 2,
                                   feh_age_scale=0.5,
                                   halo_hyperparam=halo_hyperparam,
                                   return_components=return_components) + \
                pdf.imf_lnprior(theta['mini'],
                                alpha_low=1.3 - 0.7 * imf_hyperparam + 0.7 * (1 - imf_hyperparam),
                                alpha_high=2.3 - 0.3 * imf_hyperparam + 0.3 * (1 - imf_hyperparam))

    def sample(self, sample_size, parameter_grid, halo_hyperparam=0.0, age_feh_hyperparam=0.0, imf_hyperparam=0.0):
        probs = np.exp(self.__call__(parameter_grid, halo_hyperparam, age_feh_hyperparam, imf_hyperparam))
        probs /= probs.sum()
        indices = np.random.choice(np.arange(len(probs)), size=sample_size, p=probs, replace=True)
        return indices, parameter_grid[indices]
