import numpy as np
import pandas as pd
import os
import sys
from typing import List, Dict, Union

from astropy import table
from numpy.lib.recfunctions import append_fields

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import torch

# CREDIT: asrtoNN (package implementation broken)
# https://astronn.readthedocs.io/en/stable/_modules/astroNN/apogee/chips.html#bitmask_boolean
def bitmask_boolean(bitmask, target_bit):
    """
    Turn bitmask to boolean with provided bitmask array and target bit to mask

    :param bitmask: bitmask
    :type bitmask: ndarray
    :param target_bit: target bit to mask
    :type target_bit: list[int]
    :return: boolean array, True for clean, False for masked
    :rtype: ndarray[bool]
    :History: 2018-Feb-03 - Written - Henry Leung (University of Toronto)
    """
    target_bit = np.array(target_bit)
    target_bit = np.sum(2**target_bit)
    bitmask = np.atleast_2d(bitmask)
    boolean_output = np.zeros(bitmask.shape, dtype=bool)
    boolean_output[(bitmask & target_bit) != 0] = True
    return boolean_output

def read_catalog(path: str, 
                 spectra: List[str] = ["coeffs"],
                 labels: List[str] = ["LOGG", "TEFF", "FE_H", "ALPHA_M"], 
                 filter: str = "good",
                 normalize: bool = False) -> np.ndarray:
    """
    Read Gaia/APOGEE cross-match catalog from an HDF5 file and apply quality filters 
    as defined in Laroche & Speagle (2024).

    Parameters
    ----------
    path : str
        Path to the HDF5 catalog file.
    spectra : list, optional
        List of spectral coefficient column names to include in the returned DataFrame.
        Defaults to ["coeffs"].
    labels : list, optional
        List of stellar label column names to include in the returned DataFrame.
        Defaults to ["LOGG", "TEFF", "FE_H", "ALPHA_M", "M_H"].
    filter : str, optional
        Quality cut to apply to the catalog. Options are:
            - 'full'     : No cuts; returns all entries with specified columns.
            - 'good'     : Moderate quality cut (SNR > 20, relaxed label uncertainties).
            - 'pristine' : Strict quality cut (SNR > 100, precise label uncertainties).
        Defaults to 'good'.
    normalize : bool, optional
        If True, normalize the flux coefficients by the Gaia G-band flux 
        (`phot_g_mean_flux`). Defaults to False.

    Returns
    -------
    np.ndarray
        Filtered catalog containing the specified spectra and label columns.

    Raises
    ------
    ValueError
        If `filter` is not one of 'full', 'good', or 'pristine'.

    Notes
    -----
    - Quality cut definitions follow Laroche & Speagle (2024):
      https://arxiv.org/pdf/2404.07316
    - The paper incorrectly states T_eff SNR > 30 for the "good" catalog; the actual threshold is SNR > 20.
    - Bitmask filtering excludes stars with ASPCAPFLAG bits 2 (STAR_BAD) and 3 (NO_ASPCAP_RESULT).
    
    To-Do
    -----
    - Add option to normalize flux errors.
    """
    
    catalog = table.Table.read(path, format = "hdf5")

    if normalize:
        catalog["coeffs"] = catalog["coeffs"] / catalog["phot_g_mean_flux"] # dividing by average flux

    if filter == "full":
        # apply "full labels" quality cuts from Laroche & Speagle, 2024
        return catalog[spectra + labels].as_array()
    elif filter == "good":
        # apply "good labels" quality cuts from Laroche & Speagle, 2024
        sn_teff = catalog['TEFF'] / catalog['TEFF_ERR'] > 20

        logg_err_cut = catalog['LOGG_ERR'] < 0.4

        mh_err_cut = catalog['M_H_ERR'] < 0.2

        bp_rp = catalog['GAIAEDR3_PHOT_BP_MEAN_MAG'] - catalog['GAIAEDR3_PHOT_RP_MEAN_MAG']
        color_cut = (bp_rp > 0) & (bp_rp < 4)

        g_mag_cut = (catalog['GAIAEDR3_PHOT_G_MEAN_MAG'] > 6) & (catalog['GAIAEDR3_PHOT_G_MEAN_MAG'] < 17.5)

        starflag_cut = catalog['STARFLAG'] == 0

        aspflag_cut = ~bitmask_boolean(catalog['ASPCAPFLAG'], [2, 3])[0]

        good_rows = sn_teff & logg_err_cut & mh_err_cut & color_cut & g_mag_cut & starflag_cut & aspflag_cut

        good_catalog = catalog[good_rows]

        return good_catalog[spectra + labels].as_array()

    elif filter == "pristine":
        # apply "pristine labels" quality cuts from Laroche & Speagle, 2024
        sn_teff = catalog['TEFF'] / catalog['TEFF_ERR'] > 100

        logg_err_cut = catalog['LOGG_ERR'] < 0.1

        mh_err_cut = catalog['M_H_ERR'] < 0.05

        bp_rp = catalog['GAIAEDR3_PHOT_BP_MEAN_MAG'] - catalog['GAIAEDR3_PHOT_RP_MEAN_MAG']
        color_cut = (bp_rp > 0) & (bp_rp < 4)

        g_mag_cut = (catalog['GAIAEDR3_PHOT_G_MEAN_MAG'] > 6) & (catalog['GAIAEDR3_PHOT_G_MEAN_MAG'] < 17.5)

        starflag_cut = catalog['STARFLAG'] == 0

        aspflag_cut = ~bitmask_boolean(catalog['ASPCAPFLAG'], [2, 3])[0]

        pristine_rows = sn_teff & logg_err_cut & mh_err_cut & color_cut & g_mag_cut & starflag_cut & aspflag_cut

        pristine_catalog = catalog[pristine_rows]

        return pristine_catalog[spectra + labels].as_array()
    
    else:
        raise ValueError("Filter must be one of: 'full', 'good', or 'pristine'")

def generate_splits(catalog: np.ndarray,
                    prop_calibration: float = 0.1,
                    test_size: float = 0.2,
                    training_selection: str = "full",
                    random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate training, test, and calibration splits from a stellar catalog with optional 
    control over population selection and calibration proportion.

    This function supports selection based on stellar evolutionary class 
    (Main Sequence vs. AGB giants), classification derived from the logg–Teff plane.

    Parameters
    ----------
    catalog : np.ndarray
        Structured array or array-like object containing stellar data, including columns 
        "LOGG" and "TEFF" used for classification.
    prop_calibration : float, optional
        Fraction of the catalog to be allocated to the calibration set. Must be in (0, 1). 
        Default is 0.1.
    test_size : float, optional
        Fraction of the remaining catalog to allocate to the test set after removing 
        calibration data. Must be in [0, 1]. Default is 0.2.
    training_selection : str, optional
        Strategy for constructing the training set. Options are:
            - "full" : Use the entire remaining sample (no class filtering).
            - "AGB"  : Always include all AGB stars, vary MS based on test_size.
            - "MS"   : Always include all MS stars, vary AGB based on test_size.
        Default is "full".
    random_state : int, optional
        Random seed for reproducibility of sampling operations. Default is 42.

    Returns
    -------
    dict
        Dictionary with keys:
            - "train"        : np.ndarray of training data
            - "test"         : np.ndarray of test data (may be empty if `test_size` = 0)
            - "calibration"  : np.ndarray of calibration data

    Notes
    -----
    - Stellar classification is based on:
        logg >= 5.9 - 0.4 * (teff / 1000) → Main Sequence (MS), else AGB.
    - Calibration data is always sampled randomly across the entire catalog before 
      population-based filtering is applied.
    - If `test_size` is 0 or 1, the remaining population is either entirely in train or test.
    - Output arrays have the "class" column removed before return.

    To-Do
    -----
    - Add support for user-defined classification logic (e.g., RGB, HB).
    """

    # classify stars into Main Sequence (MS) and Giants (AGB)
    classification = np.where(catalog["LOGG"] >= 5.9 - 0.4 * catalog["TEFF"] / 1000, "MS", "AGB")
    catalog = np.lib.recfunctions.append_fields(catalog, "class", classification, usemask=False)

    # calibration split
    calibration_data, remaining = train_test_split(catalog, test_size=1 - prop_calibration, random_state=random_state)

    # train and test sets based on selections
    if training_selection == "full":
        if (test_size == 0) | (test_size == 1):
            print("Changing test and train split for `full` since test_size is 0.")
            test_size = 0.01
        train_data, test_data = train_test_split(remaining, test_size=test_size, random_state=random_state)
    elif training_selection == "AGB":
        agb_data = remaining[remaining["class"] == "AGB"]
        ms_data = remaining[remaining["class"] == "MS"]

        if (test_size == 0) | (test_size == 1):
            train_data = agb_data
            test_data = ms_data
        elif (test_size > 0) & (test_size < 1):
            train_data_ms, test_data = train_test_split(ms_data, test_size=test_size, random_state=random_state)
            train_data = np.concatenate([agb_data, train_data_ms])
    elif training_selection == "MS":
        agb_data = remaining[remaining["class"] == "AGB"]
        ms_data = remaining[remaining["class"] == "MS"]

        if (test_size == 0) | (test_size == 1):
            train_data = ms_data
            test_data = agb_data
        elif (test_size > 0) & (test_size < 1):
            train_data_agb, test_data = train_test_split(agb_data, test_size=test_size, random_state=random_state)
            train_data = np.concatenate([ms_data, train_data_agb])
    else:
        raise ValueError("Selection must be one of: 'full', 'AGB', or 'MS'.")

    # drop class column 
    train_data = train_data[[name for name in train_data.dtype.names if name != "class"]]
    test_data = test_data[[name for name in test_data.dtype.names if name != "class"]]
    calibration_data = calibration_data[[name for name in calibration_data.dtype.names if name != "class"]]

    return {
        "train": train_data,
        "test": test_data,
        "calibration": calibration_data
    }

def process_split(data: np.ndarray,
                  spectra: List[str] = ["coeffs"],
                  features: List[str] = ["LOGG", "TEFF", "FE_H"],
                  n_coeffs: int = 110) -> Dict[str, torch.Tensor]:
    """
    Convert a single catalog split into PyTorch tensors for LF2I.

    Parameters
    ----------
    data : np.ndarray
        Structured array with coefficient and label columns.
    spectra : list of str, optional
        Names of coefficient columns used as model inputs. Default is ["coeffs"].
    features : list of str, optional
        Names of label columns used as parameters of interest. Default is ["LOGG", "TEFF", "FE_H"].
    n_coeffs : int, optional
        Number of coefficients per star. Default is 110.

    Returns
    -------
    dict
        Dictionary with:
            - "X": torch.Tensor of shape (n_samples, n_coeffs * len(spectra))
            - "y": torch.Tensor of shape (n_samples, len(features))

    Raises
    ------
    ValueError
        If any required fields are missing from the data.
    """
    required_fields = spectra + features
    missing_fields = [field for field in required_fields if field not in data.dtype.names]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    X = np.hstack([np.vstack(data[spec]) for spec in spectra])
    y = np.stack([data[feat] for feat in features], axis=1)

    return {
        "X": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.float32)
    }


def process_all_splits(splits: Dict[str, np.ndarray],
                       spectra: List[str] = ["coeffs"],
                       features: List[str] = ["LOGG", "TEFF", "FE_H"],
                       n_coeffs: int = 110) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convert all splits (train/test/calibration) into PyTorch tensors for LF2I.

    Parameters
    ----------
    splits : dict
        Output from `generate_splits()`, with keys "train", "test", "calibration".
    spectra : list of str, optional
        Names of coefficient columns used as inputs. Default is ["coeffs"].
    features : list of str, optional
        Names of label columns used as parameters of interest. Default is ["LOGG", "TEFF", "FE_H"].
    n_coeffs : int, optional
        Number of coefficients per entry. Default is 110.

    Returns
    -------
    dict
        Nested dictionary of the form:
        {
            "train": {"X": ..., "y": ...},
            "test": {"X": ..., "y": ...},
            "calibration": {"X": ..., "y": ...}
        }
    """
    return {
        split_name: process_split(data_block, spectra=spectra, features=features, n_coeffs=n_coeffs)
        for split_name, data_block in splits.items()
    }

def find_nearest_star(catalog: np.ndarray,
                      parameter_names: List[str],
                      query_values: Union[np.ndarray, List[List[float]]],
                      return_indices: bool = False) -> Union[np.ndarray, tuple]:
    """
    Find the star(s) in the catalog whose properties are closest to a given set of values.

    Parameters
    ----------
    catalog : np.ndarray
        Structured array containing stellar properties (e.g., from `read_catalog` or `generate_splits`).
    parameter_names : list of str
        Names of parameters (e.g., ["FE_H", "TEFF", "LOGG"]) to match on.
    query_values : array-like
        Values of the parameters to search for. Can be a single list or 2D array of shape (n_queries, n_params).
    return_indices : bool, optional
        If True, also return the indices of the matched stars. Default is False.

    Returns
    -------
    matches : np.ndarray
        Subset of `catalog` with one row per query — the nearest match(es).
    indices : np.ndarray, optional
        Indices of matched stars in the input `catalog`, returned if `return_indices=True`.

    Raises
    ------
    ValueError
        If any parameter in `parameter_names` is missing from the catalog.
    """

    # check if parameters exist in catalog
    missing = [param for param in parameter_names if param not in catalog.dtype.names]
    if missing:
        raise ValueError(f"Missing parameters in catalog: {missing}")

    # extract catalog values to match against
    catalog_params = np.stack([catalog[param] for param in parameter_names], axis=1)

    # query values
    query_values = np.atleast_2d(query_values)  # shape: (n_queries, n_params)

    # nearest neighbor search
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(catalog_params)
    distances, indices = nn.kneighbors(query_values)

    matched = catalog[indices.flatten()]

    if return_indices:
        return matched, indices.flatten()
    else:
        return matched

def add_quality_label_column(catalog):
    # Base assumption: all are full
    quality_labels = np.full(len(catalog), "full", dtype="<U8")

    # GOOD CUTS
    sn_teff_good = catalog['TEFF'] / catalog['TEFF_ERR'] > 20
    logg_err_cut_good = catalog['LOGG_ERR'] < 0.4
    mh_err_cut_good = catalog['M_H_ERR'] < 0.2
    bp_rp = catalog['GAIAEDR3_PHOT_BP_MEAN_MAG'] - catalog['GAIAEDR3_PHOT_RP_MEAN_MAG']
    color_cut = (bp_rp > 0) & (bp_rp < 4)
    g_mag_cut = (catalog['GAIAEDR3_PHOT_G_MEAN_MAG'] > 6) & (catalog['GAIAEDR3_PHOT_G_MEAN_MAG'] < 17.5)
    starflag_cut = catalog['STARFLAG'] == 0
    aspflag_cut = ~bitmask_boolean(catalog['ASPCAPFLAG'], [2, 3])[0]

    good_rows = sn_teff_good & logg_err_cut_good & mh_err_cut_good & color_cut & g_mag_cut & starflag_cut & aspflag_cut
    quality_labels[good_rows] = "good"

    # PRISTINE CUTS (stricter than good, overwrite if met)
    sn_teff_pristine = catalog['TEFF'] / catalog['TEFF_ERR'] > 100
    logg_err_cut_pristine = catalog['LOGG_ERR'] < 0.1
    mh_err_cut_pristine = catalog['M_H_ERR'] < 0.05

    pristine_rows = sn_teff_pristine & logg_err_cut_pristine & mh_err_cut_pristine & color_cut & g_mag_cut & starflag_cut & aspflag_cut
    quality_labels[pristine_rows] = "pristine"

    # Add as a new field to structured array
    catalog_with_labels = append_fields(catalog, 'QUALITY', quality_labels, dtypes='<U8', usemask=False)

    return catalog_with_labels