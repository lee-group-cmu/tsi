import os
import sys
import numpy as np
import torch
import warnings
import logging
import dill

from astropy import table

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore", message="No prior bounds were passed")
warnings.filterwarnings("ignore", message="Prior is lacking mean attribute")
warnings.filterwarnings("ignore", message="Prior is lacking variance attribute")

class NoSingularMatrixFilter(logging.Filter):
    def filter(self, record):
        return "Singular matrix" not in record.getMessage()

logging.getLogger().addFilter(NoSingularMatrixFilter())

# Add source directory
src_subdir = "../src/"
if src_subdir not in sys.path:
    sys.path.insert(1, src_subdir)

# Imports from your project
from data import read_catalog, generate_splits, process_all_splits, process_split, add_quality_label_column
from utils import setup_paths
from vsi import VSI

np.random.seed(42)
torch.manual_seed(42)
random_state = 42

#############################################
#############################################
# SETUP 1: MAF, Posterior, Pristine vs Good w/ 20% Calibration  #
#############################################
#############################################

data_paths = setup_paths(notebook=True)
catalog_path = os.path.join(data_paths["data"], "xp_apogee_cat.h5")
catalog = table.Table.read(catalog_path, format = "hdf5")
catalog = catalog.as_array()
catalog = add_quality_label_column(catalog)
catalog = catalog[["coeffs", "LOGG", "TEFF", "FE_H", "ALPHA_M", "QUALITY"]]

print("Catalog shape:", catalog.shape)

##############################
##### GLOBAL PARAMETERS ######
##############################

VERSION = "V4_QUALITY_CUTS_REDUCED_CALIB"
SAVE_POINTS = True

# Number of stars to holdout for visualization
N_HOLDOUT = 25
SUNLIKE_PARAMS = [4.43775, 5772, 0.0]
MS_LOW_PARAMS = [4, 6100, -0.5]
MS_MID_PARAMS = [4.2, 5200, 0.5]
AGB_LOW_PARAMS = [2, 5000, -1.5]
AGB_MID_PARAMS = [1.9, 4600, -1]

TRAINING_BATCH_SIZE = 20

TRAIN_SIZE = 0.5
CALIBRATION_SIZE = 0.2
TEST_SIZE = 1 - (TRAIN_SIZE + CALIBRATION_SIZE)

BIAS_TYPE = "PRISTINE"
PERC_BIAS_TYPE = 1

CONFIDENCE_LEVEL = 0.90
EVALUATION_GRID_SIZE = 1_000_000
PRIOR_METHOD = "kde"
ESTIMATOR_METHOD = "npe"
TEST_STATISTIC_METHOD = "posterior"
CALIBRATION_METHOD = "critical-values"
SEED = 42

##############################
##############################
##############################

##############################
##### HOLD OUT N STARS #######
##############################

catalog_features = np.vstack([
    catalog["LOGG"],
    catalog["TEFF"],
    catalog["FE_H"]
]).T

N_HOLDOUT_sub = int(N_HOLDOUT/5)

# SUN-LIKE STARS
sun_vec = np.array([SUNLIKE_PARAMS])  # LOGG, TEFF, FE_H

distances = pairwise_distances(catalog_features, sun_vec).flatten()

sunlike_indices = np.argsort(distances)[:N_HOLDOUT_sub]

# MS LOW METAL
ms_low_vec = np.array([MS_LOW_PARAMS])

distances = pairwise_distances(catalog_features, ms_low_vec).flatten()

ms_low_indices = np.argsort(distances)[:N_HOLDOUT_sub]

# MS MID METAL
ms_mid_vec = np.array([MS_MID_PARAMS])

distances = pairwise_distances(catalog_features, ms_mid_vec).flatten()

ms_mid_indices = np.argsort(distances)[:N_HOLDOUT_sub]

# AGB LOW METAL
agb_low_vec = np.array([AGB_LOW_PARAMS])

distances = pairwise_distances(catalog_features, agb_low_vec).flatten()

agb_low_indices = np.argsort(distances)[:N_HOLDOUT_sub]

# AGB MID MEATL
agb_mid_vec = np.array([AGB_MID_PARAMS])

distances = pairwise_distances(catalog_features, agb_mid_vec).flatten()

agb_mid_indices = np.argsort(distances)[:N_HOLDOUT_sub]

# HOLDOUT
selected_indices = np.unique(np.concatenate([sunlike_indices, 
                                             ms_low_indices, ms_mid_indices, 
                                             agb_low_indices, agb_mid_indices]))

# SAVING
catalog_holdout = catalog[selected_indices]

file_name = f"selected_stars_holdout_N{N_HOLDOUT}_V{VERSION}.pkl"
output_path = os.path.join(data_paths["data"], file_name)

with open(output_path, "wb") as f:
    dill.dump(catalog_holdout, f)

# Remove from full catalog
catalog_removed = np.delete(catalog, selected_indices)

catalog = catalog_removed

##############################
##############################
##############################

##############################
##### DATA SPLIT SETUPS ######
##############################

train_size = TRAIN_SIZE
calibration_size = CALIBRATION_SIZE
test_size = TEST_SIZE

# NO SELECTION BIAS
is_good = catalog["QUALITY"] == "good"
is_pristine = catalog["QUALITY"] == "pristine"
is_not_full = catalog["QUALITY"] != "full"

good_and_pristine_idx = np.where(is_not_full)[0]
good_and_pristine_catalog = catalog[good_and_pristine_idx]

train_data_good, test_calib_data_good = train_test_split(good_and_pristine_catalog, test_size=test_size+calibration_size, random_state=random_state)
calibration_data_good, test_data_good = train_test_split(test_calib_data_good, test_size=test_size/(test_size+calibration_size), random_state=random_state)

# Combine good calibration data with full data
# full_quality_catalog = catalog[catalog["QUALITY"] == "full"]
# calibration_data_good_full = np.concatenate([calibration_data_good, full_quality_catalog])

good_setting_splits = {
        "train": train_data_good,
        "test": test_data_good,
        "calibration": calibration_data_good#_full
    }

# SELECTION BIAS
train_data_good, test_calib_data_good = train_test_split(good_and_pristine_catalog, test_size=test_size+calibration_size, random_state=random_state)
calibration_data_good, test_data_good = train_test_split(test_calib_data_good, test_size=test_size/(test_size+calibration_size), random_state=random_state)

train_data_pristine = train_data_good[train_data_good["QUALITY"] == "pristine"]

# Combine good calibration data with full data
#full_quality_catalog = catalog[catalog["QUALITY"] == "full"]
#calibration_data_good_full = np.concatenate([calibration_data_good, full_quality_catalog])

pristine_setting_splits = {
        "train": train_data_pristine,
        "test": test_data_good,
        "calibration": calibration_data_good#_full
    }

##############################
##############################
##############################

#############################################
########## PRISTINE SELECTION BIAS ##########
#############################################

print("\n====STARTING PRISTINE SELECTION BIAS RESULTS.=====")

processed = process_all_splits(pristine_setting_splits, spectra=["coeffs"], features=["LOGG", "TEFF", "FE_H"], n_coeffs=110)

vsi = VSI(
    confidence_level=CONFIDENCE_LEVEL,
    evaluation_grid_size=EVALUATION_GRID_SIZE,
    prior_method=PRIOR_METHOD,
    estimator_method=ESTIMATOR_METHOD,
    test_statistic_method=TEST_STATISTIC_METHOD,
    calibration_method=CALIBRATION_METHOD,
    seed=SEED,
    notebook=True
)

vsi.load_data(
    X_train=processed["train"]["X"],
    y_train=processed["train"]["y"],
    X_calibration=processed["calibration"]["X"],
    y_calibration=processed["calibration"]["y"],
    X_test=processed["test"]["X"],
    y_test=processed["test"]["y"]
)

print("\nData loaded into VSI object.")

vsi.generate_evaluation_grid()

# SAVING
STEP = "EVALGRID"
folder_name = f"VSI_bias{BIAS_TYPE}_split{PERC_BIAS_TYPE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
folder_path = os.path.join(data_paths["assets"], folder_name)
file_name = f"VSI_instance_checkpoint_{STEP}.pkl"
os.makedirs(folder_path, exist_ok=True)

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

print("\nCompleted Evaluation Grid.")

vsi.set_prior()

vsi.train_estimator()

# SAVING
# STEP = "EST_TRAINING"
# folder_name = f"VSI_bias{BIAS_TYPE}_split{PERC_BIAS_TYPE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

vsi.build_test_statistic()
vsi.fit_lf2i()

# held out sun-like stars
stars = process_split(catalog_holdout)

X_query = stars["X"]
y_query = stars["y"]

confidence_sets = vsi.inference_lf2i(x = X_query)

# SAVING
# STEP = "LF2I_INFERENCE"
# folder_name = f"VSI_bias{BIAS_TYPE}_split{PERC_BIAS_TYPE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

diagnostics_lf2i = vsi.diagnostics_lf2i()

# SAVING
STEP = "LF2I_DIAG"
folder_name = f"VSI_bias{BIAS_TYPE}_split{PERC_BIAS_TYPE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
folder_path = os.path.join(data_paths["assets"], folder_name)
file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

output_path = os.path.join(folder_path, file_name)

with open(output_path, "wb") as f:
    dill.dump(vsi, f)

print("\nLF2I Inference and Diagnostics Done.")

indicators, credible_sets = vsi.inference_estimator()

# SAVING
# STEP = "EST_INFERENCE"
# folder_name = f"VSI_bias{BIAS_TYPE}_split{PERC_BIAS_TYPE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

diagnostics_estimator = vsi.diagnostics_estimator(indicators=indicators)

# SAVING
# STEP = "EST_DIAG"
# folder_name = f"VSI_bias{BIAS_TYPE}_split{PERC_BIAS_TYPE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

print("\nHPD Inference and Diagnostics Done.")

# SAVING
STEP = "COMPLETE"
folder_name = f"VSI_bias{BIAS_TYPE}_split{PERC_BIAS_TYPE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
folder_path = os.path.join(data_paths["assets"], folder_name)
file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

output_path = os.path.join(folder_path, file_name)

with open(output_path, "wb") as f:
    dill.dump(vsi, f)

print("\n====PRISTINE SELECTION BIAS RESULTS COMPLETE.=====")

##############################
##############################
##############################

############################################
########## NO SELECTION BIAS ###############
############################################

print("\n====STARTING NO SELECTION BIAS RESULTS.=====")

processed = process_all_splits(good_setting_splits, spectra=["coeffs"], features=["LOGG", "TEFF", "FE_H"], n_coeffs=110)

vsi = VSI(
    confidence_level=CONFIDENCE_LEVEL,
    evaluation_grid_size=EVALUATION_GRID_SIZE,
    prior_method=PRIOR_METHOD,
    estimator_method=ESTIMATOR_METHOD,
    test_statistic_method=TEST_STATISTIC_METHOD,
    calibration_method=CALIBRATION_METHOD,
    seed=SEED,
    notebook=True
)

vsi.load_data(
    X_train=processed["train"]["X"],
    y_train=processed["train"]["y"],
    X_calibration=processed["calibration"]["X"],
    y_calibration=processed["calibration"]["y"],
    X_test=processed["test"]["X"],
    y_test=processed["test"]["y"]
)

print("\nData loaded into VSI object.")

vsi.generate_evaluation_grid()

# SAVING
STEP = "EVALGRID"
folder_name = f"VSI_biasNO_split{TEST_SIZE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
folder_path = os.path.join(data_paths["assets"], folder_name)
file_name = f"VSI_instance_checkpoint_{STEP}.pkl"
os.makedirs(folder_path, exist_ok=True)

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

print("\nCompleted Evaluation Grid.")

vsi.set_prior()

vsi.train_estimator(training_batch_size = TRAINING_BATCH_SIZE)

# SAVING
# STEP = "EST_TRAINING"
# folder_name = f"VSI_biasNO_split{TEST_SIZE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

vsi.build_test_statistic()
vsi.fit_lf2i()

# held out sun-like stars
stars = process_split(catalog_holdout)

X_query = stars["X"]
y_query = stars["y"]

confidence_sets = vsi.inference_lf2i(x = X_query)

# SAVING
# STEP = "LF2I_INFERENCE"
# folder_name = f"VSI_biasNO_split{TEST_SIZE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

diagnostics_lf2i = vsi.diagnostics_lf2i()

# SAVING
STEP = "LF2I_DIAG"
folder_name = f"VSI_biasNO_split{TEST_SIZE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
folder_path = os.path.join(data_paths["assets"], folder_name)
file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

output_path = os.path.join(folder_path, file_name)

with open(output_path, "wb") as f:
    dill.dump(vsi, f)

print("\nLF2I Inference and Diagnostics Done.")

indicators, credible_sets = vsi.inference_estimator()

# SAVING
# STEP = "EST_INFERENCE"
# folder_name = f"VSI_biasNO_split{TEST_SIZE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

diagnostics_estimator = vsi.diagnostics_estimator(indicators=indicators)

# SAVING
# STEP = "EST_DIAG"
# folder_name = f"VSI_biasNO_split{TEST_SIZE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
# folder_path = os.path.join(data_paths["assets"], folder_name)
# file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

# output_path = os.path.join(folder_path, file_name)

# with open(output_path, "wb") as f:
#     dill.dump(vsi, f)

print("\nHPD Inference and Diagnostics Done.")

# SAVING
STEP = "COMPLETE"
folder_name = f"VSI_biasNO_split{TEST_SIZE}_conf{CONFIDENCE_LEVEL}_grid{EVALUATION_GRID_SIZE}_est{ESTIMATOR_METHOD}_test{TEST_STATISTIC_METHOD}_ver{VERSION}"
folder_path = os.path.join(data_paths["assets"], folder_name)
file_name = f"VSI_instance_checkpoint_{STEP}.pkl"

output_path = os.path.join(folder_path, file_name)

with open(output_path, "wb") as f:
    dill.dump(vsi, f)

print("\n====NO SELECTION BIAS RESULTS COMPLETE.=====")

##############################
##############################
##############################