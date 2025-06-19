import os
import sys
import warnings
import logging
import signal

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message="No prior bounds were passed")
warnings.filterwarnings("ignore", message="Prior is lacking mean attribute")
warnings.filterwarnings("ignore", message="Prior is lacking variance attribute")

class NoSingularMatrixFilter(logging.Filter):
    def filter(self, record):
        return "Singular matrix" not in record.getMessage()

logging.getLogger().addFilter(NoSingularMatrixFilter())

src_subdir = "../src/"
if src_subdir not in sys.path:
    sys.path.insert(1, src_subdir)

from data import read_catalog, generate_splits, process_all_splits
from utils import setup_paths, load_vsi
from vsi import VSI

np.random.seed(42)
torch.manual_seed(42)

def timeout_handler(signum, frame):
    """Handler for SIGALRM that raises a TimeoutError."""
    raise TimeoutError

def prompt_save_checkpoint(message, timeout=15):
    """
    Prompt the user with a checkpoint message and wait for input.
    If the user types 's' (skip) within `timeout` seconds, skip saving.
    Otherwise, if no input is received within the timeout, return True to save.
    
    Parameters:
      - message: The checkpoint message to display.
      - timeout: Timeout in seconds to wait for user input.
    
    Returns:
      - True if the checkpoint should be saved; False if saving is skipped.
    """
    print("\n" + "="*50)
    print(message)
    print(f"Press 's' and Enter within {timeout} seconds to skip saving this checkpoint,")
    print("or simply wait to auto-save the checkpoint.")
    print("="*50)
    
    # Set up the alarm signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        user_input = input("Your input (or wait): ")
        signal.alarm(0)  # Cancel the alarm
    except TimeoutError:
        print(f"\nNo input received within {timeout} seconds; proceeding with saving checkpoint.")
        user_input = ""
        
    if user_input.strip().lower() == "s":
        print("Checkpoint save skipped.")
        return False
    return True

def run_pipeline(training_selection, test_size, calibration_size,
                 confidence_level, evaluation_grid_size, prior_method,
                 estimator_method, test_statistic_method,
                 check_save=False, make_save=True,
                 notebook=False, file="xp_apogee_cat.h5", filter="good", normalize=False,
                 spectra=["coeffs"], features=["LOGG", "TEFF", "FE_H"], n_coeffs=110):
    print(f"\n==== Running Configuration: training_selection={training_selection}, test_size={test_size}, calibration_size={calibration_size} ====")

    data_paths = setup_paths(notebook=notebook) # control
    file_name = file # control
    catalog_path = os.path.join(data_paths["data"], file_name)
    catalog = read_catalog(catalog_path, filter=filter, normalize=normalize) # control

    splits = generate_splits(catalog, 
                             training_selection=training_selection, # control
                             test_size=test_size, # control
                             prop_calibration=calibration_size) # control
    processed = process_all_splits(splits,
                                   spectra=spectra, # control
                                   features=features, # control
                                   n_coeffs=n_coeffs) # control
    
    vsi_obj = VSI(
        confidence_level=confidence_level, # control
        evaluation_grid_size=evaluation_grid_size, # control
        prior_method=prior_method, # control
        estimator_method=estimator_method, # control
        test_statistic_method=test_statistic_method,
        calibration_method="critical-values", # control
        seed=42,
        notebook=notebook # control
    )
    print("\nVSI object created.")

    vsi_obj.load_data(
        X_train=processed["train"]["X"],
        y_train=processed["train"]["y"],
        X_calibration=processed["calibration"]["X"],
        y_calibration=processed["calibration"]["y"],
        X_test=processed["test"]["X"] if "test" in processed else None,
        y_test=processed["test"]["y"] if "test" in processed else None
    )

    print("\nData loaded into VSI object.")

    # EVALUATION GRID CHECKPOINT #
    if check_save:
        try:
            loaded_vsi = load_vsi(confidence_level=vsi_obj.confidence_level,
                                  evaluation_grid_size=vsi_obj.evaluation_grid_size,
                                  estimator_method=vsi_obj.estimator_method,
                                  test_statistic_method=vsi_obj.test_statistic_method,
                                  assets_dir=vsi_obj.filepaths.get("assets", "."))
            vsi_obj = loaded_vsi
            print("Existing checkpoint found for Evaluation Grid; loaded checkpoint; skipping generation.")
        except FileNotFoundError:
            print("No checkpoint found for Evaluation Grid. Generating evaluation grid...")
            vsi_obj.generate_evaluation_grid(alpha = 1e-10)
    else:
        print("\nCreating evaluation grid...")
        vsi_obj.generate_evaluation_grid(alpha = 1e-10)

    if make_save:
        if prompt_save_checkpoint("Checkpoint: Evaluation Grid generated.", timeout=15):
                try:
                    vsi_obj.save()
                    print("Checkpoint (Evaluation Grid) saved successfully.")
                except Exception as e:
                    print(f"Error saving checkpoint (Evaluation Grid): {e}")

    vsi_obj.set_prior()

    # TRAIN ESTIMATOR CHECKPOINT #
    print("\nTraining estimator...")
    vsi_obj.train_estimator()

    if make_save:
        if prompt_save_checkpoint("Checkpoint: Estimator Trained.", timeout=15):
                try:
                    vsi_obj.save()
                    print("Checkpoint (Estimator Trained) saved successfully.")
                except Exception as e:
                    print(f"Error saving checkpoint (Estimator Trained): {e}")

    vsi_obj.build_test_statistic()
    vsi_obj.fit_lf2i()
    print("\nLF2I model instantiated.")

    # CONFIDENCE SETS CHECKPOINT #
    confidence_sets = vsi_obj.inference_lf2i()

    if make_save:
        if prompt_save_checkpoint("Checkpoint: LF2I Inference", timeout=15):
                try:
                    vsi_obj.save()
                    print("Checkpoint (LF2I Inference) saved successfully.")
                except Exception as e:
                    print(f"Error saving checkpoint (LF2I Inference): {e}")
    
    # LF2I DIAGNOSTICS CHECKPOINT #
    diagnostics_lf2i = vsi_obj.diagnostics_lf2i()

    if make_save:
        if prompt_save_checkpoint("Checkpoint: LF2I Diagnostics", timeout=15):
                try:
                    vsi_obj.save()
                    print("Checkpoint (LF2I Diagnostics) saved successfully.")
                except Exception as e:
                    print(f"Error saving checkpoint (LF2I Diagnostics): {e}")

    # HPD SETS CHECKPOINT #
    indicators, credible_sets = vsi_obj.inference_estimator()

    if make_save:
        if prompt_save_checkpoint("Checkpoint: Estimator Inference", timeout=15):
                try:
                    vsi_obj.save()
                    print("Checkpoint (Estimator Inference) saved successfully.")
                except Exception as e:
                    print(f"Error saving checkpoint (Estimator Inference): {e}")

    # HPD DIAGNOSTICS CHECKPOINT #
    diagnostics_estimator = vsi_obj.diagnostics_estimator(indicators=indicators)

    if make_save:
        if prompt_save_checkpoint("Checkpoint: Estimator Diagnostics", timeout=15):
                try:
                    vsi_obj.save()
                    print("Checkpoint (Estimator Diagnostics) saved successfully.")
                except Exception as e:
                    print(f"Error saving checkpoint (Estimator Diagnostics): {e}")
    
    vsi_obj.save()
    print(f"\n==== PIPELINE COMPLETE FOR CONFIGURATION: training_selection={training_selection}, test_size={test_size}, calibration_size={calibration_size} ====")

if __name__ == "__main__":
    print("=== TEST THIS IN A SEPARATE FILE ====")