import posteriors as pos
import os
import pickle as pkl
import numpy as np
import lf2i.diagnostics.coverage_probability as cp
import nde_models


def hpd_coverage_metrics(config, out_file, model, test_ds, eval_grid_size, confidence_level, limit_count=None, overwrite=False):
    
    is_npse = None 
    if isinstance(model, nde_models.WeightedNPSE):
        is_npse = True
    if isinstance(model, nde_models.WeightedFMPE):
        is_npse = False

    if os.path.exists(out_file) and not overwrite:
        with open(out_file, "rb") as file:
            joint_hpd_hits, joint_hpd_sizes, hpd_param_values, hpd_weights, event_ids = pkl.load(file)
    else:
        joint_hpd_hits, joint_hpd_sizes, hpd_param_values, hpd_weights, event_ids = pos.joint_hpd_coverage_and_size(
            confidence_level,
            model,
            test_ds,
            config.train_param_mins,
            config.train_param_maxes,
            config.eval_param_mins,
            config.eval_param_maxes,
            eval_grid_size,
            limit_count=limit_count,
            no_azimuth=config.no_azimuth,
            is_npse=is_npse
        )
    with open(out_file, "wb") as file:
        pkl.dump((joint_hpd_hits, joint_hpd_sizes, hpd_param_values, hpd_weights, event_ids), file)
    
    return joint_hpd_hits, joint_hpd_sizes, np.array(hpd_param_values), hpd_weights
        
def lf2i_coverage_metrics(config, out_file, model, test_ds, eval_grid_size, qr, limit_count=None, overwrite=False):
    
    is_npse = None 
    if isinstance(model, nde_models.WeightedNPSE):
        is_npse = True
    if isinstance(model, nde_models.WeightedFMPE):
        is_npse = False
    
    if os.path.exists(out_file) and not overwrite:
        with open(out_file, "rb") as file:
            lf2i_param_values, joint_lf2i_sizes, joint_lf2i_hits, weights = pkl.load(file)
    else:
        lf2i_param_values, joint_lf2i_sizes, joint_lf2i_hits, weights = pos.lf2i_coverage_and_size(
            qr,
            model,
            test_ds,
            config.train_param_mins,
            config.train_param_maxes,
            config.eval_param_mins,
            config.eval_param_maxes,
            eval_grid_size,
            config.calibration_num_posterior_samples,
            limit_count=limit_count,
            no_azimuth=config.no_azimuth,
            is_npse=is_npse,
            use_posterior=config.use_posterior
        )
        with open(out_file, "wb") as file:
            pkl.dump((lf2i_param_values, joint_lf2i_sizes, joint_lf2i_hits, weights), file)

    return joint_lf2i_hits, joint_lf2i_sizes, lf2i_param_values, weights


def estimate_coverage(config, out_file, eval_grid_size, hits, param_values, overwrite=False):
    param_grid = pos.get_param_grid(
        config.eval_param_mins,
        config.eval_param_maxes,
        eval_grid_size
    )

    if overwrite or not os.path.exists(out_file):
        print("Estimating Coverage")
        coverage_estimator, out_parameters, mean_proba, upper_proba, lower_proba = cp.estimate_coverage_proba(
            np.array(hits),
            np.array(param_values),
            'cat-gb',
            estimator_kwargs=None,
            param_dim=2 if config.no_azimuth else 3,
            new_parameters=param_grid.numpy()
        )
        with open(out_file, "wb") as file:
            pkl.dump((coverage_estimator, out_parameters, mean_proba, upper_proba, lower_proba), file)
    else:
        with open(out_file, "rb") as file:
            coverage_estimator, out_parameters, mean_proba, upper_proba, lower_proba = pkl.load(file)
            
    return coverage_estimator, out_parameters, mean_proba, upper_proba, lower_proba
        
    