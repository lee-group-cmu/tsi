import click
import pickle
from tqdm import tqdm
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sbi.inference import FMPE
import os
from utils.data_generation import fetch_grid, sample_joint_distribution, sample_uniform_distribution
from utils.inference import preprocess_inputs, get_prior, get_eval_grid, get_posterior, compute_indicators_sampling_posterior, posterior_and_prior_kdeplot
from lf2i.inference import LF2I
from lf2i.test_statistics.posterior import Posterior
from lf2i.utils.other_methods import hpd_region
from lf2i.plot.parameter_regions import plot_parameter_regions
from lf2i.diagnostics.coverage_probability import compute_indicators_posterior


def main():
    """
    Steps:
    1. View priors
    2. Data generation
    3. Inference

    Current configurations:
    - Fixed:
        FLOW_TYPE = 'npe' // We 
        B = 300_000
        B_PRIME = 50_000
        B_DOUBLE_PRIME = 10_000
        EVAL_GRID_SIZE = 20_000
        DISPLAY_GRID_SIZE = 10
        NORM_POSTERIOR_SAMPLES = None
        CONFIDENCE_LEVEL = 0.9
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        POI_DIM = 5
        POIS = ['t_eff', 'logg', 'feh_surf', 'logl', 'dist']
        LABELS = [r"$T_{eff}$ (K)", r"$\log g$ (cgs)", r"$[\text{Fe/H}]_{\text{surf}}$ (relative to solar)", r"$\log L$ ($L_{\odot}$)", r"$d$ (kpc)"]
        PRIOR_SETTINGS = [-2.0, -1.0, 0.0, 1.0, 2.0]
        PRIOR_ARGS = {
            'lower_bound' : torch.tensor([2.5e3, 0.0, -4.0, -1.5, 0.0]),
            'upper_bound' : torch.tensor([1.5e4, 5.0, 0.5, 3.5, 1.0e3])
        }
        PLOT_PRIORS = False
    - Varying:
        PRIOR_HYPERPARAMETER = -2.0, -1.0, 0.0, 1.0, 2.0

    Results are structured as follows:
    - assets/
        Contains the gridmaker `mist` and all of the trained models
    - results/example_{FLOW_TYPE}/
        Contains figures and inference objects
    """
    start_time = time.time()
    FLOW_TYPE = 'npe'
    B = 300_000  # num simulations to estimate posterior anid test statistics
    B_PRIME = 100_000  # num simulations to estimate critical values
    B_DOUBLE_PRIME = 100_000  # num simulations to estimate coverage probability
    EVAL_GRID_SIZE = 50_000
    DISPLAY_GRID_SIZE = 10 # irrelevant now that grid has been defined elsewhere
    NORM_POSTERIOR_SAMPLES = None
    CONFIDENCE_LEVEL = 0.9
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    POI_DIM = 5
    POIS = ['t_eff', 'logg', 'feh_surf', 'logl', 'dist']
    LABELS = [r"$T_{eff}$ (K)",
              r"$\log g$ (cgs)",
              r"$[\text{Fe/H}]_{\text{surf}}$ (relative to solar)",
              r"$\log L$ ($L_{\odot}$)",
              r"$d$ (kpc)"]
    PRIOR_SETTINGS = [2.0] # [-2.0, -1.0, 0.0, 1.0, 2.0]
    PRIOR_ARGS = {
        'lower_bound' : torch.tensor([2.5e3, 0.0, -4.0, -1.5, 0.0]),
        'upper_bound' : torch.tensor([1.5e4, 5.0, 0.5, 3.5, 1.0e3])
    }
    PLOT_PRIORS = False # These figures have already been generated

    assets_dir = f'{os.getcwd()}/assets'
    os.makedirs(assets_dir, exist_ok=True)
    params, seds = fetch_grid(assets_dir=assets_dir) # POI grid + raw SEDs


    """
    1. VIEW PRIORS
    """


    # Set up directories
    prior_samples = []

    for PRIOR_SETTING in PRIOR_SETTINGS:
        # Get prior
        try:
            with open(f'{assets_dir}/prior_{PRIOR_SETTING}.pkl', 'rb') as f:
                prior = pickle.load(f)
        except:
            theta, x = sample_joint_distribution(params=params,
                                                seds=seds,
                                                args={'age_feh_hyperparam': PRIOR_SETTING,},
                                                n_samples=B,
                                                assets_dir=assets_dir,)
            theta_p, x_p = preprocess_inputs(theta, x, ['t_eff', 'logg', 'feh_surf', 'logl', 'dist'])
            prior = get_prior(theta_p, prior_args=PRIOR_ARGS)
            with open(f'{assets_dir}/prior_{PRIOR_SETTING}.pkl', 'wb') as f:
                pickle.dump(prior, f)

        prior_samples.append(prior.sample((1_000,)))

    if PLOT_PRIORS:
        theta_dfs = []
        # Convert theta to a pandas DataFrame
        for i, prior_sample in enumerate(prior_samples):
            theta_df_i = pd.DataFrame(prior_sample.numpy(), columns=LABELS)
            theta_df_i['set'] = str(i+1)
            theta_dfs.append(theta_df_i)

        # Convert theta to a pandas DataFrame
        theta_df = pd.concat(theta_dfs)

        # Create pairwise heatmaps
        palette = ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"]
        g = sns.pairplot(data=theta_df,
                            hue='set',
                            palette=sns.color_palette(palette, 5),
                            kind='kde',
                            diag_kind='hist')

        for ax in g.axes.ravel():
            ax.invert_xaxis()
            ax.invert_yaxis()

        plt.savefig(f'{assets_dir}/prior_heatmap.png')


    """
    2. DATA GENERATION
    """


    ## 2a. Display set
    with open(f'{assets_dir}/tryout_display_grid.pkl', 'rb') as f:
        theta_d, x_d = pickle.load(f)
    with open(f'{assets_dir}/tryout_display_grid_labels.pkl', 'rb') as f:
        display_labels = pickle.load(f)

    ## 2b. Calibration set
    try:
        with open(f'{assets_dir}/tryout_calibration_set.pkl', 'rb') as f:
            theta_c, x_c = pickle.load(f)
    except:
        params, seds = fetch_grid(assets_dir=assets_dir)
        theta, x = sample_joint_distribution(params=params,
                                                seds=seds,
                                                args={'age_feh_hyperparam': 2.0,},
                                                n_samples=B_PRIME,
                                                assets_dir=assets_dir,)
        theta_c, x_c = preprocess_inputs(theta, x, ['t_eff', 'logg', 'feh_surf', 'logl', 'dist'])
        with open(f'{assets_dir}/tryout_calibration_set.pkl', 'wb') as f:
            pickle.dump((theta_c, x_c), f)

    if PLOT_PRIORS:
        theta_df = pd.DataFrame(theta_p_c, columns=LABELS)

        # Create pairwise heatmaps
        g = sns.pairplot(data=theta_df,
                            kind='hist',
                            diag_kind='hist')

        for ax in g.axes.ravel():
            ax.invert_xaxis()
            ax.invert_yaxis()

        plt.suptitle(f'Calibration set', fontsize=20)
        plt.savefig(f'{assets_dir}/calibration_histogram.png')

    ## 2c. Evaluation grid
    try:
        with open(f'{assets_dir}/tryout_evaluation_grid.pkl', 'rb') as f:
            theta_e, x_e = pickle.load(f)
    except:
        params, seds = fetch_grid(assets_dir=assets_dir)
        theta, x = sample_joint_distribution(params=params,
                                                seds=seds,
                                                args={'age_feh_hyperparam': 2.0,},
                                                n_samples=EVAL_GRID_SIZE,
                                                assets_dir=assets_dir,)
        theta_e, x_e = preprocess_inputs(theta, x, POIS)
        with open(f'{assets_dir}/tryout_evaluation_grid.pkl', 'wb') as f:
            pickle.dump((theta_e, x_e), f)

    if PLOT_PRIORS:
        theta_df = pd.DataFrame(theta_e, columns=LABELS)

        # Create pairwise heatmaps
        g = sns.pairplot(data=theta_df,
                            kind='hist',
                            diag_kind='hist')

        for ax in g.axes.ravel():
            ax.invert_xaxis()
            ax.invert_yaxis()

        plt.suptitle(f'Evaluation grid', fontsize=20)
        plt.savefig(f'{assets_dir}/evaluation_histogram.png')

    ## 2d. Diagnostics set
    try:
        with open(f'{assets_dir}/tryout_diagnostics_set.pkl', 'rb') as f:
            theta_g, x_g = pickle.load(f)
    except:
        params, seds = fetch_grid(assets_dir=assets_dir)
        theta, x = sample_joint_distribution(params=params,
                                            seds=seds,
                                            args={'age_feh_hyperparam': 2.0,},
                                            n_samples=B_DOUBLE_PRIME,
                                            assets_dir=assets_dir,)
        theta_g, x_g = preprocess_inputs(theta, x, ['t_eff', 'logg', 'feh_surf', 'logl', 'dist'])
        with open(f'{assets_dir}/tryout_diagnostics_set.pkl', 'wb') as f:
            pickle.dump((theta_g, x_g), f)


    """
    3. INFERENCE
    """


    for PRIOR_SETTING in PRIOR_SETTINGS:
        example_dir_for_setting = f'{os.getcwd()}/results/example_{FLOW_TYPE}/setting_{PRIOR_SETTING}'
        os.makedirs(example_dir_for_setting, exist_ok=True)

        ## 3a. Prior
        # Get prior
        try:
            with open(f'{assets_dir}/prior_{PRIOR_SETTING}.pkl', 'rb') as f:
                prior = pickle.load(f)
        except:
            params, seds = fetch_grid(assets_dir=assets_dir)
            theta, x = sample_joint_distribution(params=params,
                                                seds=seds,
                                                args={'age_feh_hyperparam': PRIOR_SETTING,},
                                                n_samples=B,
                                                assets_dir=assets_dir,)
            theta_p, x_p = preprocess_inputs(theta, x, ['t_eff', 'logg', 'feh_surf', 'logl', 'dist'])
            prior = get_prior(theta_p, prior_args=PRIOR_ARGS)
            with open(f'{assets_dir}/prior_{PRIOR_SETTING}.pkl', 'wb') as f:
                pickle.dump(prior, f)

        ## 3b. Training set
        try:
            with open(f'{assets_dir}/tryout_set_{PRIOR_SETTING}.pkl', 'rb') as f:
                theta_p, x_p = pickle.load(f)
        except:
            params, seds = fetch_grid(assets_dir=assets_dir)
            theta, x = sample_joint_distribution(params=params,
                                                seds=seds,
                                                args={'age_feh_hyperparam': PRIOR_SETTING,},
                                                n_samples=B,
                                                assets_dir=assets_dir,)
            theta_p, x_p = preprocess_inputs(theta, x, POIS)
            with open(f'{assets_dir}/tryout_set_{PRIOR_SETTING}.pkl', 'wb') as f:
                pickle.dump((theta_p, x_p), f)

        ## 3c. Posterior
        try:
            with open(f'{assets_dir}/posterior_{PRIOR_SETTING}_{FLOW_TYPE}.pkl', 'rb') as f:
                posterior = pickle.load(f)
        except:
            posterior = get_posterior(theta_p, x_p, prior, flow_type=FLOW_TYPE)
            with open(f'{assets_dir}/posterior_{PRIOR_SETTING}_{FLOW_TYPE}.pkl', 'wb') as f:
                pickle.dump(posterior, f)

        ## 3d. LF2I
        try:
            with open(f'{example_dir_for_setting}/lf2i.pkl', 'rb') as f:
                lf2i = pickle.load(f)
        except:
            lf2i = LF2I(
                test_statistic=Posterior(
                    poi_dim=POI_DIM, estimator=posterior, norm_posterior_samples=NORM_POSTERIOR_SAMPLES
                )
            )
            with open(f'{example_dir_for_setting}/lf2i.pkl', 'wb') as f:
                pickle.dump(lf2i, f)

        ## 3e. Credible regions
        try:
            with open(f'{example_dir_for_setting}/credible_regions.pkl', 'rb') as f:
                credible_regions = pickle.load(f)
        except:
            credible_regions = [
                hpd_region(
                    posterior=lf2i.test_statistic.estimator,
                    param_grid=posterior.sample((100_000,), x=true_x),
                    x=true_x,
                    credible_level=CONFIDENCE_LEVEL,
                    num_level_sets=10_000,
                )
                for true_x in tqdm(x_d)
            ]
            with open(f'{example_dir_for_setting}/credible_regions.pkl', 'wb') as f:
                pickle.dump(credible_regions, f)

        if PLOT_PRIORS:
            param_space_df = pd.DataFrame(theta_e, columns=LABELS).sample(50_000)
            param_space_df['type'] = r'$\Theta$'

            for idx, (true_theta, region) in enumerate(zip(theta_d, credible_regions)):
                region_df = pd.DataFrame(region[1], columns=LABELS)#.sample(min(50_000, len(region[1])))
                region_df['type'] = 'HPD'
                truth_df = pd.DataFrame(true_theta.unsqueeze(0).numpy(), columns=LABELS)
                truth_df['type'] = 'Truth'
                theta_df = pd.concat([param_space_df, region_df.sample(1), truth_df])

                # Create pairwise heatmaps
                palette = ["#808080", "#BF40BF", "#FF0000"]
                g = sns.pairplot(data=theta_df,
                                    hue='type',
                                    palette=sns.color_palette(palette, 3),
                                    kind='hist',
                                    diag_kind='hist',)

                for ax in g.axes.ravel():
                    if ax.get_xlabel() and ax.get_ylabel():
                        sns.scatterplot(data=truth_df, x=ax.get_xlabel(), y=ax.get_ylabel(), color='red', marker='*', s=100, ax=ax)
                        sns.scatterplot(data=region_df, x=ax.get_xlabel(), y=ax.get_ylabel(), color="#BF40BF", marker='.', s=5, ax=ax)
                    ax.invert_xaxis()
                    ax.invert_yaxis()

                plt.suptitle(f'HPD Set ({display_labels[idx]})', fontsize=20)
                plt.savefig(f'{example_dir_for_setting}/hpd_for_pt_{display_labels[idx]}.png')

        ## 3f. Confidence sets
        try:
            with open(f'{example_dir_for_setting}/confidence_sets.pkl', 'rb') as f:
                confidence_sets = pickle.load(f)
        except:
            confidence_sets = [
                lf2i.inference( # TODO: LF2I returns failed QR (lack of coverage in all cases?)
                    x=true_x.unsqueeze(0),
                    evaluation_grid=torch.vstack([posterior.sample((100_000,), x=true_x.unsqueeze(0)),
                                                theta_e]),
                    confidence_level=CONFIDENCE_LEVEL,
                    calibration_method='critical-values',
                    calibration_model='cat-gb',
                    calibration_model_kwargs={
                        'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
                        'n_iter': 25
                    },
                    T_prime=(theta_c, x_c),
                    retrain_calibration=False
                )[0]
                for true_x in tqdm(x_d)
            ]
            with open(f'{example_dir_for_setting}/confidence_sets.pkl', 'wb') as f:
                pickle.dump(confidence_sets, f)
            with open(f'{example_dir_for_setting}/calibration_model.pkl', 'wb') as f:
                pickle.dump(lf2i.calibration_model, f)

        if PLOT_PRIORS:
            param_space_df = pd.DataFrame(theta_e, columns=LABELS).sample(50_000)
            param_space_df['type'] = r'$\Theta$'

            for idx, (true_theta, region) in enumerate(zip(theta_d, confidence_sets)):
                region_df = pd.DataFrame(region, columns=LABELS)#.sample(min(50_000, len(region[1])))
                region_df['type'] = 'LF2I'
                truth_df = pd.DataFrame(true_theta.unsqueeze(0).numpy(), columns=LABELS)
                truth_df['type'] = 'Truth'
                theta_df = pd.concat([param_space_df, region_df.sample(1), truth_df])

                # Create pairwise heatmaps
                palette = ["#808080", "#3cb371", "#FF0000"]
                g = sns.pairplot(data=theta_df,
                                    hue='type',
                                    palette=sns.color_palette(palette, 3),
                                    kind='hist',
                                    diag_kind='hist',)

                for ax in g.axes.ravel():
                    if ax.get_xlabel() and ax.get_ylabel():
                        sns.scatterplot(data=truth_df, x=ax.get_xlabel(), y=ax.get_ylabel(), color='red', marker='*', s=100, ax=ax)
                        sns.scatterplot(data=region_df, x=ax.get_xlabel(), y=ax.get_ylabel(), color="#3cb371", marker='.', s=5, ax=ax)
                    ax.invert_xaxis()
                    ax.invert_yaxis()

                plt.suptitle(f'LF2I Set ({display_labels[idx]})', fontsize=20)
                plt.savefig(f'{example_dir_for_setting}/lf2i_for_pt_{display_labels[idx]}.png')

        # 3g. HPD diagnostics
        try:
            with open(f'{example_dir_for_setting}/hpd_diagnostics.pkl', 'rb') as f:
                diagnostics_estimator, out_parameters, mean_proba, upper_proba, lower_proba = pickle.load(f)
        except:
            # Compute indicators.
            hpd_indicators = compute_indicators_sampling_posterior(posterior=posterior,
                                                                    parameters=theta_g, 
                                                                    samples=x_g,
                                                                    credible_level=CONFIDENCE_LEVEL,
                                                                    verbose=True,
                                                                    n_jobs=-2)

            ## Compute diagnostics.
            diagnostics_estimator_hpd, out_parameters, mean_proba, upper_proba, lower_proba = lf2i.diagnostics(
                region_type='posterior',
                confidence_level=CONFIDENCE_LEVEL,
                coverage_estimator='cat-gb',
                coverage_estimator_kwargs={
                    'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
                    'n_iter': 25
                },
                T_double_prime=(theta_g, x_g),
                new_parameters=theta_e,
                indicators=hpd_indicators,
                parameters=theta_g,
                n_jobs=-2,
                verbose=True,
            )
            with open(f'{example_dir_for_setting}/hpd_diagnostics.pkl', 'wb') as f:
                pickle.dump((diagnostics_estimator_hpd, out_parameters, mean_proba, upper_proba, lower_proba), f)

        if PLOT_PRIORS:
            plt.figure(figsize=(8, 8))
            theta_diag_df = pd.DataFrame(out_parameters, columns=LABELS)[LABELS[:2]]
            x_bins = np.histogram_bin_edges(theta_diag_df[LABELS[0]], bins='auto')
            y_bins = np.histogram_bin_edges(theta_diag_df[LABELS[1]], bins='auto')
            binned_sum_proba, xedges, yedges = np.histogram2d(theta_diag_df[LABELS[0]], theta_diag_df[LABELS[1]], bins=[x_bins, y_bins], weights=np.round(mean_proba*100, 2))
            bin_counts, xedges, yedges = np.histogram2d(theta_diag_df[LABELS[0]], theta_diag_df[LABELS[1]], bins=[x_bins, y_bins])
            heatmap_values = binned_sum_proba / bin_counts
            heatmap = plt.imshow(heatmap_values.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.xlabel(LABELS[0])
            plt.ylabel(LABELS[1])
            plt.title('HPD diagnostics')
            cb = plt.colorbar(heatmap, label='Coverage prob. (%)')
            plt.savefig(f'{example_dir_for_setting}/hpd_diagnostics.png')
            plt.close()

        # 3h. LF2I diagnostics
        try:
            with open(f'{example_dir_for_setting}/lf2i_diagnostics.pkl', 'rb') as f:
                diagnostics_estimator_lf2i, out_parameters, mean_proba, upper_proba, lower_proba = pickle.load(f)
        except:
            diagnostics_estimator_lf2i, out_parameters, mean_proba, upper_proba, lower_proba = lf2i.diagnostics(
                region_type='lf2i',
                calibration_method='critical-values',
                confidence_level=CONFIDENCE_LEVEL,
                coverage_estimator='cat-gb',
                coverage_estimator_kwargs={
                    'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
                    'n_iter': 25
                },
                T_double_prime=(theta_g, x_g),
                new_parameters=theta_e,
                n_jobs=-2,
                verbose=True,
            )
            with open(f'{example_dir_for_setting}/lf2i_diagnostics.pkl', 'wb') as f:
                pickle.dump((diagnostics_estimator_lf2i, out_parameters, mean_proba, upper_proba, lower_proba), f)

        if PLOT_PRIORS:
            plt.figure(figsize=(8, 8))
            theta_diag_df = pd.DataFrame(out_parameters, columns=LABELS)[LABELS[:2]]
            x_bins = np.histogram_bin_edges(theta_diag_df[LABELS[0]], bins='auto')
            y_bins = np.histogram_bin_edges(theta_diag_df[LABELS[1]], bins='auto')
            binned_sum_proba, xedges, yedges = np.histogram2d(theta_diag_df[LABELS[0]], theta_diag_df[LABELS[1]], bins=[x_bins, y_bins], weights=np.round(mean_proba*100, 2))
            bin_counts, xedges, yedges = np.histogram2d(theta_diag_df[LABELS[0]], theta_diag_df[LABELS[1]], bins=[x_bins, y_bins])
            heatmap_values = binned_sum_proba / bin_counts
            heatmap = plt.imshow(heatmap_values.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.xlabel(LABELS[0])
            plt.ylabel(LABELS[1])
            plt.title('LF2I diagnostics')
            cb = plt.colorbar(heatmap, label='Coverage prob. (%)')
            plt.savefig(f'{example_dir_for_setting}/lf2i_diagnostics.png')
            plt.close()

        # 3i. Plot coverage distribution for HPD
        coverage_probs_for_setting = []
        for PRIOR_SETTING in PRIOR_SETTINGS:
            # Test set
            try:
                with open(f'{assets_dir}/tryout_set_{PRIOR_SETTING}.pkl', 'rb') as f:
                    theta_t, x_t = pickle.load(f)
            except:
                params, seds = fetch_grid(assets_dir=assets_dir)
                theta, x = sample_joint_distribution(params=params,
                                                    seds=seds,
                                                    args={'age_feh_hyperparam': PRIOR_SETTING,},
                                                    n_samples=B,
                                                    assets_dir=assets_dir,)
                theta_t, x_t = preprocess_inputs(theta, x, POIS)
                with open(f'{assets_dir}/tryout_set_{PRIOR_SETTING}.pkl', 'wb') as f:
                    pickle.dump((theta_t, x_t), f)
            coverage_probs_for_setting.append(diagnostics_estimator_hpd.predict_proba(theta_t.numpy())[:, 1])

        if PLOT_PRIORS:
            sns.boxplot(np.vstack(coverage_probs_for_setting).T, color='grey', medianprops=dict(color='red'))
            plt.title('Coverage of 90% HPD Sets')
            plt.axhline(0.9, 0, 6, color='grey', linestyle='--')
            plt.xlabel('Setting')
            plt.ylim(0, 1)
            plt.savefig(f'{example_dir_for_setting}/coverage_hpd_across_settings.png')
            plt.close()

        # 3j. Plot coverage distribution for LF2I
        coverage_probs_for_setting = []
        for PRIOR_SETTING in PRIOR_SETTINGS:
            # Test set
            try:
                with open(f'{assets_dir}/tryout_set_{PRIOR_SETTING}.pkl', 'rb') as f:
                    theta_t, x_t = pickle.load(f)
            except:
                params, seds = fetch_grid(assets_dir=assets_dir)
                theta, x = sample_joint_distribution(params=params,
                                                    seds=seds,
                                                    args={'age_feh_hyperparam': PRIOR_SETTING,},
                                                    n_samples=B,
                                                    assets_dir=assets_dir,)
                theta_t, x_t = preprocess_inputs(theta, x, POIS)
                with open(f'{assets_dir}/tryout_set_{PRIOR_SETTING}.pkl', 'wb') as f:
                    pickle.dump((theta_t, x_t), f)
            coverage_probs_for_setting.append(diagnostics_estimator_lf2i.predict_proba(theta_t.numpy())[:, 1])

        if PLOT_PRIORS:
            sns.boxplot(np.vstack(coverage_probs_for_setting).T, color='grey', medianprops=dict(color='red'))
            plt.title('Coverage of 90% LF2I Sets')
            plt.axhline(0.9, 0, 6, color='grey', linestyle='--')
            plt.xlabel('Setting')
            plt.ylim(0, 1)
            plt.savefig(f'{example_dir_for_setting}/coverage_lf2i_across_settings.png')
            plt.close()

    print(f"Total time: {time.time() - start_time}")
    return


if __name__ == '__main__':
    main()
