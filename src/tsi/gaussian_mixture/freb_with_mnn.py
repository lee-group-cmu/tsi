from tqdm import tqdm
import dill
import click
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import torch
from torch.distributions import MultivariateNormal
from sbi.inference import FMPE, SNPE, NPSE
from sbi.analysis import pairplot
from sbi.utils import BoxUniform
import sbibm
from lf2i.inference import LF2I
from lf2i.test_statistics.posterior import Posterior
from lf2i.test_statistics.waldo import Waldo
from lf2i.calibration.critical_values import train_qr_algorithm
from lf2i.utils.other_methods import hpd_region
from lf2i.plot.parameter_regions import plot_parameter_regions
from lf2i.plot.coverage_diagnostics import coverage_probability_plot
from lf2i.plot.power_diagnostics import set_size_plot
from tsi.common.monotone_nn import train_monotonic_nn
from tsi.common.utils import create_experiment_hash, IntList, TrainingLogger
from tsi.temp.utils import kdeplots2D


@click.command()
@click.option('--hidden-layers', default='64,32', help='Hidden layer sizes (comma-separated)', type=IntList())
@click.option('--num-augment', default=10, help='Number of augmentation samples', type=int)
@click.option('--batch-size', default=128, help='Training batch size', type=int)
@click.option('--lr', default=1e-3, help='Learning rate', type=float)
@click.option('--weight-decay', default=1e-5, help='Weight decay for regularization', type=float)
@click.option('--n-epochs', default=100, help='Number of training epochs', type=int)
@click.option('--num-workers', default=1, help='Number of data loading workers', type=int)
@click.option('--device', default='cpu', help='Device for computation', type=click.Choice(['cpu', 'cuda', 'mps']))
@click.option('--lambda-gp', default=0.0, help='Gradient penalty weight', type=float)
@click.option('--dropout-rate', default=0.0, help='Dropout probability for hidden layers', type=float)
def main(hidden_layers,
         num_augment,
         batch_size, 
         lr,
         weight_decay,
         n_epochs,
         num_workers,
         device,
         lambda_gp,
         dropout_rate):
    EXPERIMENT_ID = create_experiment_hash(locals())
    FREB_KWARGS = {
        'num_augment': num_augment,
        'hidden_layers': list(hidden_layers),
        'DEVICE': device,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'lr': lr,
        'weight_decay': weight_decay,
        'n_epochs': n_epochs,
        'lambda_gp': lambda_gp,
        'dropout_rate': dropout_rate,
    }

    experiment_dir = f"results/{EXPERIMENT_ID}"
    os.makedirs(Path(experiment_dir), exist_ok=True)

    ### Settings
    POI_DIM = 2  # parameter of interest
    PRIOR_LOC = [0, 0]
    PRIOR_VAR = 2.0 # (6*np.sqrt(2.0))**2
    POI_BOUNDS = {r'$\theta_1$': (-10, 10), r'$\theta_2$': (-10, 10)}
    PRIOR = MultivariateNormal(
        loc=torch.Tensor(PRIOR_LOC), covariance_matrix=PRIOR_VAR*torch.eye(n=POI_DIM)
    )

    B = 100_000  # num simulations to estimate posterior and test statistics
    B_PRIME = 50_000  # num simulations to estimate critical values
    B_DOUBLE_PRIME = 30_000  # num simulations to do diagnostics
    EVAL_GRID_SIZE = 30_000  # num evaluation points over parameter space to construct confidence sets
    CONFIDENCE_LEVEL = 0.954, 0.683  # 0.99

    # REFERENCE = BoxUniform(
    #     low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0]-1, POI_BOUNDS[r'$\theta_2$'][0]-1)),
    #     high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1]+1, POI_BOUNDS[r'$\theta_2$'][1]+1))
    # )
    REFERENCE = PRIOR
    EVAL_GRID_DISTR = BoxUniform(
        low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0], POI_BOUNDS[r'$\theta_2$'][0])),
        high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1], POI_BOUNDS[r'$\theta_2$'][1]))
    )

    POSTERIOR_KWARGS = {
        # 'norm_posterior': None
    }
    DEVICE = 'cpu'
    task = sbibm.get_task('gaussian_mixture')
    simulator = task.get_simulator()
    try:
        with open(f'{experiment_dir}/snpe_strong_prior.pkl', 'rb') as f:
            snpe_posterior = dill.load(f)
    except:
        b_params = PRIOR.sample(sample_shape=(B, ))
        b_samples = simulator(b_params)
        b_params.shape, b_samples.shape
        snpe = SNPE(
            prior=PRIOR,
            density_estimator='maf',
            device='cpu'
        )

        _ = snpe.append_simulations(b_params, b_samples).train()
        snpe_posterior = snpe.build_posterior()
        with open(f'{experiment_dir}/snpe_strong_prior.pkl', 'wb') as f:
            dill.dump(snpe_posterior, f)
    b_prime_params = REFERENCE.sample(sample_shape=(B_PRIME, ))
    b_prime_samples = simulator(b_prime_params)
    b_prime_params.shape, b_prime_samples.shape
    try:
        with open(f'{experiment_dir}/obs_x_theta.pkl', 'rb') as f:
            examples = dill.load(f)
            true_theta = examples['true_theta']
            obs_x = examples['obs_x']
    except:
        true_theta = torch.Tensor([[-8.5, -8.5], [-8.5, 8.5], [8.5, -8.5], [8.5, 8.5], [-3.5, -3.5], [-3.5, 3.5], [3.5, -3.5], [3.5, 3.5], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])
        obs_x = simulator(true_theta)
        with open(f'{experiment_dir}/obs_x_theta.pkl', 'wb') as f:
            dill.dump({
                'true_theta': true_theta,
                'obs_x': obs_x
            }, f)

    lf2i = LF2I(test_statistic=Posterior(poi_dim=POI_DIM, estimator=snpe_posterior,))
    logger = TrainingLogger(f'{experiment_dir}/logs')
    model = train_monotonic_nn(
        T_prime=(b_prime_params, b_prime_samples),
        test_statistic=lf2i.test_statistic,
        config=FREB_KWARGS,
        logger=logger
    )
    logger.save_losses()
    logger.save_losses_csv()
    logger.plot_training_curves()
    logger.print_summary()

    try:
        with open(f'{experiment_dir}/lf2i_strong_prior.pkl', 'rb') as f:
            lf2i = dill.load(f)
        confidence_sets = lf2i.inference(
            x=obs_x,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='p-values',
            calibration_model=model,
            calibration_model_kwargs={
                # 'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
                # 'n_iter': 25
                'cv': {'iterations': [100], 'depth': [3]},
                'n_iter': 1
            },
            T_prime=(b_prime_params, b_prime_samples),
            retrain_calibration=False
        )
    except:
        lf2i = LF2I(test_statistic=Posterior(poi_dim=2, estimator=snpe_posterior, **POSTERIOR_KWARGS))
        confidence_sets = lf2i.inference(
            x=obs_x,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='p-values',
            calibration_model=model,
            calibration_model_kwargs={
                # 'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
                # 'n_iter': 25
                'cv': {'iterations': [100], 'depth': [3]},
                'n_iter': 1
            },
            T_prime=(b_prime_params, b_prime_samples),
            retrain_calibration=False
        )
        with open(f'{experiment_dir}/lf2i_strong_prior.pkl', 'wb') as f:
            dill.dump(lf2i, f)

    remaining = len(obs_x)
    credible_sets = []
    for x in obs_x:  # torch.vstack([task.get_observation(i) for i in range(1, 11)])
        print(f'Remaining: {remaining}', flush=True)
        credible_sets_x = []
        for cl in CONFIDENCE_LEVEL:
            actual_cred_level, credible_set = hpd_region(
                posterior=snpe_posterior,
                param_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
                x=x.reshape(-1, ),
                credible_level=cl,
                num_level_sets=10_000,
                **POSTERIOR_KWARGS
            )
            #print(actual_cred_level, flush=True)
            credible_sets_x.append(credible_set)
        credible_sets.append(credible_sets_x)
        remaining -= 1
    plt.rc('text', usetex=True)  # Enable LaTeX
    plt.rc('font', family='serif')  # Use a serif font (e.g., Computer Modern)
    plt.rcParams['text.latex.preamble'] = r'''
        \usepackage{amsmath}  % For \mathbb
        \usepackage{amssymb}  % For \mathbb
        \usepackage{bm}       % For bold math symbols
        \usepackage{underscore} % If underscores are needed
    '''

    for idx_obs, _ in enumerate(obs_x):

        if idx_obs <= 4:
            title = r'\textbf{a)} Prior poorly aligned with $\theta^{\star}$'
        else:
            title = r'\textbf{b)} Prior well aligned with $\theta^{\star}$'

        plot_parameter_regions(
            *credible_sets[idx_obs], #*[confidence_sets[j][idx_obs] for j in range(len(CONFIDENCE_LEVEL))],
            param_dim=2,
            true_parameter=true_theta[idx_obs, :],
            prior_samples=PRIOR.sample(sample_shape=(50_000, )).numpy(),
            parameter_space_bounds={
                r'$\theta_1$': dict(zip(['low', 'high'], POI_BOUNDS[r'$\theta_1$'])), 
                r'$\theta_2$': dict(zip(['low', 'high'], POI_BOUNDS[r'$\theta_2$'])), 
            },
            # parameter_space_bounds={
            #     r'$\theta_1$': dict(zip(['low', 'high'], [-1.0, 1.0])), 
            #     r'$\theta_2$': dict(zip(['low', 'high'], [-1.0, 1.0])), 
            # },
            colors=[
                'purple', 'deeppink', # 'hotpink',  # credible sets
                #'teal', 'mediumseagreen', 'darkseagreen', # confidence sets
            ],
            region_names=[
                *[f'HPD {int(cl*100):.0f}\%' for cl in CONFIDENCE_LEVEL],
                #*[f'CS {cl*100:.1f}%' for cl in CONFIDENCE_LEVEL],
            ],
            labels=[r'$\theta_1$', r'$\theta_2$'],
            linestyles=['-', '--'],  # , ':'
            param_names=[r'$\theta_1$', r'$\theta_2$'],
            alpha_shape=False,
            alpha=3,
            scatter=True,
            figsize=(5, 5),
            save_fig_path=f'{experiment_dir}/hpd{idx_obs}.png',
            remove_legend=True,
            title='HPD Regions',
            custom_ax=None
        )

        plot_parameter_regions(
            *[confidence_sets[j][idx_obs] for j in range(len(CONFIDENCE_LEVEL))],
            param_dim=2,
            true_parameter=true_theta[idx_obs, :],
            prior_samples=PRIOR.sample(sample_shape=(50_000, )).numpy(),
            parameter_space_bounds={
                r'$\theta_1$': dict(zip(['low', 'high'], POI_BOUNDS[r'$\theta_1$'])), 
                r'$\theta_2$': dict(zip(['low', 'high'], POI_BOUNDS[r'$\theta_2$'])), 
            },
            colors=[
                #'purple', 'deeppink', 'hotpink',  # credible sets
                'teal', 'mediumseagreen', # 'darkseagreen', # confidence sets
            ],
            region_names=[
                #*[f'HPD {cl*100:.1f}%' for cl in CONFIDENCE_LEVEL],
                *[f'FreB {int(cl*100):.0f}\%' for cl in CONFIDENCE_LEVEL],
            ],
            labels=[r'$\theta_1$', r'$\theta_2$'],
            linestyles=['-', '--'],  # , ':'
            param_names=[r'$\theta_1$', r'$\theta_2$'],
            alpha_shape=False,
            alpha=3,
            scatter=True,
            figsize=(5, 5),
            save_fig_path=f'{experiment_dir}/freb{idx_obs}.png',
            remove_legend=True,
            title='FreB with Posterior',
            custom_ax=None
        )

if __name__ == "__main__":
    main()