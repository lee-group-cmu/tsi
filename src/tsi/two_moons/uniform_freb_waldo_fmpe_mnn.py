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
from sbi.neural_nets import posterior_flow_nn
import sbibm
from lf2i.inference import LF2I
from lf2i.test_statistics.posterior import Posterior
from lf2i.test_statistics.waldo import Waldo
from lf2i.calibration.critical_values import train_qr_algorithm
from lf2i.utils.other_methods import hpd_region
from lf2i.plot.parameter_regions import plot_parameter_regions
from lf2i.plot.coverage_diagnostics import coverage_probability_plot
from lf2i.plot.power_diagnostics import set_size_plot
from tsi.common.monotone_nn import train_monotonic_nn, MonotonicNN
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
    experiment_dir = f"results/fmpe/uniform_setting/{EXPERIMENT_ID}"
    os.makedirs(Path(experiment_dir), exist_ok=True)

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
        'assets_dir': experiment_dir
    }    

    ### Settings
    POI_DIM = 2  # parameter of interest
    PRIOR_LOC = [0.5, 0.5]
    PRIOR_VAR = 0.1 # (6*np.sqrt(2.0))**2
    POI_BOUNDS = {r'$\theta_1$': (-1, 1), r'$\theta_2$': (-1, 1)}
    PRIOR = BoxUniform(
        low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0], POI_BOUNDS[r'$\theta_2$'][0])),
        high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1], POI_BOUNDS[r'$\theta_2$'][1]))
    )

    B = 50_000  # num simulations to estimate posterior and test statistics
    B_PRIME = 30_000  # num simulations to estimate critical values
    B_DOUBLE_PRIME = 10_000  # num simulations to do diagnostics
    EVAL_GRID_SIZE = 25_000  # num evaluation points over parameter space to construct confidence sets
    CONFIDENCE_LEVEL = 0.954, 0.683  # 0.99

    REFERENCE = BoxUniform(
        low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0]-1, POI_BOUNDS[r'$\theta_2$'][0]-1)),
        high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1]+1, POI_BOUNDS[r'$\theta_2$'][1]+1))
    )
    # REFERENCE = PRIOR
    EVAL_GRID_DISTR = BoxUniform(
        low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0], POI_BOUNDS[r'$\theta_2$'][0])),
        high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1], POI_BOUNDS[r'$\theta_2$'][1]))
    )

    POSTERIOR_KWARGS = {
        # 'norm_posterior': None
    }
    DEVICE = 'cpu'
    task = sbibm.get_task('two_moons')
    simulator = task.get_simulator()

    try:
        # with open(f'{experiment_dir}/fmpe_strong_prior.pkl', 'rb') as f:
        with open('results/fmpe/fmpe_uniform_prior.pkl', 'rb') as f:
            fmpe_posterior = dill.load(f)
        print('FMPE loaded...')
    except:
        b_params = PRIOR.sample(sample_shape=(B, ))
        b_samples = simulator(b_params)
        b_params.shape, b_samples.shape
        net_builder = posterior_flow_nn(
            model='mlp',
            num_layers=3,
        )
        fmpe = FMPE(
            prior=PRIOR,
            vf_estimator=net_builder,
            device='cpu',
        )

        _ = fmpe.append_simulations(b_params, b_samples).train()
        fmpe_posterior = fmpe.build_posterior()
        with open('results/fmpe/fmpe_uniform_prior.pkl', 'wb') as f:
            dill.dump(fmpe_posterior, f)

    ### TSI
    b_prime_params = REFERENCE.sample(sample_shape=(B_PRIME, ))
    b_prime_samples = simulator(b_prime_params)
    b_prime_params.shape, b_prime_samples.shape
    try:
        with open(f'{experiment_dir}/obs_x_theta.pkl', 'rb') as f:
            examples = dill.load(f)
            true_theta = examples['true_theta']
            obs_x = examples['obs_x']
    except:
        true_theta = torch.Tensor([[0, 0], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        obs_x = simulator(true_theta)
        with open(f'{experiment_dir}/obs_x_theta.pkl', 'wb') as f:
            dill.dump({
                'true_theta': true_theta,
                'obs_x': obs_x
            }, f)

    ### FreB
    try:
        with open(f'{experiment_dir}/lf2i_strong_prior.pkl', 'rb') as f:
            lf2i = dill.load(f)
        print('LF2I loaded...')
        with open(f"{experiment_dir}/input_bounds.pkl", 'rb') as f:
            input_bounds = dill.load(f)

        model = MonotonicNN(
            in_d=POI_DIM + 1,
            hidden_layers=FREB_KWARGS['hidden_layers'],
            sigmoid=True,
            input_bounds=input_bounds
        )
        model.load_state_dict(torch.load(f"{experiment_dir}/best_monotonic_nn.pt", weights_only=True))
        model.eval()
        print('MNN loaded...')

        lf2i.calibration_model = {
            'multiple_levels': model,
        }
    except Exception as e:
        print(e)

        lf2i = LF2I(test_statistic=Posterior(poi_dim=POI_DIM, estimator=fmpe_posterior,))
        logger = TrainingLogger(f'{experiment_dir}/logs')
        model, input_bounds = train_monotonic_nn(
            T_prime=(b_prime_params, b_prime_samples),
            test_statistic=lf2i.test_statistic,
            config=FREB_KWARGS,
            logger=logger
        )
        with open(f'{experiment_dir}/input_bounds.pkl', 'wb') as f:
            dill.dump(input_bounds, f)

        logger.save_losses()
        logger.save_losses_csv()
        logger.plot_training_curves()
        logger.print_summary()

        with open(f'{experiment_dir}/lf2i_strong_prior.pkl', 'wb') as f:
            dill.dump(lf2i, f)
        lf2i.calibration_model = {
            'multiple_levels': model,
        }

    try:
        with open(f'{experiment_dir}/confidence_sets_strong_prior.pkl', 'rb') as f:
            confidence_sets = dill.load(f)
    except:
        confidence_sets = lf2i.inference(
            x=obs_x,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='p-values',
            retrain_calibration=False
        )
        with open(f'{experiment_dir}/confidence_sets_strong_prior.pkl', 'wb') as f:
            dill.dump(confidence_sets, f)

    ### Waldo
    try:
        with open(f'{experiment_dir}/lf2i_strong_prior_waldo.pkl', 'rb') as f:
            lf2i_waldo = dill.load(f)
        print('LF2I loaded...')
        with open(f"{experiment_dir}/input_bounds_waldo.pkl", 'rb') as f:
            input_bounds = dill.load(f)

        model = MonotonicNN(
            in_d=POI_DIM + 1,
            hidden_layers=FREB_KWARGS['hidden_layers'],
            sigmoid=True,
            input_bounds=input_bounds
        )
        model.load_state_dict(torch.load(f"{experiment_dir}/best_monotonic_nn_waldo.pt", weights_only=True))
        model.eval()
        print('MNN loaded...')

        lf2i_waldo.calibration_model = {
            'multiple_levels': model,
        }
    except Exception as e:
        print(e)

        lf2i_waldo = LF2I(test_statistic=Waldo(poi_dim=2, estimator=fmpe_posterior, estimation_method='posterior', num_posterior_samples=5_000,))
        logger = TrainingLogger(f'{experiment_dir}/logs')
        model, input_bounds = train_monotonic_nn(
            T_prime=(b_prime_params, b_prime_samples),
            test_statistic=lf2i_waldo.test_statistic,
            config=FREB_KWARGS,
            logger=logger
        )
        with open(f'{experiment_dir}/input_bounds_waldo.pkl', 'wb') as f:
            dill.dump(input_bounds, f)

        logger.save_losses()
        logger.save_losses_csv()
        logger.plot_training_curves()
        logger.print_summary()

        with open(f'{experiment_dir}/lf2i_strong_prior_waldo.pkl', 'wb') as f:
            dill.dump(lf2i_waldo, f)
        lf2i_waldo.calibration_model = {
            'multiple_levels': model,
        }

    try:
        with open(f'{experiment_dir}/confidence_sets_strong_prior_waldo.pkl', 'rb') as f:
            confidence_sets = dill.load(f)
    except:
        confidence_sets_waldo = lf2i_waldo.inference(
            x=obs_x,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='p-values',
            retrain_calibration=False
        )
        with open(f'{experiment_dir}/confidence_sets_strong_prior_waldo.pkl', 'wb') as f:
            dill.dump(confidence_sets_waldo, f)

    plt.rc('text', usetex=True)  # Enable LaTeX
    plt.rc('font', family='serif')  # Use a serif font (e.g., Computer Modern)
    plt.rcParams['text.latex.preamble'] = r'''
        \usepackage{amsmath}  % For \mathbb
        \usepackage{amssymb}  % For \mathbb
        \usepackage{bm}       % For bold math symbols
        \usepackage{underscore} % If underscores are needed
    '''

    for idx_obs, _ in enumerate(obs_x):
        print(f'Making draft sets for pt {idx_obs}...')

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

        plot_parameter_regions(
            *[confidence_sets_waldo[j][idx_obs] for j in range(len(CONFIDENCE_LEVEL))],
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
            save_fig_path=f'{experiment_dir}/waldo{idx_obs}.png',
            remove_legend=True,
            title='FreB with Waldo',
            custom_ax=None
        )

    try:
        with open(f'{experiment_dir}/diagn_confset_strong_prior.pkl', 'rb') as f:
            diagn_objects = dill.load(f)
        with open(f'{experiment_dir}/diagn_confset_strong_prior_waldo.pkl', 'rb') as f:
            diagn_objects_waldo = dill.load(f)
        with open(f'{experiment_dir}/b_double_prime.pkl', 'rb') as f:
            b_double_prime = dill.load(f)
            b_double_prime_params, b_double_prime_samples = b_double_prime['params'], b_double_prime['samples']
        print(f'Loaded diagnostics stuff...')
    except:
        b_double_prime_params = REFERENCE.sample(sample_shape=(B_DOUBLE_PRIME, ))
        b_double_prime_samples = simulator(b_double_prime_params)
        b_double_prime_params.shape, b_double_prime_samples.shape
        with open(f'{experiment_dir}/b_double_prime.pkl', 'wb') as f:
            dill.dump({
                'params': b_double_prime_params,
                'samples': b_double_prime_samples
            }, f)

        diagn_objects = {}
        for cl in CONFIDENCE_LEVEL[:1]:  # 0.954
            print(cl, flush=True)
            diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset = lf2i.diagnostics(
                region_type='lf2i',
                confidence_level=cl,
                calibration_method='p-values',
                coverage_estimator='splines',
                T_double_prime=(b_double_prime_params, b_double_prime_samples),
            )
            diagn_objects[cl] = (diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset)
        with open(f'{experiment_dir}/diagn_confset_strong_prior.pkl', 'wb') as f:
            dill.dump(diagn_objects, f)

        plt.scatter(out_parameters_confset[:, 0], out_parameters_confset[:, 1], c=mean_proba_confset)
        plt.title('Coverage of FreB confidence sets')
        plt.clim(vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(f'{experiment_dir}/freb_coverage')
        plt.close()

        diagn_objectsw = {}
        for cl in CONFIDENCE_LEVEL[:1]:  # 0.954
            print(cl, flush=True)
            diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset = lf2i_waldo.diagnostics(
                region_type='lf2i',
                confidence_level=cl,
                calibration_method='critical-values',
                coverage_estimator='splines',
                T_double_prime=(b_double_prime_params, b_double_prime_samples),
            )
            diagn_objectsw[cl] = (diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset)
        with open(f'{experiment_dir}/diagn_confset_strong_prior_waldo.pkl', 'wb') as f:
            dill.dump(diagn_objectsw, f)

        plt.scatter(out_parameters_confset[:, 0], out_parameters_confset[:, 1], c=mean_proba_confset)
        plt.title('Coverage of Waldo confidence sets')
        plt.clim(vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(f'{experiment_dir}/waldo_coverage')
        plt.close()

    try:
        with open(f'{experiment_dir}/confidence_sets_for_size.pkl', 'rb') as f:
            confidence_sets_for_size = dill.load(f)
        with open(f'{experiment_dir}/set_for_size.pkl', 'rb') as f:
            set_for_size = dill.load(f)
            params_for_size = set_for_size['params']
            samples_for_size = set_for_size['samples']
    except:
        params_for_size = EVAL_GRID_DISTR.sample(sample_shape=(B_DOUBLE_PRIME, ))
        samples_for_size = simulator(params_for_size)
        params_for_size.shape, samples_for_size.shape
        with open(f'{experiment_dir}/set_for_size.pkl', 'wb') as f:
            dill.dump({
                'params': params_for_size,
                'samples': samples_for_size
            }, f)

        size_grid_for_sizes = 1_000
        confidence_sets_for_size = lf2i.inference(
            x=samples_for_size,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(size_grid_for_sizes, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='critical-values',
        )
        confset_sizes = np.array([100*cs.shape[0]/size_grid_for_sizes for cs in confidence_sets_for_size[0]])
        with open(f'{experiment_dir}/confidence_sets_for_size.pkl', 'wb') as f:
            dill.dump(confidence_sets_for_size, f)

        set_size_plot(
            parameters=params_for_size,
            set_sizes=confset_sizes,
            param_dim=2,
            figsize=(10, 10),
            vmin_vmax=(0, 50),
            title='FreB sizes',
            save_fig_path=f'{experiment_dir}/freb_sizes_0_50.png'
        )

        set_size_plot(
            parameters=params_for_size,
            set_sizes=confset_sizes,
            param_dim=2,
            figsize=(10, 10),
            vmin_vmax=(0, 100),
            title='FreB sizes',
            save_fig_path=f'{experiment_dir}/freb_sizes.png'
        )

    try:
        with open(f'{experiment_dir}/confidence_sets_for_size_waldo.pkl', 'rb') as f:
            confidence_sets_for_size = dill.load(f)
    except:
        confidence_sets_for_size = lf2i_waldo.inference(
            x=samples_for_size,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(size_grid_for_sizes, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='critical-values',
        )
        confset_sizes = np.array([100*cs.shape[0]/size_grid_for_sizes for cs in confidence_sets_for_size[0]])
        with open(f'{experiment_dir}/confidence_sets_for_size_waldo.pkl', 'wb') as f:
            dill.dump(confidence_sets_for_size, f)

        set_size_plot(
            parameters=params_for_size,
            set_sizes=confset_sizes,
            param_dim=2,
            figsize=(10, 10),
            vmin_vmax=(0, 50),
            title='Waldo sizes',
            save_fig_path=f'{experiment_dir}/waldo_sizes_0_50.png'
        )

        set_size_plot(
            parameters=params_for_size,
            set_sizes=confset_sizes,
            param_dim=2,
            figsize=(10, 10),
            vmin_vmax=(0, 100),
            title='Waldo sizes',
            save_fig_path=f'{experiment_dir}/waldo_sizes.png'
        )


if __name__ == "__main__":
    main()