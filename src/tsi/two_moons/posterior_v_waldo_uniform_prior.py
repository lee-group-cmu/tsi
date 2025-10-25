from tqdm import tqdm
import dill
import numpy as np
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

### Settings
POI_DIM = 2  # parameter of interest
PRIOR_LOC = [0, 0]
PRIOR_VAR = 0.1 # (6*np.sqrt(2.0))**2
POI_BOUNDS = {r'$\theta_1$': (-1, 1), r'$\theta_2$': (-1, 1)}
PRIOR = BoxUniform(
    low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0]-1, POI_BOUNDS[r'$\theta_2$'][0]-1)),
    high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1]+1, POI_BOUNDS[r'$\theta_2$'][1]+1))
)

B = 100_000  # num simulations to estimate posterior and test statistics
B_PRIME = 50_000  # num simulations to estimate critical values
B_DOUBLE_PRIME = 30_000  # num simulations to do diagnostics
EVAL_GRID_SIZE = 50_000  # num evaluation points over parameter space to construct confidence sets
CONFIDENCE_LEVEL = 0.954, 0.683  # 0.99

REFERENCE = BoxUniform(
    low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0]-1, POI_BOUNDS[r'$\theta_2$'][0]-1)),
    high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1]+1, POI_BOUNDS[r'$\theta_2$'][1]+1))
)
EVAL_GRID_DISTR = BoxUniform(
    low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0], POI_BOUNDS[r'$\theta_2$'][0])),
    high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1], POI_BOUNDS[r'$\theta_2$'][1]))
)

POSTERIOR_KWARGS = {
    'norm_posterior': None
}
DEVICE = 'cpu'
task = sbibm.get_task('two_moons')
simulator = task.get_simulator()

### NDE
try:
    with open('results/uniform_prior/fmpe_uniform_prior.pkl', 'rb') as f:
        fmpe_posterior = dill.load(f)
except:
    b_params = PRIOR.sample(sample_shape=(B, ))
    b_samples = simulator(b_params)
    b_params.shape, b_samples.shape
    fmpe = FMPE(
        prior=PRIOR,
        # density_estimator='maf',
        device='cpu'
    )

    _ = fmpe.append_simulations(b_params, b_samples).train()
    fmpe_posterior = fmpe.build_posterior()
    with open('results/uniform_prior/fmpe_uniform_prior.pkl', 'wb') as f:
        dill.dump(fmpe_posterior, f)

### VSI
b_prime_params = REFERENCE.sample(sample_shape=(B_PRIME, ))
b_prime_samples = simulator(b_prime_params)
b_prime_params.shape, b_prime_samples.shape
try:
    with open('results/uniform_prior/obs_x_theta.pkl', 'rb') as f:
        examples = dill.load(f)
        true_theta = examples['true_theta']
        obs_x = examples['obs_x']
except:
    true_theta = torch.Tensor([[0, 0], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    obs_x = simulator(true_theta)
    with open('results/uniform_prior/obs_x_theta.pkl', 'wb') as f:
        dill.dump({
            'true_theta': true_theta,
            'obs_x': obs_x
        }, f)

try:
    with open('results/uniform_prior/lf2i_uniform_prior.pkl', 'rb') as f:
        lf2i = dill.load(f)
    confidence_sets = lf2i.inference(
        x=obs_x,
        evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
        confidence_level=CONFIDENCE_LEVEL,
        calibration_method='critical-values',
        calibration_model='cat-gb',
        calibration_model_kwargs={
            'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
            'n_iter': 25
        },
        T_prime=(b_prime_params, b_prime_samples),
        retrain_calibration=False
    )
except:
    lf2i = LF2I(test_statistic=Posterior(poi_dim=2, estimator=fmpe_posterior, **POSTERIOR_KWARGS))
    confidence_sets = lf2i.inference(
        x=obs_x,
        evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
        confidence_level=CONFIDENCE_LEVEL,
        calibration_method='critical-values',
        calibration_model='cat-gb',
        calibration_model_kwargs={
            'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
            'n_iter': 25
        },
        T_prime=(b_prime_params, b_prime_samples),
        retrain_calibration=False
    )
    with open('results/uniform_prior/lf2i_uniform_prior.pkl', 'wb') as f:
        dill.dump(lf2i, f)

try:
    with open('results/uniform_prior/lf2i_uniform_prior_waldo.pkl', 'rb') as f:
        lf2iw = dill.load(f)
    confidence_setsw = lf2iw.inference(
        x=obs_x,
        evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
        confidence_level=CONFIDENCE_LEVEL,
        calibration_method='critical-values',
        calibration_model='cat-gb',
        calibration_model_kwargs={
            'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
            'n_iter': 25
        },
        T_prime=(b_prime_params, b_prime_samples),
        retrain_calibration=False
    )
except:
    lf2iw = LF2I(test_statistic=Waldo(poi_dim=2, estimator=fmpe_posterior, estimation_method='posterior', num_posterior_samples=10_000,))
    confidence_setsw = lf2iw.inference(
        x=obs_x,
        evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
        confidence_level=CONFIDENCE_LEVEL,
        calibration_method='critical-values',
        calibration_model='cat-gb',
        calibration_model_kwargs={
            'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
            'n_iter': 25
        },
        T_prime=(b_prime_params, b_prime_samples),
        retrain_calibration=False
    )
    with open('results/uniform_prior/lf2i_uniform_prior_waldo.pkl', 'wb') as f:
        dill.dump(lf2iw, f)

remaining = len(obs_x)
credible_sets = []
for x in obs_x:  # torch.vstack([task.get_observation(i) for i in range(1, 11)])
    print(f'Remaining: {remaining}', flush=True)
    credible_sets_x = []
    for cl in CONFIDENCE_LEVEL:
        actual_cred_level, credible_set = hpd_region(
            posterior=fmpe_posterior,
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

for idx_obs in range(8):

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
        save_fig_path=f'results/uniform_prior/hpd{idx_obs}.png',
        remove_legend=True,
        title=title,
        custom_ax=None
    )

for idx_obs in range(8):

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
        save_fig_path=f'results/uniform_prior/freb{idx_obs}.png',
        remove_legend=True,
        title='FreB with Posterior',
        custom_ax=None
    )

for idx_obs in range(8):

    plot_parameter_regions(
        *[confidence_setsw[j][idx_obs] for j in range(len(CONFIDENCE_LEVEL))],
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
        save_fig_path=f'results/uniform_prior/freb_waldo{idx_obs}.png',
        remove_legend=True,
        title='FreB with Waldo',
        custom_ax=None
    )

try:
    with open('results/uniform_prior/diagn_confset_uniform_prior.pkl', 'rb') as f:
        diagn_objects = dill.load(f)
    with open('results/uniform_prior/diagn_confset_uniform_prior_waldo.pkl', 'rb') as f:
        diagn_objectsw = dill.load(f)
    with open('results/uniform_prior/diagn_cred_uniform_prior.pkl', 'rb') as f:
        diagn_objects_cred = dill.load(f)
    with open('results/uniform_prior/b_double_prime.pkl', 'rb') as f:
        b_double_prime = dill.load(f)
        b_double_prime_params, b_double_prime_samples = b_double_prime['params'], b_double_prime['samples']
except:
    b_double_prime_params = REFERENCE.sample(sample_shape=(B_DOUBLE_PRIME, ))
    b_double_prime_samples = simulator(b_double_prime_params)
    b_double_prime_params.shape, b_double_prime_samples.shape
    with open('results/uniform_prior/b_double_prime.pkl', 'wb') as f:
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
            calibration_method='critical-values',
            coverage_estimator='splines',
            T_double_prime=(b_double_prime_params, b_double_prime_samples),
        )
        diagn_objects[cl] = (diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset)
    with open('results/uniform_prior/diagn_confset_uniform_prior.pkl', 'wb') as f:
        dill.dump(diagn_objects, f)

    plt.scatter(out_parameters_confset[:, 0], out_parameters_confset[:, 1], c=mean_proba_confset)
    plt.title('Coverage of FreB confidence sets')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig('results/uniform_prior/freb_coverage')
    plt.close()

    diagn_objectsw = {}
    for cl in CONFIDENCE_LEVEL[:1]:  # 0.954
        print(cl, flush=True)
        diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset = lf2iw.diagnostics(
            region_type='lf2i',
            confidence_level=cl,
            calibration_method='critical-values',
            coverage_estimator='splines',
            T_double_prime=(b_double_prime_params, b_double_prime_samples),
        )
        diagn_objects[cl] = (diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset)
    with open('results/uniform_prior/diagn_confset_uniform_prior_waldo.pkl', 'wb') as f:
        dill.dump(diagn_objectsw, f)

    plt.scatter(out_parameters_confset[:, 0], out_parameters_confset[:, 1], c=mean_proba_confset)
    plt.title('Coverage of Waldo confidence sets')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig('results/uniform_prior/waldo_coverage')
    plt.close()

    diagn_objects_cred = {}
    size_grid_for_sizes = 5_000
    for cl in CONFIDENCE_LEVEL[:1]:  # 0.954
        print(cl, flush=True)
        diagnostics_estimator_credible, out_parameters_credible, mean_proba_credible, upper_proba_credible, lower_proba_credible, sizes = lf2i.diagnostics(
            region_type='posterior',
            confidence_level=cl,
            coverage_estimator='splines',
            T_double_prime=(b_double_prime_params, b_double_prime_samples),
            posterior_estimator=lf2i.test_statistic.estimator,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(size_grid_for_sizes, )),
            num_level_sets=5_000,
            **POSTERIOR_KWARGS
        )
        diagn_objects_cred[cl] = (diagnostics_estimator_credible, out_parameters_credible, mean_proba_credible, upper_proba_credible, lower_proba_credible, sizes)
    with open('results/uniform_prior/diagn_cred_uniform_prior.pkl', 'wb') as f:
        dill.dump(diagn_objects_cred, f)

    plt.scatter(out_parameters_credible[:, 0], out_parameters_credible[:, 1], c=mean_proba_credible)
    plt.title('Coverage of credible regions')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig('results/uniform_prior/hpd_coverage')
    plt.close()

try:
    with open('results/uniform_prior/confidence_sets_for_size.pkl', 'rb') as f:
        confidence_sets_for_size = dill.load(f)
    with open('results/uniform_prior/set_for_size.pkl', 'rb') as f:
        set_for_size = dill.load(f)
        params_for_size = set_for_size['params']
        samples_for_size = set_for_size['samples']
except:
    params_for_size = EVAL_GRID_DISTR.sample(sample_shape=(B_DOUBLE_PRIME, ))
    samples_for_size = simulator(params_for_size)
    params_for_size.shape, samples_for_size.shape
    with open('results/uniform_prior/set_for_size.pkl', 'wb') as f:
        dill.dump({
            'params': params_for_size,
            'samples': samples_for_size
        }, f)

    size_grid_for_sizes = 5_000
    confidence_sets_for_size = lf2i.inference(
        x=samples_for_size,
        evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(size_grid_for_sizes, )),
        confidence_level=CONFIDENCE_LEVEL,
        calibration_method='critical-values',
    )
    confset_sizes = np.array([100*cs.shape[0]/size_grid_for_sizes for cs in confidence_sets_for_size[0]])
    with open('results/uniform_prior/confidence_sets_for_size.pkl', 'wb') as f:
        dill.dump(confidence_sets_for_size, f)

    set_size_plot(
        parameters=params_for_size,
        set_sizes=confset_sizes,
        param_dim=2,
        figsize=(10, 10),
        vmin_vmax=(0, 50),
        title='FreB sizes',
        save_fig_path='results/uniform_prior/freb_sizes_0_50.png'
    )

    set_size_plot(
        parameters=params_for_size,
        set_sizes=confset_sizes,
        param_dim=2,
        figsize=(10, 10),
        vmin_vmax=(0, 100),
        title='FreB sizes',
        save_fig_path='results/uniform_prior/freb_sizes.png'
    )

try:
    with open('results/uniform_prior/confidence_sets_for_size_waldo.pkl', 'rb') as f:
        confidence_sets_for_size = dill.load(f)
except:
    confidence_sets_for_size = lf2iw.inference(
        x=samples_for_size,
        evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(size_grid_for_sizes, )),
        confidence_level=CONFIDENCE_LEVEL,
        calibration_method='critical-values',
    )
    confset_sizes = np.array([100*cs.shape[0]/size_grid_for_sizes for cs in confidence_sets_for_size[0]])
    with open('results/uniform_prior/confidence_sets_for_size_waldo.pkl', 'wb') as f:
        dill.dump(confidence_sets_for_size, f)

    set_size_plot(
        parameters=params_for_size,
        set_sizes=confset_sizes,
        param_dim=2,
        figsize=(10, 10),
        vmin_vmax=(0, 50),
        title='Waldo sizes',
        save_fig_path='results/uniform_prior/waldo_sizes_0_50.png'
    )

    set_size_plot(
        parameters=params_for_size,
        set_sizes=confset_sizes,
        param_dim=2,
        figsize=(10, 10),
        vmin_vmax=(0, 100),
        title='Waldo sizes',
        save_fig_path='results/uniform_prior/waldo_sizes.png'
    )
