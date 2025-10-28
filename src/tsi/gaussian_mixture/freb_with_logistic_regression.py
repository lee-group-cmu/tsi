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
from sklearn.linear_model import LogisticRegression
from lf2i.inference import LF2I
from lf2i.test_statistics.posterior import Posterior
from lf2i.test_statistics.waldo import Waldo
from lf2i.calibration.critical_values import train_qr_algorithm
from lf2i.utils.other_methods import hpd_region
from lf2i.plot.parameter_regions import plot_parameter_regions
from lf2i.plot.coverage_diagnostics import coverage_probability_plot
from lf2i.plot.power_diagnostics import set_size_plot
from tsi.common.monotone_nn import train_monotonic_nn, MonotonicNN
from tsi.common.utils import IntList, TrainingLogger
from tsi.temp.utils import kdeplots2D

### Settings
POI_DIM = 2  # parameter of interest
PRIOR_LOC = [0, 0]
PRIOR_VAR = 2.0 # (6*np.sqrt(2.0))**2
POI_BOUNDS = {r'$\theta_1$': (-10, 10), r'$\theta_2$': (-10, 10)}
PRIOR = MultivariateNormal(
    loc=torch.Tensor(PRIOR_LOC), covariance_matrix=PRIOR_VAR*torch.eye(n=POI_DIM)
)

B = 50_000  # num simulations to estimate posterior and test statistics
B_PRIME = 1_000  # num simulations to estimate critical values
B_DOUBLE_PRIME = 10_000  # num simulations to do diagnostics
EVAL_GRID_SIZE = 50_000  # num evaluation points over parameter space to construct confidence sets
CONFIDENCE_LEVEL = 0.954, 0.683  # 0.99
MIXING_PROPORTION = 0.03

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
    with open('results/test/fmpe_strong_prior.pkl', 'rb') as f:
        fmpe_posterior = dill.load(f)
except:
    # b_params = torch.vstack([
    #     PRIOR.sample(sample_shape=(int(MIXING_PROPORTION*B), )),
    #     EVAL_GRID_DISTR.sample(sample_shape=(int((1-MIXING_PROPORTION)*B), ))
    # ])
    b_params = PRIOR.sample(sample_shape=(B, ))
    b_samples = simulator(b_params)
    b_params.shape, b_samples.shape
    fmpe = FMPE(
        prior=PRIOR,
        device='cpu'
    )

    _ = fmpe.append_simulations(b_params, b_samples).train()
    fmpe_posterior = fmpe.build_posterior()
    with open('results/test/fmpe_strong_prior.pkl', 'wb') as f:
        dill.dump(fmpe_posterior, f)

try:
    with open('results/test/obs_x_theta.pkl', 'rb') as f:
        examples = dill.load(f)
        true_theta = examples['true_theta']
        obs_x = examples['obs_x']
except:
    true_theta = torch.Tensor([[-8.5, -8.5], [-8.5, 8.5], [8.5, -8.5], [8.5, 8.5], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])
    obs_x = simulator(true_theta)
    with open('results/test/obs_x_theta.pkl', 'wb') as f:
        dill.dump({
            'true_theta': true_theta,
            'obs_x': obs_x
        }, f)

try:
    with open('results/test/lf2i_strong_prior.pkl', 'rb') as f:
        lf2i = dill.load(f)
    with open('results/test/confidence_sets_strong_prior.pkl', 'rb') as f:
        confidence_sets = dill.load(f)
except:
    b_prime_params = torch.vstack([
        REFERENCE.sample(sample_shape=(int(MIXING_PROPORTION*B_PRIME), )),
        EVAL_GRID_DISTR.sample(sample_shape=(int((1-MIXING_PROPORTION)*B_PRIME), ))
    ])
    b_prime_samples = simulator(b_prime_params)
    b_prime_params.shape, b_prime_samples.shape
    lf2i = LF2I(test_statistic=Posterior(poi_dim=2, estimator=fmpe_posterior, **POSTERIOR_KWARGS))
    confidence_sets = lf2i.inference(
        x=obs_x,
        evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
        confidence_level=CONFIDENCE_LEVEL,
        calibration_method='p-values',
        calibration_model='logistic',
        T_prime=(b_prime_params, b_prime_samples),
        retrain_calibration=False
    )
    with open('results/test/lf2i_strong_prior.pkl', 'wb') as f:
        dill.dump(lf2i, f)
    with open('results/test/confidence_sets_strong_prior.pkl', 'wb') as f:
        dill.dump(confidence_sets, f)

try:
    with open('results/test/credible_sets_strong_prior.pkl', 'rb') as f:
        credible_sets = dill.load(f)
except:
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
    with open('results/test/credible_sets_strong_prior.pkl', 'wb') as f:
        dill.dump(credible_sets, f)

plt.rc('text', usetex=True)  # Enable LaTeX
plt.rc('font', family='serif')  # Use a serif font (e.g., Computer Modern)
plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{amsmath}  % For \mathbb
    \usepackage{amssymb}  % For \mathbb
    \usepackage{bm}       % For bold math symbols
    \usepackage{underscore} % If underscores are needed
'''

for idx_obs, _ in enumerate(obs_x):
    plot_parameter_regions(
        *credible_sets[idx_obs],
        param_dim=2,
        true_parameter=true_theta[idx_obs, :],
        prior_samples=PRIOR.sample(sample_shape=(5_000, )).numpy(),
        parameter_space_bounds={
            r'$\theta_1$': dict(zip(['low', 'high'], POI_BOUNDS[r'$\theta_1$'])), 
            r'$\theta_2$': dict(zip(['low', 'high'], POI_BOUNDS[r'$\theta_2$'])), 
        },
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
        save_fig_path=f'results/test/hpd{idx_obs}.png',
        remove_legend=True,
        title='HPD',
        custom_ax=None
    )

    plot_parameter_regions(
        *[confidence_sets[j][idx_obs] for j in range(len(CONFIDENCE_LEVEL))],
        param_dim=2,
        true_parameter=true_theta[idx_obs, :],
        prior_samples=PRIOR.sample(sample_shape=(5_000, )).numpy(),
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
        save_fig_path=f'results/test/freb{idx_obs}.png',
        remove_legend=True,
        title='FreB with Posterior',
        custom_ax=None
    )

try:
    with open('results/test/diagn_confset_strong_prior.pkl', 'rb') as f:
        diagn_objects = dill.load(f)
    with open('results/test/diagn_cred_strong_prior.pkl', 'rb') as f:
        diagn_objects_cred = dill.load(f)
    with open('results/test/b_double_prime.pkl', 'rb') as f:
        b_double_prime = dill.load(f)
        b_double_prime_params, b_double_prime_samples = b_double_prime['params'], b_double_prime['samples']
except:
    b_double_prime_params = EVAL_GRID_DISTR.sample(sample_shape=(B_DOUBLE_PRIME, ))
    b_double_prime_samples = simulator(b_double_prime_params)
    b_double_prime_params.shape, b_double_prime_samples.shape
    with open('results/test/b_double_prime.pkl', 'wb') as f:
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
    with open('results/test/diagn_confset_strong_prior.pkl', 'wb') as f:
        dill.dump(diagn_objects, f)

    plt.scatter(out_parameters_confset[:, 0], out_parameters_confset[:, 1], c=mean_proba_confset)
    plt.title('Coverage of FreB confidence sets')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig('results/test/freb_coverage')
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
    with open('results/test/diagn_cred_strong_prior.pkl', 'wb') as f:
        dill.dump(diagn_objects_cred, f)

    plt.scatter(out_parameters_credible[:, 0], out_parameters_credible[:, 1], c=mean_proba_credible)
    plt.title('Coverage of credible regions')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig('results/test/hpd_coverage')
    plt.close()

# Polished fig
fig, axs = plt.subplots(2, 3, figsize=(12, 10))

idx_obs = 1
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
    alpha_shape=True,
    alpha=3,
    scatter=False,
    # figsize=(5, 5),
    # save_fig_path=f'results/test/hpd{idx_obs}.png',
    remove_legend=True,
    title='Misaligned prior',
    custom_ax=axs[0, 0]
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
    alpha_shape=True,
    alpha=3,
    scatter=False,
    # figsize=(5, 5),
    # save_fig_path=f'results/test/freb{idx_obs}.png',
    remove_legend=True,
    title=None,
    custom_ax=axs[1, 0]
)

idx_obs = 4
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
    alpha_shape=True,
    alpha=3,
    scatter=False,
    # figsize=(5, 5),
    # save_fig_path=f'results/test/hpd{idx_obs}.png',
    remove_legend=True,
    title='Well-aligned prior',
    custom_ax=axs[0, 1]
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
    alpha_shape=True,
    alpha=3,
    scatter=False,
    # figsize=(5, 5),
    # save_fig_path=f'results/test/freb{idx_obs}.png',
    remove_legend=True,
    title=None,
    custom_ax=axs[1, 1]
)

with open('results/test/diagn_confset_strong_prior.pkl', 'rb') as f:
    diagn_objects = dill.load(f)
with open('results/test/diagn_cred_strong_prior.pkl', 'rb') as f:
    diagn_objects_cred = dill.load(f)

diagnostics_estimator_credible, out_parameters_credible, mean_proba_credible, upper_proba_credible, lower_proba_credible, sizes = diagn_objects_cred[CONFIDENCE_LEVEL[0]]
diagnostics_estimator_confset, out_parameters_confset, mean_proba_confset, upper_proba_confset, lower_proba_confset = diagn_objects[CONFIDENCE_LEVEL[0]]

# coverage_probability_plot(
#     parameters=out_parameters_credible,
#     coverage_probability=mean_proba_credible,
#     confidence_level=CONFIDENCE_LEVEL[0],
#     param_dim=2,
#     params_labels=[r'$\theta_1$', r'$\theta_2$'],
#     vmin_vmax=(0, 1),
#     custom_ax=axs[0,2],  # if passing custom ax for pairplot
#     title=None
# )
# Correct syntax
scatter = axs[0, 2].scatter(
    out_parameters_credible[:, 0], 
    out_parameters_credible[:, 1], 
    c=mean_proba_credible
)
axs[0, 2].set_title('Local diagnostics', size=25, pad=20)

# Set color limits on the ScalarMappable (scatter object), not the axis
scatter.set_clim(vmin=0, vmax=1)

# Add colorbar - needs fig and the scatter object
plt.colorbar(scatter, ax=axs[0, 2])

# coverage_probability_plot(
#     parameters=out_parameters_confset,
#     coverage_probability=mean_proba_confset,
#     confidence_level=CONFIDENCE_LEVEL[0],
#     param_dim=2,
#     params_labels=[r'$\theta_1$', r'$\theta_2$'],
#     vmin_vmax=(0, 1),
#     custom_ax=axs[1,2],  # if passing custom ax for pairplot
#     title=None
# )
scatter = axs[1, 2].scatter(
    out_parameters_confset[:, 0],
    out_parameters_confset[:, 1],
    c=mean_proba_confset
)
# axs[1, 2].set_title('Coverage of FreB sets')

scatter.set_clim(vmin=0, vmax=1)
plt.colorbar(scatter, ax=axs[1, 2])
plt.savefig('results/test/polished_fig.png', dpi=150)
plt.savefig('results/test/polished_fig.pdf', dpi=150)
plt.close()
