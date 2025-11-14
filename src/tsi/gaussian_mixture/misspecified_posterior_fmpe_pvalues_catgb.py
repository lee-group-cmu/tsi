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
# import seaborn as sns
import torch
from torch.distributions import MultivariateNormal
from sbi.inference import FMPE #, SNPE, NPSE
from sbi.analysis import pairplot
from sbi.utils import BoxUniform
from sbi.simulators.gaussian_mixture import gaussian_mixture
import sbibm
from lf2i.inference import LF2I
from lf2i.test_statistics.posterior import Posterior
from lf2i.test_statistics.waldo import Waldo
from lf2i.calibration.critical_values import train_qr_algorithm
from lf2i.calibration.p_values import augment_calibration_set
from lf2i.utils.other_methods import hpd_region
from lf2i.plot.parameter_regions import plot_parameter_regions
from lf2i.plot.coverage_diagnostics import coverage_probability_plot
from lf2i.plot.power_diagnostics import set_size_plot
from tsi.common.monotone_nn import train_monotonic_nn, MonotonicNN
from tsi.common.utils import create_experiment_hash, IntList, TrainingLogger
from tsi.temp.utils import kdeplots2D
from sklearn.base import BaseEstimator, ClassifierMixin
from pygam import LogisticGAM, s, te, l, f
import numpy as np

class GAMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, add_radius=True):
        """
        add_radius: If True, expects X of shape (N, 3) and adds radius as 4th column
                    If False, expects X of shape (N, 4) with radius already included
        """
        self.add_radius = add_radius
        self.classes_ = np.array([0, 1])

    def _add_radius_feature(self, X):
        """Add radius as ||theta|| = sqrt(X[:, 1]^2 + X[:, 2]^2)"""
        if not self.add_radius:
            return X
        
        if X.shape[1] != 3:
            raise ValueError(f"Expected X with 3 columns when add_radius=True, got {X.shape[1]}")
        
        radius = np.linalg.norm(X[:, [1, 2]], axis=1, keepdims=True)
        return np.column_stack([X, radius])

    def _build_formula(self):
        """Reconstruct the formula from config"""
        formula = (
            s(0, constraints='monotonic_inc', n_splines=12, spline_order=3) +
            l(3) +
            te(0, 3, n_splines=3, spline_order=2) +
            s(3, constraints='concave', n_splines=6, spline_order=2)
        )
        return formula

    def fit(self, X, y):
        """
        Fit the GAM model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Features [t, theta1, theta2]
        y : array-like of shape (n_samples,)
            Target values
        """
        # Add radius feature
        X_with_radius = self._add_radius_feature(X)
        
        # Reconstruct the formula inside fit (after cloning)
        self.gam_model_ = LogisticGAM(self._build_formula(), tol=1e-4)
        self.gam_model_.gridsearch(
            X_with_radius,
            y,
            lam=np.logspace(-1, 3, 9),
            progress=False
        )
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Features [t, theta1, theta2]
        
        Returns
        -------
        array of shape (n_samples,)
            Predicted class labels
        """
        X_with_radius = self._add_radius_feature(X)
        return self.gam_model_.predict(X_with_radius)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            Features [t, theta1, theta2]
        
        Returns
        -------
        array of shape (n_samples, 2)
            Predicted probabilities for each class [P(class=0), P(class=1)]
        """
        X_with_radius = self._add_radius_feature(X)
        probs = self.gam_model_.predict_proba(X_with_radius)
        return np.column_stack([1 - probs, probs])


@click.command()
@click.option('--num-augment', default=10, help='Number of augmentation samples', type=int)
def main(num_augment):
    asset_dir = 'results/concept_shift/gaussian_reference/posterior_fmpe/p_values_catgb'
    experiment_dir = f'results/concept_shift/gaussian_reference/posterior_fmpe/p_values_catgb'
    os.makedirs(Path(experiment_dir), exist_ok=True)

    ### Settings
    POI_DIM = 2  # parameter of interest
    PRIOR_LOC = [0, 0]
    PRIOR_VAR = 2.0 # (6*np.sqrt(2.0))**2
    POI_BOUNDS = {r'$\theta_1$': (-10, 10), r'$\theta_2$': (-10, 10)}
    PRIOR = MultivariateNormal(
        loc=torch.Tensor(PRIOR_LOC), covariance_matrix=PRIOR_VAR*torch.eye(n=POI_DIM)
    )

    B = 50_000  # num simulations to estimate posterior and test statistics
    B_PRIME = 50_000  # num simulations to estimate critical values
    B_DOUBLE_PRIME = 10_000  # num simulations to do diagnostics
    EVAL_GRID_SIZE = 25_000  # num evaluation points over parameter space to construct confidence sets
    CONFIDENCE_LEVEL = 0.954, 0.683  # 0.99

    REFERENCE = MultivariateNormal(
        loc=torch.Tensor(PRIOR_LOC), covariance_matrix=36*torch.eye(n=POI_DIM)
    )
    REFERENCE_DIAGNOSTICS = BoxUniform(
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
    task = sbibm.get_task('gaussian_mixture')
    simulator = task.get_simulator()
    SIM_PARAMS = {
        "mixture_locs_factor": [0.75, 0.75],
        "mixture_scales": [1.0, 0.1],
        "mixture_weights": [0.5, 0.5],
    }
    train_simulator = lambda theta: gaussian_mixture(theta, mixture_locs_factor=SIM_PARAMS['mixture_locs_factor'])

    try:
        with open(f'{asset_dir}/fmpe_concept_shift.pkl', 'rb') as f:
            fmpe_posterior = dill.load(f)
    except:
        b_params = PRIOR.sample(sample_shape=(B, ))
        b_samples = train_simulator(b_params)
        b_params.shape, b_samples.shape
        fmpe = FMPE(
            prior=PRIOR,
            device='cpu'
        )

        _ = fmpe.append_simulations(b_params, b_samples).train()
        fmpe_posterior = fmpe.build_posterior()
        with open(f'{asset_dir}/fmpe_concept_shift.pkl', 'wb') as f:
            dill.dump(fmpe_posterior, f)
    try:
        with open(f'{asset_dir}/obs_x_theta.pkl', 'rb') as f:
            examples = dill.load(f)
            true_theta = examples['true_theta']
            obs_x = examples['obs_x']
    except:
        true_theta = torch.Tensor([[-8.5, -8.5], [-8.5, 8.5], [8.5, -8.5], [8.5, 8.5], [-3.5, -3.5], [-3.5, 3.5], [3.5, -3.5], [3.5, 3.5], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])
        obs_x = simulator(true_theta)
        with open(f'{asset_dir}/obs_x_theta.pkl', 'wb') as f:
            dill.dump({
                'true_theta': true_theta,
                'obs_x': obs_x
            }, f)

    try:
        with open(f'{experiment_dir}/lf2i_strong_prior.pkl', 'rb') as f:
            lf2i = dill.load(f)
        with open(f'{experiment_dir}/confidence_sets_strong_prior.pkl', 'rb') as f:
            confidence_sets = dill.load(f)
    except Exception as e:
        print(e)
        lf2i = LF2I(test_statistic=Posterior(poi_dim=POI_DIM, estimator=fmpe_posterior, n_jobs=1))

        b_prime_params = REFERENCE.sample(sample_shape=(B_PRIME, ))
        b_prime_samples = simulator(b_prime_params)
        confidence_sets = lf2i.inference(
            x=obs_x,
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='p-values',
            calibration_model='cat-gb',
            calibration_model_kwargs={
                'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
                'n_iter': 25
            },
            T_prime=(b_prime_params, b_prime_samples),
            retrain_calibration=False
        )
        with open(f'{experiment_dir}/lf2i_strong_prior.pkl', 'wb') as f:
            dill.dump(lf2i, f)
        with open(f'{experiment_dir}/confidence_sets_strong_prior.pkl', 'wb') as f:
            dill.dump(confidence_sets, f)


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

        if idx_obs <= 4:
            title = r'\textbf{a)} Prior poorly aligned with $\theta^{\star}$'
        else:
            title = r'\textbf{b)} Prior well aligned with $\theta^{\star}$'

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

    try:
        with open(f'{experiment_dir}/diagn_confset_strong_prior.pkl', 'rb') as f:
            diagn_objects = dill.load(f)
        with open(f'{experiment_dir}/b_double_prime.pkl', 'rb') as f:
            b_double_prime = dill.load(f)
            b_double_prime_params, b_double_prime_samples = b_double_prime['params'], b_double_prime['samples']
        print(f'Loaded diagnostics stuff...')
    except:
        b_double_prime_params = REFERENCE_DIAGNOSTICS.sample(sample_shape=(B_DOUBLE_PRIME, ))
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
                coverage_estimator='cat-gb',
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


if __name__ == "__main__":
    main()
