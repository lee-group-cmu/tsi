import numpy as np
import torch
from torch.distributions import Distribution
from sbi.utils import get_kde, process_prior
from sbi.inference import FMPE, NPE, DirectPosterior
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
from matplotlib.colors import LogNorm
import seaborn as sns


def preprocess_inputs(parameters: np.ndarray, samples: np.ndarray, query_params: list):
    if query_params is not None:
        parameters = parameters[query_params]
    parameters_tensor = torch.tensor(np.array(parameters.tolist())).to(torch.float32)
    samples_tensor = torch.tensor(np.array(samples.tolist())).reshape(len(samples), -1).to(torch.float32)

    return parameters_tensor, samples_tensor


def get_prior(prior_samples: np.ndarray, prior_args: dict) -> Distribution:
    return process_prior(get_kde(samples=prior_samples, bandwidth='silvermann'),
                         custom_prior_wrapper_kwargs=prior_args)[0]


def get_eval_grid(parameters, true_theta, eval_grid_size):
    # TODO: Project or fix some components?
    select_from_grid = np.random.choice(len(parameters), eval_grid_size, replace=False)

    # If no parameters are fixed
    eval_grid_pois = torch.tensor(np.array(parameters[select_from_grid][['t_eff', 'logg', 'feh_surf', 'logl']].tolist())).float()
    dist_grid = np.logspace(-2, 2, 10_000)
    eval_grid_nps = torch.tensor(np.random.choice(dist_grid, (len(eval_grid_pois), 1))).float()
    eval_grid = torch.hstack([eval_grid_pois, eval_grid_nps])

    # If distance is fixed
    # eval_grid_pois = torch.tensor(np.array(parameters[select_from_grid][['t_eff', 'logg', 'feh_surf', 'logl']].tolist()))
    # eval_grid_nps = true_theta[4:].repeat(len(eval_grid_pois), 1)
    # eval_grid = torch.hstack([eval_grid_pois, eval_grid_nps]).float()

    # If all but first two are fixed:
    # eval_grid_pois = torch.tensor(np.array(parameters[select_from_grid][['t_eff', 'logg']].tolist()))
    # eval_grid_nps = true_theta[2:].repeat(len(eval_grid_pois), 1)
    # eval_grid = torch.hstack([eval_grid_pois, eval_grid_nps]).float()

    return eval_grid


def get_posterior(parameters: torch.Tensor, samples: torch.Tensor, prior: Distribution, flow_type: str='npe', **kwargs):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if flow_type == 'fmpe':
        net_builder = flowmatching_nn(
            model="mlp",
            z_score_theta='none',
            z_score_x='none',
            num_blocks=3,
            hidden_features=8,
        )
        trainer = FMPE(
            prior=prior,
            density_estimator=net_builder,
            device=DEVICE
        )
        _ = trainer.append_simulations(parameters, samples).train()
        posterior = trainer.build_posterior()
    elif flow_type == 'npe':
        inference = NPE()
        posterior_estimator = inference.append_simulations(parameters, samples).train()
        posterior = DirectPosterior(posterior_estimator, prior=prior)
    else:
        raise ValueError("Invalid flow type. Choose from 'fmpe' or 'npe'.")

    return posterior


def compute_indicators_sampling_posterior(posterior, parameters, samples, credible_level, verbose=False, n_jobs=-2):
    from sbi.simulators.simutils import tqdm_joblib
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import numpy as np

    def single_hpd_region(idx):
        p = next(posterior)
        _, credible_region = hpd_region(
            posterior=p,
            param_grid=torch.cat((p.sample((1_000,), x=samples[idx].unsqueeze(0)), parameters[idx].unsqueeze(0))),
            x=samples[idx],
            credible_level=credible_level,
            num_level_sets=10_000,
        )
        indicator = 1 if parameters[idx].unsqueeze(0).numpy() in credible_region else 0
        return credible_region, indicator

    with tqdm_joblib(tqdm(it:=range(samples.shape[0]), desc=f"Computing indicators for {len(it)} credible regions", total=len(it), disable=not verbose)) as _:
        out = list(zip(*Parallel(n_jobs=n_jobs)(delayed(single_hpd_region)(idx) for idx in it)))
    credible_regions, indicators = out[0], np.array(out[1])
    return indicators


def posterior_and_prior_kdeplot(
    samples_prior: np.ndarray,
    samples_posterior: np.ndarray,
    true_theta: np.ndarray,
    plot_marginals: bool = False,
    assets_dir: str = 'assets',
    **kwargs
) -> None:
    grid = sns.JointGrid(x=samples_posterior[:, 0], y=samples_posterior[:, 1])
    main_ax = sns.kdeplot(
        x=samples_prior[:, 0], y=samples_prior[:, 1], ax=grid.ax_joint, 
        color='black', label='Prior', linewidths=0.5, linestyles='--', zorder=1, **kwargs
    )
    if plot_marginals:
        sns.kdeplot(x=samples_prior[:, 0], ax=grid.ax_marg_x, color='grey', zorder=1, **kwargs)
        sns.kdeplot(y=samples_prior[:, 1], ax=grid.ax_marg_y, color='grey', zorder=1, **kwargs)

    # POSTERIOR
    sns.kdeplot(
        x=samples_posterior[:, 0], y=samples_posterior[:, 1], ax=grid.ax_joint, 
        cmap=sns.color_palette("Blues_d", as_cmap=True), fill=True, label='Posterior', zorder=10, **kwargs
    )
    sns.kdeplot(
        x=samples_posterior[:, 0], y=samples_posterior[:, 1], ax=grid.ax_joint, 
        color='black', label='Posterior', linewidths=0.5, zorder=10, **kwargs
    )
    if plot_marginals:
        sns.kdeplot(x=samples_posterior[:, 0], ax=grid.ax_marg_x, color='tab:blue', zorder=10, **kwargs)
        sns.kdeplot(y=samples_posterior[:, 1], ax=grid.ax_marg_y, color='tab:blue', zorder=10, **kwargs)
    # TRUE THETA
    main_ax.scatter(x=true_theta[0, 0].item(), y=true_theta[0, 1].item(), marker='*', s=77, color='red', zorder=100)

    main_ax.set_xlabel(r"$T_{eff}$", fontsize=15)
    main_ax.set_ylabel(r"$log g$", fontsize=15, rotation=0)
    handles = [
        mpatches.Patch(facecolor=plt.cm.Greys(100), label="Prior"),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label="Posterior")
    ]
    main_ax.invert_xaxis()
    main_ax.invert_yaxis()
    plt.legend(handles=handles)
    plt.savefig(f'{assets_dir}/posterior_from_sampling_{true_theta[0, 0].item():.2f}_{true_theta[0, 1].item():.2f}.png')
    plt.close()
