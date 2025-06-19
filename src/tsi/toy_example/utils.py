from typing import List, Optional, Tuple, Dict, Union
import contextlib
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Uniform
from sbi.utils import MultipleIndependent
from statsmodels.nonparametric.smoothers_lowess import lowess
from sbibm.tasks.gaussian_mixture.task import GaussianMixture


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as
    argument

    This wrapped context manager obtains the number of finished tasks from the tqdm
    print function and uses it to update the pbar, as suggested in
    https://stackoverflow.com/a/61689175. See #419, #421
    """

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def lowess_bootstrap_ci(
    x: np.ndarray, 
    y: np.ndarray, 
    n_boot: int = 1000, 
    lowess_frac: float = 2/3, 
    ci: Union[float, np.ndarray] = 95.0,
    n_jobs: int = -2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(ci, (float, int)):
        ci = np.array([ci])
    assert all(0 <= level <= 100 for level in ci)

    lowess_fit = lowess(y, x, frac=lowess_frac, return_sorted=True)
    x_smooth, y_smooth = lowess_fit[:, 0], lowess_fit[:, 1]

    def bootstrap_iteration():
        sample_idx = np.random.choice(len(x), size=len(x), replace=True)
        x_sampled, y_sampled = x[sample_idx], y[sample_idx]

        sorted_indices = np.argsort(x_sampled)
        x_sampled, y_sampled = x_sampled[sorted_indices], y_sampled[sorted_indices]

        y_boot = lowess(y_sampled, x_sampled, frac=lowess_frac, return_sorted=True)[:, 1]
        return y_boot.reshape(1, -1)

    with tqdm_joblib(tqdm(range(n_boot), desc="Bootstrapping", total=n_boot)):
        y_boot_samples = np.vstack(Parallel(n_jobs=n_jobs)(delayed(bootstrap_iteration)() for _ in range(n_boot)))

    lower_bound = np.percentile(y_boot_samples, (100 - ci) / 2, axis=0)
    upper_bound = np.percentile(y_boot_samples, 100 - (100 - ci) / 2, axis=0)

    return x_smooth, y_smooth, lower_bound, upper_bound


class GaussianMixtureWithLogProb(GaussianMixture):

    def log_prob(
        self,
        mu: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Eval one sample at a time, possibly for multiple p.
        Using torch because it allows to evaluate more easily multiple batches of n samples together, if necessary.
        """
        assert x.ndim <= 2, 'Only evaluate one sample x (of size n) at a time. No leading batch dimension'
        mu, x = torch.from_numpy(mu).reshape(-1, 1, self.dim_parameters).float(), torch.from_numpy(x).reshape(1, -1, self.dim_data).float()

        comp0 = MultivariateNormal(loc=mu, covariance_matrix=torch.eye(self.dim_data).float() * self.simulator_params["mixture_scales"][0].item()**2)
        comp1 = MultivariateNormal(loc=mu, covariance_matrix=torch.eye(self.dim_data).float() * self.simulator_params["mixture_scales"][1].item()**2)
        
        comp0_log_prob = torch.exp(comp0.log_prob(x)).double()
        comp1_log_prob = torch.exp(comp1.log_prob(x)).double()
        per_sample_log_prob = torch.log(
            self.simulator_params['mixture_weights'][0]*comp0_log_prob + self.simulator_params['mixture_weights'][1]*comp1_log_prob
        ).numpy()
        
        assert per_sample_log_prob.shape == (mu.shape[0], x.shape[1]), f'{per_sample_log_prob.shape}'
        return per_sample_log_prob.sum(axis=1)


def likelihood_ratio_ts(
    mu: np.ndarray,
    x: np.ndarray,
    task: GaussianMixtureWithLogProb,
    poi_bounds: Tuple[float],
    num_poi_grid_mle: Optional[int] = 1e5
) -> np.ndarray:
    """LR test statistic one sample at a time, possibly fro multiple p. Actual output  is -2*log(LR) since asymptotically chi2.
    """
    if mu.ndim == 1:
        mu = mu.reshape(1, task.dim_parameters)
    
    null = task.log_prob(mu, x).reshape(mu.shape[0], )
    
    poi_grid = MultipleIndependent(dists=[
        *[Uniform(torch.Tensor([poi_bounds[0]]), torch.Tensor([poi_bounds[1]])) for _ in range(task.dim_parameters)]
    ]).sample(sample_shape=(int(num_poi_grid_mle), )).numpy()
    alternative = task.log_prob(poi_grid, x).reshape(poi_grid.shape[0], ).max()
    
    assert alternative.size == 1, f'{alternative.size}'
    return (-2 * (null - alternative.item()))
