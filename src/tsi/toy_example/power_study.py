import click
import pickle
import time
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from sbi.inference import FMPE
from sbi.utils import BoxUniform
import sbibm
from lf2i.inference import LF2I
from lf2i.test_statistics.posterior import Posterior


@click.group()
def main():
    pass


@main.command()
def run():

    POI_DIM = 2  # parameter of interest
    POI_BOUNDS = {r'$\theta_1$': (-10, 10), r'$\theta_2$': (-10, 10)}
    POSTERIOR_KWARGS = {
        # 'norm_posterior': None
    }
    B = 50_000  # num simulations to estimate posterior and test statistics
    B_PRIME = 30_000  # num simulations to estimate critical values
    NUM_OBS = 10_000  # num simulations to do diagnostics
    EVAL_GRID_SIZE = 5_000  # num evaluation points over parameter space to construct confidence sets
    CONFIDENCE_LEVEL = 0.954
    REFERENCE = BoxUniform(
        low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0]-1, POI_BOUNDS[r'$\theta_2$'][0]-1)),
        high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1]+1, POI_BOUNDS[r'$\theta_2$'][1]+1))
    )
    EVAL_GRID_DISTR = BoxUniform(
        low=torch.tensor((POI_BOUNDS[r'$\theta_1$'][0], POI_BOUNDS[r'$\theta_2$'][0])),
        high=torch.tensor((POI_BOUNDS[r'$\theta_1$'][1], POI_BOUNDS[r'$\theta_2$'][1]))
    )

    PRIOR_LOC = [0, 0]
    PRIOR_SIGMA = np.sqrt(2.0), 3*np.sqrt(2.0), 6*np.sqrt(2.0)

    TARGET_LOC = ([0, 0], [2.5, 0], [5, 0], [7.5, 0])
    TARGET_SIGMA = PRIOR_SIGMA[0]

    task = sbibm.get_task('gaussian_mixture')
    simulator = task.get_simulator()


    for prior_sigma in PRIOR_SIGMA:

        print(f'\nConfiguration: prior sigma {prior_sigma:.2f}', flush=True)
        start_time = time.time()

        PRIOR = MultivariateNormal(
            loc=torch.Tensor(PRIOR_LOC), covariance_matrix=(prior_sigma**2)*torch.eye(n=POI_DIM)
        )

        # NDE
        b_params = PRIOR.sample(sample_shape=(B, ))
        b_samples = simulator(b_params)
        fmpe = FMPE(
            prior=PRIOR,
            device='cpu'
        )
        _ = fmpe.append_simulations(b_params, b_samples).train()
        fmpe_posterior = fmpe.build_posterior()


        # VSI
        b_prime_params = REFERENCE.sample(sample_shape=(B_PRIME, ))
        b_prime_samples = simulator(b_prime_params)
        lf2i = LF2I(test_statistic=Posterior(poi_dim=2, estimator=fmpe_posterior, **POSTERIOR_KWARGS))
        _ = lf2i.inference(
            x=torch.Tensor([[0, 0]]),  # placeholder
            evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
            confidence_level=CONFIDENCE_LEVEL,
            calibration_method='critical-values',
            calibration_model='cat-gb',
            calibration_model_kwargs={
                'cv': {'iterations': [100, 300, 500, 700, 1000], 'depth': [1, 3, 5, 7, 9]},
                'n_iter': 25
            },
            T_prime=(b_prime_params, b_prime_samples),
        )

        # POWER STUDY
        all_confset_sizes = []
        all_target_params = []
        for target_loc in TARGET_LOC:
            TARGET = MultivariateNormal(
                loc=torch.Tensor(target_loc), covariance_matrix=(TARGET_SIGMA**2)*torch.eye(n=POI_DIM)
            )
            
            target_params = TARGET.sample(sample_shape=(NUM_OBS, ))
            target_samples = simulator(target_params)

            confidence_sets = lf2i.inference(
                x=target_samples,  # placeholder
                evaluation_grid=EVAL_GRID_DISTR.sample(sample_shape=(EVAL_GRID_SIZE, )),
                confidence_level=CONFIDENCE_LEVEL,
                calibration_method='critical-values'
            )

            confset_sizes = np.array([100*cs.shape[0]/EVAL_GRID_SIZE for cs in confidence_sets])
            assert confset_sizes.shape[0] == NUM_OBS, f'{confset_sizes.shape[0]}'
            all_confset_sizes.append(confset_sizes)
            all_target_params.append(target_params)
        assert len(all_confset_sizes) == len(TARGET_LOC), f'{len(all_confset_sizes)}, {len(TARGET_LOC)}'
        
        with open(f'./results/sbibm_example/power_study_priorsigma{prior_sigma:.2f}', 'wb') as f:
            pickle.dump({
                'lf2i': lf2i,
                'target_params': all_target_params,
                'confset_sizes': all_confset_sizes
            }, f)
        
        elapsed_time = time.time() - start_time 
        print(f'Experiment completed. Elapsed time: {(elapsed_time/60):.2f} mins', flush=True)


if __name__ == '__main__':
    main()
