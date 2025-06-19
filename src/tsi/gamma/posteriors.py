import sbi.inference
import sbi.inference.posteriors
import sbi.inference.posteriors.base_posterior
import sbi.inference.posteriors.score_posterior
import torch.utils
import tiled_events as te
import torch
from typing import List, Union
import nde_models 
import numpy as np
from tqdm import tqdm
from lf2i.calibration.critical_values import train_qr_algorithm
from lf2i.test_statistics import Waldo
from lf2i.test_statistics.posterior import Posterior
import pickle as pkl
import os
import sbi
 
def approximate_posterior_mean(
    model: nde_models.JointCRNF,
    dense_features: torch.Tensor,
    param_grid: torch.Tensor,
    train_param_mins: List[float],
    train_param_maxes: List[float]
):
    # Posterior Mean
    grid_batch = {
        'params': param_grid.to(model.device),
        'features': dense_features[None, :].to(model.device)
    }
    te.scale_batch_params_inplace(grid_batch, train_param_mins, train_param_maxes)

    model.eval()
    with torch.no_grad():
        log_probs, _ = nde_models.log_probs(model, grid_batch['features'], grid_batch['params'])
        return (
            (param_grid.to(model.device) * torch.exp(log_probs[:, None])).sum(dim=0)/
            torch.exp(log_probs).sum()
        ).cpu()

def slice_param_set(param_set, component_index, value):
    component_values = param_set[:, component_index].unique()
    nearest_value_index = torch.argmin(torch.abs(component_values - value))
    nearest_value = component_values[nearest_value_index]
    mask = param_set[:, component_index] == nearest_value
    return param_set[mask], mask
    
def get_hpd_set(
    model: Union[nde_models.JointCRNF,nde_models.SplitCRNF],
    unscaled_param_grid: torch.Tensor,
    confidence_level: float,
    dense_features: torch.Tensor,
    train_param_mins: List[float],
    train_param_maxes: List[float],
    hpd_set_type: str,
    is_npse: bool = None
):
    assert hpd_set_type in ["energy", "angle", "joint"]
    
    grid_batch = {
        'params': unscaled_param_grid.to(model.device),
        'features': dense_features[None, :].to(model.device)
    }
    # te.scale_batch_params_inplace(grid_batch, train_param_mins, train_param_maxes)

    with torch.no_grad():
        if is_npse is not None:
            posterior = nde_models.SbiPosteriorLF2IWrapper(
                model.build_posterior(prior=torch.distributions.Uniform(-99, 99)),
                3,
                grid_batch['features'].device,
                is_npse
            )
            log_probs = posterior.log_prob(grid_batch['params'], x=grid_batch['features'])
        else:
            te.scale_batch_params_inplace(grid_batch, train_param_mins, train_param_maxes)
            log_probs1, log_probs2 = nde_models.log_probs(model, grid_batch['features'], grid_batch['params'])
            if type(model) is nde_models.SplitCRNF and log_probs2 is None and hpd_set_type == "angle":
                raise Exception("Requested angle HPD from energy only model")
            if type(model) is nde_models.SplitCRNF and hpd_set_type == "joint":
                raise Exception("Requested joint HPD from split model")
            
            if type(model) is nde_models.SplitCRNF and hpd_set_type == "angle":
                log_probs = log_probs2.cpu()
            else:
                log_probs = log_probs1.cpu()

        prob_scale_factor = 1/torch.exp(log_probs).sum()
        rescaled_probs = torch.exp(log_probs) * prob_scale_factor
    
    # def area_over_cut(cut):
    #     mask = rescaled_probs >= cut
    #     return rescaled_probs[mask].sum()/rescaled_probs.sum() - confidence_level
    
    # prob_critical = opt.brentq(
    #     area_over_cut,
    #     a=0,
    #     b=5
    # )
    sorted_probs = torch.sort(rescaled_probs, descending=True).values
    cut_index = (torch.cumsum(sorted_probs, dim=0) < confidence_level).sum()
    prob_critical = sorted_probs[cut_index]
    # print(f"Critical Value: {prob_critical}")
    return unscaled_param_grid[rescaled_probs >= prob_critical], torch.log(prob_critical/prob_scale_factor)

def get_param_grid(
    eval_param_mins: torch.Tensor,
    eval_param_maxes: torch.Tensor,
    target_grid_size: int,
    fixed_energy: float = None,
    fixed_zenith: float = None,
    fixed_azimuth: float = None
):
    fixed_params = [fixed_energy, fixed_zenith, fixed_azimuth]
    num_free_params = len([p for p in fixed_params if p is None])
    assert num_free_params > 0
    free_param_length = torch.pow(torch.tensor(target_grid_size), 1/num_free_params).int().item()
    linspaces = [
        torch.linspace(
            eval_param_mins[i] if fixed_params[i] is None else fixed_params[i],
            eval_param_maxes[i] if fixed_params[i] is None else fixed_params[i],
            free_param_length if fixed_params[i] is None else 1
        ) for i in range(len(fixed_params))
    ]
    
    return torch.cartesian_prod(*linspaces)
    

def joint_hpd_coverage_and_size(
    confidence_level: float,
    model: Union[nde_models.JointCRNF, nde_models.SplitCRNF],
    ds: te.CRImageDataset,
    train_param_mins: torch.Tensor,
    train_param_maxes: torch.Tensor,
    eval_param_mins: torch.Tensor,
    eval_param_maxes: torch.Tensor,
    param_grid_num_points: int,
    limit_count: int = None,
    no_azimuth: bool = False,
    is_npse: bool = None,
    max_log10_energy: float = None
):
    if no_azimuth:
        hpd_joint_grid = get_param_grid(eval_param_mins, eval_param_maxes, param_grid_num_points, fixed_azimuth=0).to(model.device)[:, 0:2]
    else:
        hpd_joint_grid = get_param_grid(eval_param_mins, eval_param_maxes, param_grid_num_points).to(model.device)
    joint_hpd_sizes = list()
    joint_hpd_hits = list()
    param_values = list()
    event_ids = list()
    weights = list()
    
    if max_log10_energy is not None:
        print(f"Max Log10 Energy: {max_log10_energy}")
    
    for event_id, event in enumerate(tqdm(ds, desc="HPD Coverage and Size")):
        if limit_count is not None and event_id >= limit_count:
            break
        if event is None:
            continue
        if max_log10_energy is not None and event['params'][0] > max_log10_energy:
            continue
        
        event_ids.append(event_id)
        param_values.append(event['params'].cpu().numpy())
        weights.append(event['weights'])
        eval_batch = {
            'params': event['params'][None, :],
            'features': event['features'][None, :]
        }
        if is_npse is None:
            te.scale_batch_params_inplace(eval_batch, train_param_mins, train_param_maxes)
            true_log_prob1, true_log_prob2 = nde_models.log_probs(model, eval_batch['features'].to(model.device), eval_batch['params'].to(model.device))
        else:
            device = model._device
            posterior = nde_models.SbiPosteriorLF2IWrapper(
                model.build_posterior(prior=torch.distributions.Uniform(-99, 99)),
                3,
                device,
                is_npse
            )
            true_log_prob1 = posterior.log_prob(eval_batch['params'].to(device), x=eval_batch['features'].to(device))
        
        hpd_set, joint_log_prob_cut = get_hpd_set(
            model,
            hpd_joint_grid,
            confidence_level,
            event['features'],
            train_param_mins,
            train_param_maxes,
            "joint",
            is_npse=is_npse
        )
        
        
        joint_hpd_sizes.append(hpd_set.shape[0]/hpd_joint_grid.shape[0])
        joint_hpd_hits.append(true_log_prob1.item() >= joint_log_prob_cut)
        
    return joint_hpd_hits, joint_hpd_sizes, param_values, weights, event_ids

    

def hpd_coverage_and_size(
    confidence_level: float,
    model: Union[nde_models.JointCRNF, nde_models.SplitCRNF],
    ds: te.CRImageDataset,
    train_param_mins: torch.Tensor,
    train_param_maxes: torch.Tensor,
    eval_param_mins: torch.Tensor,
    eval_param_maxes: torch.Tensor,
    test_grid_length: int,
    limit_count: int
):
    raise NotImplementedError("deprecated")
    if type(model) is nde_models.JointCRNF:
        is_joint = True 
    elif type(model) is nde_models.SplitCRNF:
        is_joint = False
    else:
        raise NotImplementedError("unknown model class")
    
    energy_hpd_sizes = list()
    energy_hpd_hits = list()
    angle_hpd_sizes = list()
    angle_hpd_hits = list()
    joint_hpd_sizes = list()
    joint_hpd_hits = list()
    param_values = list()
    weights = list()
    
    model.eval()
    with torch.no_grad():
        if is_joint:
            hpd_joint_grid = get_param_grid(eval_param_mins, eval_param_maxes, test_grid_length).to(model.device)
        else:
            raise NotImplementedError
            
        for event in tqdm(ds, desc="HPD Coverage and Size"):
            limit_count -= 1
            if limit_count <= 0:
                break
            if event is None:
                continue

            if is_joint:
                posterior_mean = approximate_posterior_mean(
                    model, 
                    event['features'],
                    hpd_joint_grid,
                    train_param_mins,
                    train_param_maxes
                )
            else:
                posterior_mean = torch.zeros(3, dtype=torch.float32, device=model.device)
                
            # Posterior at Truth
            param_values.append(event['params'].cpu().numpy())
            weights.append(event['weights'])
            eval_batch = {
                'params': event['params'][None, :],
                'features': event['features'][None, :]
            }
            te.scale_batch_params_inplace(eval_batch, train_param_mins, train_param_maxes)
            true_log_prob1, true_log_prob2 = nde_models.log_probs(model, eval_batch['features'], eval_batch['params'])
                
            # Energy HPD
            hpd_energy_grid = torch.cartesian_prod(
                torch.linspace(eval_param_mins[0], eval_param_maxes[0], test_grid_length**2),
                posterior_mean[1, None],
                posterior_mean[2, None]
            ).to(model.device)
            
            hpd_set, energy_log_prob_cut = get_hpd_set(
                model,
                hpd_energy_grid,
                confidence_level,
                event['features'],
                train_param_mins,
                train_param_maxes,
                "energy"
            )
            
            energy_hpd_sizes.append(hpd_set.shape[0]/hpd_energy_grid.shape[0])
            energy_hpd_hits.append(true_log_prob1.item() >= energy_log_prob_cut)
            del hpd_energy_grid
            
            # Angle HPD
            hpd_angle_grid = torch.cartesian_prod(
                posterior_mean[0, None],
                *[torch.linspace(eval_min, eval_max, test_grid_length) for eval_min, eval_max in zip(
                    eval_param_mins[1:],
                    eval_param_maxes[1:]
            )]).to(model.device)
            
            hpd_set, angle_log_prob_cut = get_hpd_set(
                model,
                hpd_angle_grid,
                confidence_level,
                event['features'],
                train_param_mins,
                train_param_maxes,
                "angle"
            )
            
            angle_hpd_sizes.append(hpd_set.shape[0]/hpd_angle_grid.shape[0])
            angle_hpd_hits.append((true_log_prob1.item() if is_joint else true_log_prob2.item()) >= angle_log_prob_cut)
            del hpd_angle_grid
                        
            if is_joint:
                hpd_set, joint_log_prob_cut = get_hpd_set(
                    model,
                    hpd_joint_grid,
                    confidence_level,
                    event['features'],
                    train_param_mins,
                    train_param_maxes,
                    "joint"
                )
                
                joint_hpd_sizes.append(hpd_set.shape[0]/hpd_joint_grid.shape[0])
                joint_hpd_hits.append(true_log_prob1.item() >= joint_log_prob_cut)
            else:
                joint_hpd_sizes.append(energy_hpd_sizes[-1] * angle_hpd_sizes[-1])
                joint_hpd_hits.append(energy_hpd_hits[-1] and angle_hpd_hits[-1])
                
                
    return (
        np.array(param_values),
        np.array(energy_hpd_sizes),
        np.array(energy_hpd_hits),
        np.array(angle_hpd_sizes),
        np.array(angle_hpd_hits),
        np.array(joint_hpd_sizes),
        np.array(joint_hpd_hits),
        np.array(weights)
    )
    
def _batch_cov(points: torch.Tensor):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)
    
def fast_waldo_evaluate(
    model: Union[nde_models.JointCRNF, nde_models.SplitCRNF, nde_models.WeightedNPSE],
    scaled_param_batch: torch.Tensor,
    scaled_feature_batch: torch.Tensor,
    posterior_samples_per_obs: int,
    is_npse: bool = None
):
    assert len(scaled_param_batch.shape) == 2
    assert len(scaled_feature_batch.shape) == 4
    model.eval()
    with torch.no_grad():
        if type(model) is nde_models.JointCRNF:
            context = model.context_model(scaled_feature_batch)
            samples, _ = model.nf.sample(
                posterior_samples_per_obs * context.shape[0], 
                context[:, None].repeat(1, posterior_samples_per_obs, 1).view(-1, context.shape[-1])
            )
            samples = samples.view(context.shape[0], posterior_samples_per_obs, -1)
            posterior_diffs = samples.mean(dim=1) - scaled_param_batch.to(model.device) # shape N, D
            posterior_variances = _batch_cov(samples) # shape N, D, D
            step1 = torch.matmul(posterior_diffs[:, None], torch.linalg.inv(posterior_variances)).squeeze(1) # shape N, D
            waldos = (step1 * posterior_diffs).sum(dim=1)
            
            return posterior_diffs, posterior_variances, waldos
            
            
        else:
            raise NotImplementedError
                
    
def sbi_waldo_evaluate(
    posterior: sbi.inference.posteriors.score_posterior.ScorePosterior,
    unscaled_param: torch.Tensor,
    scaled_features: torch.Tensor,
    posterior_samples: int,
    device: torch.device,
    is_npse: bool,
    n_jobs: int = 1,
):
    waldo = Waldo(
        estimator=nde_models.SbiPosteriorLF2IWrapper(posterior, 3, scaled_features.device, is_npse),
        poi_dim=3,
        estimation_method="posterior",
        num_posterior_samples=posterior_samples,
        n_jobs=n_jobs
    )

    samples = posterior.sample(
        sample_shape=torch.Size([posterior_samples]),
        x=scaled_features,
        ts=torch.linspace(1, 1e-5, 2, device=device)
    )
    posterior_diff = samples.mean(dim=0) - unscaled_param
    inv_posterior_variance = torch.inverse(torch.cov(torch.transpose(samples, 1, 0)))
    return (torch.matmul(posterior_diff, inv_posterior_variance) * posterior_diff).sum()
    
    
def lf2i_qr(
    confidence_level: float,
    precomupted_ts_path: str,
    model: Union[nde_models.JointCRNF, nde_models.SplitCRNF, nde_models.WeightedNPSE],
    cal_loader: torch.utils.data.DataLoader,
    train_param_mins: torch.Tensor,
    train_param_maxes: torch.Tensor,
    save_dir: str,
    num_posterior_samples: int,
    cal_loader_repeats: int,
    limit_batches: int = None,
    retrain_qr: bool = False,
    additional_loader: torch.utils.data.DataLoader = None,
    no_azimuth: bool = False,
    use_posterior: bool = False,
    max_sbi_energy: float = 999,
):
    # cal_features = list()
    cal_ts = list()
    cal_params = list()
    
    # waldo = Waldo(
    #     estimator= nde_models.LF2IModelWrapper(model),
    #     poi_dim=3,
    #     estimation_method="posterior",
    #     num_posterior_samples=num_posterior_samples,
    #     n_jobs=num_jobs
    # )
        
    if not os.path.exists(f"{save_dir}qr.pkl") or retrain_qr:
        
        if type(model) is nde_models.WeightedNPSE or type(model) is nde_models.WeightedFMPE:
            do_sbi = True
            print(f"Max Energy: {max_sbi_energy}")
            posterior = model.build_posterior(prior=torch.distributions.Uniform(-99, 99))
            cal_features = list()
        else:
            do_sbi = False
            model.eval()
        
        if os.path.exists(precomupted_ts_path):
            with open(precomupted_ts_path, 'rb') as file:
                cal_params, cal_ts = pkl.load(file)
            if use_posterior:
                confidence_level = 1 - confidence_level
        else:

            with torch.no_grad():
                for _ in range(cal_loader_repeats):
                    for batch_id, batch in enumerate(tqdm(cal_loader, desc="Extracting Cal Params/Features")):
                        if batch is None:
                            continue
                        if do_sbi:
                            mask = batch['params'][:, 0] <= max_sbi_energy
                            if mask.float().sum() == 0:
                                continue
                            cal_params.append(batch['params'][mask])
                            cal_features.append(batch['features'][mask])
                        else:
                            te.scale_batch_params_inplace(batch, train_param_mins, train_param_maxes)
                            cal_params.append(batch['params'])
                            # cal_features.append(batch['features'].cpu())
                            if use_posterior:
                                cal_ts.append(nde_models.log_probs(model, batch['features'].to(model.device), batch['params'].to(model.device))[0].cpu())
                            else:
                                cal_ts.append(fast_waldo_evaluate(model, batch['params'].to(model.device), batch['features'].to(model.device), num_posterior_samples)[2].cpu())
                        if limit_batches is not None and batch_id >= limit_batches - 1:
                            break
                    
                if additional_loader is not None:
                    for batch_id, batch in enumerate(tqdm(additional_loader, desc="Extracting Cal Params/Features (additional loader)")):
                        if batch is None:
                            continue
                        te.scale_batch_params_inplace(batch, train_param_mins, train_param_maxes)
                        cal_params.append(batch['params'].cpu())
                        # cal_features.append(batch['features'].cpu())
                        cal_ts.append(fast_waldo_evaluate(model, batch['params'].to(model.device), batch['features'].to(model.device), num_posterior_samples)[2].cpu())
                        if limit_batches is not None and batch_id >= limit_batches - 1:
                            break
                    
                
            cal_params = np.concatenate(cal_params, axis=0)
            if do_sbi:
                cal_features = torch.stack(cal_features, dim=0).to(model._device)
                estimator = nde_models.SbiPosteriorLF2IWrapper(
                    posterior, 
                    3, 
                    cal_features.device,
                    type(model) is nde_models.WeightedNPSE  
                )
                if use_posterior:
                    print("Using Posterior")
                    posterior_ts = Posterior(
                        poi_dim=3,
                        estimator=estimator,
                        norm_posterior_samples=None,
                        n_jobs=1
                    )
                    cal_ts = posterior_ts.evaluate(torch.tensor(cal_params).to(model._device), cal_features, mode="critical_values")
                    confidence_level = 1 - confidence_level
                else:
                    print("Using Waldo")
                    waldo = Waldo(
                        estimator=estimator,
                        poi_dim=3,
                        estimation_method="posterior",
                        num_posterior_samples=num_posterior_samples,
                        n_jobs=1
                    )
                    cal_ts = waldo.evaluate(cal_params, cal_features, mode="critical_values")
                del cal_features
            else:
                cal_ts = np.concatenate(cal_ts, axis=0)
            with open(precomupted_ts_path, 'wb') as file:
                pkl.dump((cal_params, cal_ts), file)
        # waldo_values = waldo.evaluate(
        #     parameters=cal_params[:limit_obs],
        #     samples=cal_features[:limit_obs],
        #     mode='critical_values'
        # )
        print(f"QR data size: {cal_params.shape[0]}")
        print(f"Confidence Level: {confidence_level}")

        qr = train_qr_algorithm(
            test_statistics=cal_ts,
            parameters=cal_params,
            algorithm='cat-gb',
            alpha=confidence_level,  # acceptance region on the left
            param_dim=2 if no_azimuth else 3,
            algorithm_kwargs={
                'cv': {
                    'n_estimators': [100, 300, 500, 700, 1000, 1500] if no_azimuth else [500, 700, 1000, 1500, 2000, 2500],
                    'max_depth': [7, 10, 12, 15],
                    # 'n_estimators': [100],
                    # 'max_depth': [10]
                },
                'n_iter': 24
                # 'n_estimators': 2500,
                # 'max_depth': 15
            },
            n_jobs=16
        )
        print(qr.best_params_)
        
        with open(f"{save_dir}qr.pkl", "wb") as file:
            pkl.dump(qr, file)
    else:
        with open(f"{save_dir}qr.pkl", "rb") as file:
            qr = pkl.load(file)
     
    return qr

def lf2i_confidence_set(
    qr,
    model: Union[nde_models.JointCRNF, nde_models.SplitCRNF],
    num_posterior_samples: int,
    param_grid: torch.Tensor,
    train_param_mins: torch.Tensor,
    train_param_maxes: torch.Tensor,
    scaled_features: torch.Tensor,
    is_npse: bool = None,
    use_posterior: bool = False
):
    
    # ts_over_grid = waldo.evaluate(
    #     scaled_param_grid,
    #     features[None].cpu(),
    #     mode='confidence_sets'
    # ).flatten()
    if is_npse is None:
        if use_posterior:
            raise NotImplementedError("use_posterior not implemented for non-SBI models")
        scaled_param_grid = (param_grid - train_param_mins)/(train_param_maxes-train_param_mins)
        critical_values_over_grid = qr.predict(scaled_param_grid.cpu().numpy()).flatten()
        ts_over_grid = fast_waldo_evaluate(
            model,
            scaled_param_grid,
            scaled_features[None],
            num_posterior_samples
        )[2].flatten().cpu().numpy()
    else:
        scaled_param_grid = param_grid
        critical_values_over_grid = qr.predict(scaled_param_grid.cpu().numpy()).flatten()
        posterior = nde_models.SbiPosteriorLF2IWrapper(
            model.build_posterior(prior=torch.distributions.Uniform(-99, 99)),
            3,
            scaled_param_grid.device,
            is_npse
        )
        if use_posterior:
            posterior_ts = Posterior(
                poi_dim=3,
                estimator=posterior,
                norm_posterior_samples=None,
                n_jobs=1
            )
            ts_over_grid = posterior_ts.evaluate(
                scaled_param_grid.to(model._device), 
                scaled_features[None], 
                mode="confidence_sets"
            ).flatten()
        else:
            waldo = Waldo(
                estimator=posterior,
                poi_dim=3,
                estimation_method="posterior",
                num_posterior_samples=num_posterior_samples,
                n_jobs=1
            )
            ts_over_grid = waldo.evaluate(
                scaled_param_grid,
                scaled_features[None],
                mode='confidence_sets'
            ).flatten()
    mask = ts_over_grid < critical_values_over_grid if not use_posterior else ts_over_grid > critical_values_over_grid
    return param_grid[mask], ts_over_grid, critical_values_over_grid
    
    
def lf2i_coverage_and_size(
    qr,
    model: Union[nde_models.JointCRNF, nde_models.SplitCRNF],
    ds: te.CRImageDataset,
    train_param_mins: torch.Tensor,
    train_param_maxes: torch.Tensor,
    eval_param_mins: torch.Tensor,
    eval_param_maxes: torch.Tensor,
    param_grid_num_points: int,
    num_posterior_samples: int,
    limit_count: int,
    no_azimuth: bool = False,
    log_ts: bool = False,
    is_npse: bool = None,
    use_posterior: bool = False,
    max_log10_energy: float = None
):
    energy_waldo_sizes = list()
    energy_waldo_hits = list()
    angle_waldo_sizes = list()
    angle_waldo_hits = list()
    joint_waldo_sizes = list()
    joint_waldo_hits = list()
    param_values = list()
    weights = list()
    
    print("Currently not calculating sizes")
    
    if max_log10_energy is not None:
        print(f"Max Log10 Energy: {max_log10_energy}")
    
    if no_azimuth:
        param_grid = get_param_grid(
            eval_param_mins,
            eval_param_maxes,
            param_grid_num_points,
            fixed_azimuth=0
        )[:, 0:2]
    else:
        param_grid = get_param_grid(
            eval_param_mins,
            eval_param_maxes,
            param_grid_num_points
        )
    scaled_param_grid = (param_grid - train_param_mins)/(train_param_maxes-train_param_mins)
    critical_values_over_grid = qr.predict(scaled_param_grid.numpy())
    if log_ts:
        critical_values_over_grid = np.exp(critical_values_over_grid)
    
    for event_id, event in enumerate(tqdm(ds, desc="Waldo Metrics")):
        if limit_count is not None and event_id >= limit_count:
            break
        if event is None:
            continue
        if max_log10_energy is not None and event['params'][0] > max_log10_energy:
            continue
        
        
        eval_batch = {
            'params': event['params'][None, :],
            'features': event['features'][None, :]
        }
        if is_npse is None:
            te.scale_batch_params_inplace(eval_batch, train_param_mins, train_param_maxes)
            
            # ts_over_grid = waldo.evaluate(
            #     param_grid,
            #     eval_batch['features'].cpu(),
            #     mode='confidence_sets'
            # )
            _, _, ts_over_grid = fast_waldo_evaluate(
                model,
                scaled_param_grid,
                eval_batch['features'].to(model.device),
                num_posterior_samples
            )

            # true_ts = waldo.evaluate(
            #     eval_batch['params'].cpu(),
            #     eval_batch['features'].cpu(),
            #     mode="critical_values"
            # )[0]
            _, _, true_ts = fast_waldo_evaluate(
                model,
                eval_batch['params'],
                eval_batch['features'].to(model.device),
                num_posterior_samples
            )
        else: 
            if use_posterior:
                posterior_ts = Posterior(
                    poi_dim=3,
                    estimator=nde_models.SbiPosteriorLF2IWrapper(
                        model.build_posterior(prior=torch.distributions.Uniform(-99, 99)),
                        3,
                        model._device,
                        is_npse
                    ),
                    norm_posterior_samples=None,
                    n_jobs=1
                )
                true_ts = posterior_ts.evaluate(eval_batch['params'].to(model._device), eval_batch['features'].to(model._device), mode="critical_values")[0]
            else:
                waldo = Waldo(
                    estimator=nde_models.SbiPosteriorLF2IWrapper(
                        model.build_posterior(prior=torch.distributions.Uniform(-99, 99)),
                        3,
                        model._device,
                        is_npse
                    ),
                    poi_dim=3,
                    estimation_method="posterior",
                    num_posterior_samples=num_posterior_samples,
                    n_jobs=1
                )
                true_ts = waldo.evaluate(eval_batch['params'].cpu(), eval_batch['features'].to(model._device), mode="critical_values")[0]
        
        param_values.append(event['params'].cpu().numpy())
        assert event['params'].shape[0] == 3
        assert len(event['params'].shape) == 1
        weights.append(event['weights'])
        critical_value = qr.predict(eval_batch['params'].cpu().numpy())[0]
        if log_ts:
            critical_value = np.exp(critical_value)
        joint_waldo_hits.append(true_ts.item() < critical_value if not use_posterior else true_ts.item() > critical_value)
        # joint_waldo_sizes.append((ts_over_grid.flatten().cpu().numpy() < critical_values_over_grid.flatten()).sum()/param_grid.shape[0])
        
    return (
        np.array(param_values),
        # np.array(energy_hpd_sizes),
        # np.array(energy_hpd_hits),
        # np.array(angle_hpd_sizes),
        # np.array(angle_hpd_hits),
        np.array(joint_waldo_sizes),
        np.array(joint_waldo_hits),
        np.array(weights),
    )


def get_params_and_posteriors(
    model: Union[nde_models.JointCRNF, nde_models.SplitCRNF], 
    loader: torch.utils.data.DataLoader
):
    
    model.eval()
    with torch.no_grad():
        pass
    
def waldo_sets():
    pass
    