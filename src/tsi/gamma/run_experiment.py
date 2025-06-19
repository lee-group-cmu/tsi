
import torch
import glob
import torch.utils
from tqdm import tqdm
import pickle as pkl
import tiled_events as te
from universal import PrimaryParticleId
from configs import ExperimentConfig, config_dict
import numpy as np
import argparse
import nde_models
import os
import posteriors as pos
import sys
import random
import cr_sources as crs
import diagnostics as diag
import plotting

def load_dataset(
    config: ExperimentConfig,
    pkl_filename: str,
    loaded_datasets: dict,
    device: torch.device,
    sample_ratio: float,
    min_zenith: float,
    train_high_energy_sampler: te.TrainHighEnergySampler = None,
    energy_plateau_bin_index: int = None
):
    if pkl_filename in loaded_datasets:
        return loaded_datasets[pkl_filename], loaded_datasets
    
    full_path = f"{config.source_data_config.splits_dir}{pkl_filename}"
    print(f"Opening {full_path}")
    with open(full_path, "rb") as file:
        event_list = pkl.load(file)
        if min_zenith > 0:
            print(f"Minimum Zenith: {min_zenith:0.5f}")
            event_list = [e for e in tqdm(event_list, desc="Zenith Chop") if e.zenith >= min_zenith]
        
        ds = te.CRImageDataset(
            events=event_list,
            downsample_factor=config.downsample_factor,
            detector_layout=config.detector_layout,
            max_shift_radius=config.max_shift_radius,
            min_nonzero_features=config.min_features_threshold,
            device=device,
            sample_ratio=sample_ratio,
            restrict_to_pm90=config.restrict_azimuth_to_pm90deg,
            high_energy_sampler=train_high_energy_sampler,
            energy_plateau_bin_index=energy_plateau_bin_index,
            no_azimuth=config.no_azimuth
        )
    loaded_datasets[pkl_filename] = ds 
    return ds, loaded_datasets
    
def run_experiment(
    config_name: str,
    results_dir: str, 
    out_dir: str, 
    posterior_only: bool,
    debug_mode: bool,
    force_cpu: bool
):
    config = config_dict[config_name]
    
    torch.manual_seed(config.seed)
    random.seed(config.seed + 12)
    np.random.seed(config.seed + 99)
    
    if config.gpu_index >= 0 and not force_cpu:
        DEVICE = torch.device(f"cuda:{config.gpu_index}")
    else:
        DEVICE = torch.device("cpu")
    loaded_datasets = dict()
    
    train_trajectory = crs.get_source_trajectory(config.train_astropy_source_name, config.train_observer_latitude)
    train_min_zenith = np.deg2rad(90 - np.array(train_trajectory.alt).max())
    
    train_high_energy_sampler = None 
    energy_plateau_bin_index = None
    if config.train_higher_energy_prior_config is not None:
        print("Using higher energy prior")
        with open(config.source_data_config.filtered_manifest_filename, 'rb') as file:
            filtered_manifest = pkl.load(file)
        
        split_df = filtered_manifest[filtered_manifest["split"] == 1]
        bin_samples, _ = np.histogram(split_df["log10_energy"], config.train_higher_energy_prior_config["num_bins"])
        
        cumulative_sum = 0
        energy_plateau_bin_index = len(bin_samples) - 1
        while bin_samples[energy_plateau_bin_index] * (energy_plateau_bin_index + 1) < config.train_higher_energy_prior_config["target_size"] - cumulative_sum:
            cumulative_sum += bin_samples[energy_plateau_bin_index]
            energy_plateau_bin_index -= 1
        
        bin_samples[0:(energy_plateau_bin_index + 1)] = int((config.train_higher_energy_prior_config["target_size"] - cumulative_sum)/(energy_plateau_bin_index + 1))
        assert config.train_higher_energy_prior_config["target_size"] - bin_samples.sum() < config.train_higher_energy_prior_config["num_bins"]
        train_high_energy_sampler = te.TrainHighEnergySampler(
            config.eval_param_mins[0],
            config.eval_param_maxes[0],
            bin_samples
        )
        
    with open(config.source_data_config.manifest_filename, "rb") as file:
        unfiltered_manifest = pkl.load(file)
    with open(config.source_data_config.filtered_manifest_filename, "rb") as file:
        filtered_manifest = pkl.load(file)
    train_feature_maxes_file = out_dir + "train_feature_maxes.pkl"
    if os.path.exists(train_feature_maxes_file):
        with open(train_feature_maxes_file, 'rb') as file:
            train_feature_maxes = pkl.load(file)
    else:
        train_ds, loaded_datasets = load_dataset(
            config,
            config.train_data_pkl,
            loaded_datasets,
            torch.device("cpu") if config.train_ds_on_cpu else DEVICE,
            sample_ratio=config.train_sample_ratio,
            min_zenith=train_min_zenith,
            train_high_energy_sampler=train_high_energy_sampler
        )
        train_feature_maxes = te.get_feature_bounds(train_ds)
        with open(train_feature_maxes_file, 'wb') as file:
            pkl.dump(train_feature_maxes, file)

    torch.manual_seed(config.seed)
    random.seed(config.seed + 12)
    np.random.seed(config.seed + 99)
    
    model = config.model_constructor(
        DEVICE,
        len(config.time_averages),
        torch.tensor(config.detector_layout.shape)/config.downsample_factor
    )

    if config.use_existing_trained_model and os.path.exists(f"{results_dir}{config.model_save_path}.done"):
        if config.use_sbi:
            with open("vsi_cosmic_rays/single_event.pkl", "rb") as file:
                single_event = pkl.load(file)
            model._neural_net = model._build_neural_net(
                single_event["params"][None].repeat(2, 1).to(DEVICE), 
                single_event["features"][None].repeat(2, 1, 1, 1).to(DEVICE)
            )
            model._neural_net.load_state_dict(torch.load(f"{results_dir}{config.model_save_path}"))
            model._neural_net.to(DEVICE)
        else:
            model.load_state_dict(torch.load(f"{results_dir}{config.model_save_path}"))
        print(f"Loading existing model at {results_dir}{config.model_save_path}")
    else:
        print(f"Training new model")
        
        train_weighter_file = out_dir + "train_weighter.pkl"
        if os.path.exists(train_weighter_file):
            with open(train_weighter_file, 'rb') as file:
                train_weighter = pkl.load(file)
        else:
            train_weighter = te.SplitParameterWeights(
                unfiltered_manifest, 
                filtered_manifest[filtered_manifest["split"] == 1],
                config.train_differential_flux,
                40
            ) if config.train_higher_energy_prior_config is None else -1
        
        if config.use_sbi and not os.path.exists(f"{out_dir}stacked_data.pkl"):
            train_ds, loaded_datasets = load_dataset(
                config,
                config.train_data_pkl,
                loaded_datasets,
                DEVICE if not config.train_ds_on_cpu else torch.device("cpu"),
                sample_ratio=config.train_sample_ratio,
                min_zenith=train_min_zenith,
                train_high_energy_sampler=train_high_energy_sampler
            )
            
            
            val_ds, loaded_datasets = load_dataset(
                config,
                config.val_data_pkl,
                loaded_datasets,
                DEVICE,
                sample_ratio=config.val_sample_ratio,
                min_zenith=train_min_zenith,
                train_high_energy_sampler=train_high_energy_sampler,
                energy_plateau_bin_index=energy_plateau_bin_index
            )
            
                
            train_ds.set_feature_scale(train_feature_maxes)
            train_ds.set_param_weighter(train_weighter)
            val_ds.set_feature_scale(train_feature_maxes)
            val_ds.set_param_weighter(train_weighter)
            if config.use_subsampling:
                train_ds.use_subsampling()
                val_ds.use_subsampling()
            train_ds.output_manifest(out_dir + "train_manifest.pkl")
            val_ds.output_manifest(out_dir + "val_manifest.pkl")
            print(f"Train Obs: {len(train_ds)} | Val Obs: {len(val_ds)}")
            
        
        if config.use_sbi:
            if not os.path.exists(f"{out_dir}stacked_data.pkl"):
                train_params = list()
                train_features = list()
                for ds in [train_ds, val_ds]:
                    for i in tqdm(range(len(ds))):
                        event = ds[i]
                        if event is None:
                            continue
                        train_params.append(event['params'].cpu())
                        train_features.append(event['features'].cpu())
                del train_ds
                del val_ds
                del loaded_datasets[config.train_data_pkl]
                del loaded_datasets[config.val_data_pkl]
                torch.cuda.empty_cache()
                train_params = torch.stack(train_params)
                train_features = torch.stack(train_features)
                with open(f"{out_dir}stacked_data.pkl", "wb") as file:
                    pkl.dump((train_params, train_features), file)
            else:
                print("Going to use pre-saved stacked data")
                with open(f"{out_dir}stacked_data.pkl", "rb") as file:
                    print(f"Loading stacked data from {out_dir}stacked_data.pkl")
                    train_params, train_features = pkl.load(file)
            # i = 0
            # while i < train_params.shape[0]:
            #     model.append_simulations(train_params[i:(i + 1000)].to(DEVICE), train_features[i:(i + 1000)].to(DEVICE))
            #     i += 1000
            model.append_simulations(train_params, train_features)
            # model._data_round_index = [0]
            # model._theta_roundwise = [train_params]
            # model._x_roundwise = [train_features]
            # model._prior_masks = [torch.ones((train_params.shape[0], 1), dtype=torch.bool, device=DEVICE)]
            del train_params
            del train_features
            
            
            
            if config.no_train_weights:
                print("Training without weights") 
                model.train(show_train_summary=True)
            else:
                print("Training with weights")
                model.train(
                    calibration_kernel=lambda theta: torch.tensor(train_weighter.get_weight(theta[:, 0].cpu().numpy(), theta[:, 1].cpu().numpy()), device=DEVICE, dtype=torch.float32),
                    show_train_summary=True
                )
            with open(f"{out_dir}sbi_train_summary.pkl", "wb") as file:
                pkl.dump(model._summary, file)
            torch.save(model._neural_net.state_dict(), f"{results_dir}{config.model_save_path}")
            with open(f"{results_dir}{config.model_save_path}.done", "w") as file:
                pass
        else:
            train_ds, loaded_datasets = load_dataset(
                config,
                config.train_data_pkl,
                loaded_datasets,
                DEVICE if not config.train_ds_on_cpu else torch.device("cpu"),
                sample_ratio=config.train_sample_ratio,
                min_zenith=train_min_zenith,
                train_high_energy_sampler=train_high_energy_sampler
            )
            
            
            val_ds, loaded_datasets = load_dataset(
                config,
                config.val_data_pkl,
                loaded_datasets,
                DEVICE,
                sample_ratio=config.val_sample_ratio,
                min_zenith=train_min_zenith,
                train_high_energy_sampler=train_high_energy_sampler,
                energy_plateau_bin_index=energy_plateau_bin_index
            )
            
            train_ds.set_feature_scale(train_feature_maxes)
            train_ds.set_param_weighter(train_weighter)
            val_ds.set_feature_scale(train_feature_maxes)
            val_ds.set_param_weighter(train_weighter)
            
            if config.use_subsampling:
                train_ds.use_subsampling()
                val_ds.use_subsampling()
            
            train_ds.output_manifest(out_dir + "train_manifest.pkl")
            val_ds.output_manifest(out_dir + "val_manifest.pkl")
            print(f"Train Obs: {len(train_ds)} | Val Obs: {len(val_ds)}")
            
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=config.train_batch_size,
                collate_fn=nde_models.img_collate,
                shuffle=True
            )

            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=config.train_batch_size,
                collate_fn=nde_models.img_collate,
            )
            model.train()

            loss_hist = list()
            val_epoch_loss_hist = list()


            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 
            best_val_loss = 999999999
            best_val_epoch = -1
            for epoch in tqdm(range(config.max_epochs)):
                for batch_id, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                    if config.train_limit_batches_per_epoch is not None and batch_id >= config.train_limit_batches_per_epoch:
                        break
                    if batch is None or batch['features'].shape[0] <= 1:
                        continue
                
                    te.scale_batch_params_inplace(batch, config.train_param_mins.to(DEVICE), config.train_param_maxes.to(DEVICE))
                    
                    optimizer.zero_grad()
                    loss1, loss2 = nde_models.forward_kld(model, batch['features'], batch['params'], batch['weights'].to(DEVICE))
                    
                    if ~(torch.isnan(loss1) | torch.isinf(loss1)):
                        if loss2 is not None:
                            if ~(torch.isnan(loss2) | torch.isinf(loss2)):
                                loss2 = loss2 * config.loss2_ratio 
                                loss1.backward(retain_graph=True)
                                loss2.backward()
                            else:
                                raise Exception("loss2 invalid")
                        else:
                            loss1.backward()
                        optimizer.step()
                    else:
                        raise Exception("loss1 invalid")
                    loss_hist.append([loss1.item(), 0 if loss2 is None else loss2.item()])
                    

                    
                
                val_loss1, val_loss2 = nde_models.model_loss_acc_loader(model, val_loader, config.train_param_mins, config.train_param_maxes, None)
                val_loss2 = val_loss2 * config.loss2_ratio
                val_epoch_loss_hist.append([val_loss1, val_loss2])
                if val_loss1 + val_loss2 < best_val_loss:
                    torch.save(model.state_dict(), f"{results_dir}{config.model_save_path}")
                    best_val_loss = val_loss1 + val_loss2
                    best_val_epoch = epoch 
                    train_loss1, train_loss2 = nde_models.model_loss_acc_loader(model, train_loader, config.train_param_mins, config.train_param_maxes, config.train_error_estimate_batches)
                    print(f"New best val loss {val_loss1 + val_loss2:0.5f} | Train Loss  {train_loss1 + train_loss2:0.5f}")
                    
                if epoch - best_val_epoch >= config.val_epoch_cooldown:
                    print(f"No improvement in {config.val_epoch_cooldown} epochs")
                    break  
            
            with open(f"{results_dir}{config.model_save_path}.done", "w") as file:
                pass
            torch.save(torch.tensor(loss_hist), out_dir + "train_loss.pt")
            torch.save(torch.tensor(val_epoch_loss_hist), out_dir + "val_epoch_loss.pt")
            del train_loader
            del train_ds
        if not config.use_sbi:
            del loaded_datasets[config.train_data_pkl]
            del loaded_datasets[config.val_data_pkl]
    
    if posterior_only:
        sys.exit(0)
    
    
    if not os.path.exists(out_dir + "qr_data.pkl"):
        cal_ds, loaded_datasets = load_dataset(
            config,
            config.calibration_data_pkl,
            loaded_datasets,
            torch.device("cpu"),
            sample_ratio=config.cal_sample_ratio,
            min_zenith=train_min_zenith if config.cal_limit_zenith_to_train else 0
        )
        cal_ds.set_feature_scale(train_feature_maxes)
        cal_ds.set_param_weighter(-1)
        cal_ds.output_manifest(out_dir + "cal_manifest.pkl")
        cal_loader = torch.utils.data.DataLoader(
            cal_ds,
            batch_size=1, #config.train_batch_size,
            collate_fn=nde_models.img_collate
        )
        
        if config.additional_calibration_data_pkl is not None:
            cal_ds2, loaded_datasets = load_dataset(
                config,
                config.additional_calibration_data_pkl,
                loaded_datasets,
                DEVICE
            )
            cal_ds2.set_feature_scale(train_feature_maxes)
            cal_loader2 = torch.utils.data.DataLoader(
                cal_ds2,
                batch_size=config.train_batch_size,
                collate_fn=nde_models.img_collate
            )
        else:
            cal_loader2 = None
            cal_ds2 = None
    else:
        cal_ds = None
        cal_ds2 = None
        cal_loader = None 
        cal_loader2 = None
        loaded_datasets[config.calibration_data_pkl] = None
        
    qr = pos.lf2i_qr(
        confidence_level=config.confidence_levels[0],
        precomupted_ts_path=out_dir + "qr_data.pkl",
        model=model,
        cal_loader=cal_loader,
        train_param_mins=config.train_param_mins,
        train_param_maxes=config.train_param_maxes,
        save_dir=out_dir,
        num_posterior_samples=config.calibration_num_posterior_samples,
        limit_batches=config.calibration_batch_limit_count,
        additional_loader=cal_loader2,
        cal_loader_repeats=config.cal_loader_repeats,
        retrain_qr=False,
        no_azimuth=config.no_azimuth,
        use_posterior=config.use_posterior,
        max_sbi_energy=config.max_sbi_energy,
    )
    
    del cal_loader
    del cal_loader2
    del cal_ds
    del cal_ds2
    del loaded_datasets[config.calibration_data_pkl]
    
    is_npse = None
    if type(model) is nde_models.WeightedNPSE:
        is_npse = True
    if type(model) is nde_models.WeightedFMPE:
        is_npse = False
    print("Is NPSE: ", str(is_npse))
    
    test_ds, loaded_datasets = load_dataset(
        config,
        config.eval_data_pkl,
        loaded_datasets,
        DEVICE,
        sample_ratio=1.0,
        min_zenith=train_min_zenith if config.cal_limit_zenith_to_train else 0
    )
    test_ds.set_feature_scale(train_feature_maxes)
    test_ds.set_param_weighter(-1)
    test_ds.max_shift_radius = config.max_shift_radius
    test_ds.output_manifest(out_dir + "test_manifest.pkl")

    print("Estimating HPD Coverage")
    joint_hpd_hits, _, hpd_param_values, _ = diag.hpd_coverage_metrics(
        config,
        out_dir + "hpd_metrics.pkl",
        model,
        test_ds,
        config.coverage_calc_grid_num_points,
        config.confidence_levels[0],
        limit_count=100 if debug_mode else None
    )
    
    diag.estimate_coverage(
        config,
        out_dir + "hpd_coverage_estimator.pkl",
        config.coverage_calc_grid_num_points,
        joint_hpd_hits,
        hpd_param_values
    )
    
    print("Estimating LF2I Coverage")
    joint_lf2i_hits, _, lf2i_param_values, _ = diag.lf2i_coverage_metrics(
        config,
        out_dir + "lf2i_metrics.pkl",
        model,
        test_ds,
        config.coverage_calc_grid_num_points,
        qr,
        limit_count=100 if debug_mode else None
    )
    
    diag.estimate_coverage(
        config,
        out_dir + "lf2i_coverage_estimator.pkl",
        config.coverage_calc_grid_num_points,
        joint_lf2i_hits,
        lf2i_param_values
    )
    
    del joint_hpd_hits
    del joint_lf2i_hits
    del hpd_param_values
    del lf2i_param_values
    
    print("Generating Examples")
    
    
    with open(out_dir + "test_manifest.pkl", "rb") as file:
        test_manifest = pkl.load(file)
    test_manifest["azimuth_pm90"] = (test_manifest["azimuth"] + np.pi/2) % np.pi - np.pi/2
    
    plotting.generate_examples(
        config,
        config_name,
        out_dir,
        model,
        qr,
        DEVICE,
        filtered_manifest,
        unfiltered_manifest,
        [None, None, None],
        test_ds,
        test_manifest,
        1_000_000
    )
    

from dataclasses import dataclass

@dataclass
class FakeArgs:
    out_dir: str 
    config_name: str
    posterior_only: bool
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSI Cosmic Ray Experiments")
    
    # Add the arguments
    parser.add_argument('--config_name', type=str, required=True, help='Which config to use')
    parser.add_argument('--out_dir', type=str, required=True, help='Where to put results')
    parser.add_argument('--posterior_only', action="store_true", help='Only estimate posterior?')
    parser.add_argument('--debug', action="store_true", help='Debug Mode')
    parser.add_argument('--cpu', action="store_true", help='Force run on CPU')
    
    # Parse the arguments
    args = parser.parse_args()
    # args = FakeArgs(
    #     "results/",
    #     "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_more_train_data",
    #     True
    # )
    # print("Using fake args")
    
    experiment_out_dir = args.out_dir + f"{args.config_name}_{config_dict[args.config_name].seed}/"
    if not os.path.exists(experiment_out_dir):
        os.makedirs(experiment_out_dir)
    
    run_experiment(
        args.config_name,
        args.out_dir,
        experiment_out_dir,
        args.posterior_only,
        args.debug,
        args.cpu
    )