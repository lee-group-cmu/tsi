import numpy as np
import torch
import torch.utils
import torch.utils.data
from universal import PrimaryParticleId
import torch
import torchvision.transforms.functional as ttf
from dataclasses import dataclass
from typing import List, Callable
from tqdm import tqdm
import glob
import pickle as pkl
import random
import pandas as pd


@dataclass
class TiledEvent:
    primary_type: PrimaryParticleId
    log10_energy_gev: float
    zenith: float 
    azimuth: float 
    time_tail_trims: List[float]
    global_features: dict
    channel_features: List[dict] 
    grid_length: int 
                
@dataclass
class SparseStackedGridFeatures:
    grid_x_list: List[torch.Tensor]
    grid_y_list: List[torch.Tensor]
    grid_values_list: List[torch.Tensor]
    grid_length: int
    is_time_column: List[bool]
    
    def __post_init__(self):
        assert len(self.grid_x_list) == len(self.grid_y_list)
        assert len(self.grid_y_list) == len(self.grid_values_list)
        assert len(self.grid_values_list) == len(self.is_time_column)
        for grid_x, grid_y, grid_values in zip(self.grid_x_list, self.grid_y_list, self.grid_values_list):
            if grid_x is None:
                assert grid_y is None 
                assert grid_values is None
                continue 
            
            assert grid_x.shape == grid_y.shape
            assert grid_y.shape == grid_values.shape
            assert (grid_x < self.grid_length).all()
            assert (grid_x >= 0).all()
            assert (grid_y < self.grid_length).all()
            assert (grid_y >= 0).all()
            

class SplitParameterWeights:
    
    def __init__(
        self,
        unfiltered_manifest: pd.DataFrame,
        split_df: pd.DataFrame,
        differential_flux: Callable[[float], float],
        num_bins: int
    ) -> None:
        
        self.log10_energy_quantiles = np.quantile(split_df["log10_energy"], np.linspace(0, 1, num_bins+1))
        self.zenith_quantiles = np.quantile(split_df["zenith"], np.linspace(0, 1, num_bins+1))
        self.weight_matrix = np.zeros((num_bins, num_bins))
        for i, eqlu in enumerate(tqdm(zip(self.log10_energy_quantiles[:-1], self.log10_energy_quantiles[1:]), desc="Param Weight Bins", total=num_bins)):
            eql, equ = eqlu
            for j, zqlu in enumerate(zip(self.zenith_quantiles[:-1], self.zenith_quantiles[1:])):
                zql, zqu = zqlu
                actual_counts = (
                    (split_df["log10_energy"] >= eql) & 
                    (split_df["log10_energy"] < equ) &
                    (split_df["zenith"] > zql) & 
                    (split_df["zenith"] <= zqu)
                ).sum()
                
                eq_mid = (eql + equ)/2
                theoretical_counts = (
                    (unfiltered_manifest["zenith"] > zql) & 
                    (unfiltered_manifest["zenith"] <= zqu)
                ).sum() * differential_flux(eq_mid)
                
                assert actual_counts > 0
                self.weight_matrix[i, j] = theoretical_counts/actual_counts
    
        self.weight_matrix = self.weight_matrix/self.weight_matrix.max()
        self.num_bins = num_bins
                
    def get_weight(self, log10_energies, zeniths):
        non_zero_mask = (
            (log10_energies >= self.log10_energy_quantiles.min()) &
            (log10_energies <= self.log10_energy_quantiles.max()) &
            (zeniths >= self.zenith_quantiles.min()) & 
            (zeniths <= self.zenith_quantiles.max())
        )
            
        energin_bins = np.minimum(np.digitize(log10_energies, self.log10_energy_quantiles), self.num_bins) - 1
        zenith_bins = np.minimum(np.digitize(zeniths, self.zenith_quantiles), self.num_bins) - 1
        raw_weights = self.weight_matrix[energin_bins, zenith_bins]
        if type(raw_weights) is np.ndarray:
            raw_weights[~non_zero_mask] = 0
            return raw_weights
        elif non_zero_mask:
            return raw_weights
        else:
            return 0
                    
                    


def get_sparse_stacked_grid_features(event: TiledEvent, device: torch.device):
    grid_x = list()
    grid_y = list()
    grid_values = list()
    is_time_column = list()
    for gkey in event.global_features:
        is_time_column.append(True)
        
        raw_indices = torch.from_numpy(event.global_features[gkey]['ti'].copy()).int().to(device)
        raw_averages = torch.from_numpy(event.global_features[gkey]['avg'].copy()).float().to(device)
        grid_x.append(raw_indices % event.grid_length)
        grid_y.append(raw_indices // event.grid_length)
        grid_values.append(raw_averages)
        
        
    for cid in range(len(event.channel_features)):
        for gkey in event.channel_features[cid]:
            is_time_column.append(False)
            if event.channel_features[cid][gkey] is None: # no counts for this particular channel 
                grid_x.append(None)
                grid_y.append(None)
                grid_values.append(None)
                continue
            raw_indices = torch.from_numpy(event.channel_features[cid][gkey]['ti'].copy()).int().to(device)
            raw_counts = torch.log(torch.from_numpy(event.channel_features[cid][gkey]['tc'].copy()).float().to(device) + 1)
            grid_x.append(raw_indices % event.grid_length)
            grid_y.append(raw_indices // event.grid_length)
            grid_values.append(raw_counts)
            
    return SparseStackedGridFeatures(grid_x, grid_y, grid_values, event.grid_length, is_time_column)

class DenseFeaturesAllZero(Exception):
    pass

def sparse_to_dense_grid_features(sparse_features: SparseStackedGridFeatures, detector_layout: torch.Tensor, downsample_factor: int, shower_shift_x: int, shower_shift_y: int):
    v_margin = max((sparse_features.grid_length - detector_layout.shape[0]) // 2, 0)
    h_margin = max((sparse_features.grid_length - detector_layout.shape[1]) // 2, 0)
    
    # final_grid_shape = (torch.tensor(detector_layout.shape)/downsample_factor).int()
    final_grid_shape = torch.tensor(detector_layout.shape).int()
    downsampled = torch.zeros(len(sparse_features.grid_x_list), *(final_grid_shape/downsample_factor).int(), device=sparse_features.grid_values_list[0].device)
    

    for i in range(len(sparse_features.grid_x_list)):
        base = torch.zeros(*final_grid_shape, device=sparse_features.grid_values_list[0].device)
        if sparse_features.grid_values_list[i] is None:
            continue
        shifted_x = sparse_features.grid_x_list[i] + shower_shift_x
        shifted_y = sparse_features.grid_y_list[i] + shower_shift_y
        bin_mask = ~(
            (shifted_x < h_margin) |
            (shifted_x >= sparse_features.grid_length - h_margin) | 
            (shifted_y < v_margin) | 
            (shifted_y >= sparse_features.grid_length - v_margin)  
        )
        
        base[(shifted_x[bin_mask] - h_margin).int(), (shifted_y[bin_mask] - v_margin).int()] = sparse_features.grid_values_list[i][bin_mask]
        if sparse_features.is_time_column[i]:
            base = torch.nn.functional.max_pool2d(base[None], downsample_factor)
        else:
            base = torch.nn.functional.avg_pool2d(base[None], downsample_factor)
        downsampled[i, :, :] = base
    
    if downsampled.sum() == 0:
        raise DenseFeaturesAllZero
    
    return downsampled

@dataclass
class TrainHighEnergySampler:
    min_log10_energy: float 
    max_log10_energy: float 
    bin_samples: np.ndarray

class CRImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        events: List[TiledEvent], 
        downsample_factor: int,
        detector_layout: torch.Tensor,
        max_shift_radius: float,
        min_nonzero_features: int,
        device: torch.device,
        sample_ratio: float,
        restrict_to_pm90: bool,
        defer_sparse_computation: bool = False,
        random_seed: int = 1234,
        high_energy_sampler: TrainHighEnergySampler = None,
        energy_plateau_bin_index: int = None,
        no_azimuth: bool = False
    ):
        if high_energy_sampler is not None:
            bin_edges = np.linspace(
                high_energy_sampler.min_log10_energy, 
                high_energy_sampler.max_log10_energy,
                len(high_energy_sampler.bin_samples) + 1
            )
            
            if energy_plateau_bin_index is None:
                bin_samples = high_energy_sampler.bin_samples
            else:
                energies = np.array([event.log10_energy_gev for event in tqdm(events, desc="Energies for Plateau")])
                bin_samples, _ = np.histogram(energies, bin_edges)
                bin_samples[0:energy_plateau_bin_index] = bin_samples[energy_plateau_bin_index]
            print(bin_samples)
            running_bin_counts = np.zeros_like(bin_samples)
            self.events = list()
            for event in tqdm(events, desc="High Energy Sampler"):
                bin_index = np.digitize(event.log10_energy_gev, bin_edges) - 1
                if running_bin_counts[bin_index] < bin_samples[bin_index]:
                    self.events.append(event)
                    running_bin_counts[bin_index] += 1
        elif sample_ratio == 1:
            self.events = events 
        else:
            random.seed(random_seed)
            self.events = random.sample(events, int(len(events) * sample_ratio))
        self.detector_layout = detector_layout
        self.max_shift_radius = max_shift_radius
        self.downsample_factor = downsample_factor
        self.min_nonzero_features = min_nonzero_features
        self.device = device
        self.defer = defer_sparse_computation
        self.restrict_to_pm90 = restrict_to_pm90
        self.debug_only_params = False
        self.param_weighter = None
        self.no_azimuth = no_azimuth
        self.skip_scaling = False
        self.using_subsampling = False
        
        self.sparse_features: List[SparseStackedGridFeatures] = list()
        if not self.defer:
            for event in tqdm(self.events, desc="Sparse Grid Features"):
                self.sparse_features.append(
                    get_sparse_stacked_grid_features(event, self.device)
                )
            
            self.feature_maxes = None
    
    def set_feature_scale(self, feature_maxes):
        self.feature_maxes = torch.tensor(feature_maxes, device=self.device)
        
    def set_param_weighter(self, param_weighter: SplitParameterWeights):
        self.param_weighter = param_weighter
        if param_weighter == -1:
            print("Parameter Weighter set to -1")
        else:
            print("Parameter Weighter set")
            
    def output_manifest(self, filename):
        if self.using_subsampling:
            if filename is not None:
                with open(filename, 'wb') as file:
                    pkl.dump(self.internal_train_manifest, file)
            return self.internal_train_manifest
        
        rows = list()
        for event in self.events:
            gkey = list(event.global_features.keys())[0]
            
            rows.append((
                event.log10_energy_gev, 
                event.zenith, 
                event.azimuth, 
                len(event.global_features[gkey]['ti']), 
                self.param_weighter.get_weight(event.log10_energy_gev, event.zenith) if self.param_weighter != -1 else 1
            ))
        out_df = pd.DataFrame(
            data=rows,
            columns=["log10_energy", "zenith", "azimuth", "num_features", "weight"]
        )
        if filename is not None:
            with open(filename, 'wb') as file:
                pkl.dump(out_df, file)
        return out_df
        
    def __len__(self):
        if self.using_subsampling:
            return self.internal_train_manifest['subsample_mask'].sum()
        return len(self.events)
    
    def __getitem__(self, idx):
        if self.debug_only_params:
            event = self.events[idx]
            return {
                "params": torch.tensor([event.log10_energy_gev, event.zenith, event.azimuth], dtype=torch.float32, device=self.device),
            }
            
        if self.using_subsampling and not self.internal_train_manifest["subsample_mask"].to_numpy()[idx]:
            return None
        
        assert self.feature_maxes is not None, "feature scale not set"
        assert self.param_weighter is not None, "param weighter not set"
        event = self.events[idx]
        
        if self.max_shift_radius > 0:
            shower_shift = ((torch.rand(2) - 0.5) * 2 * self.max_shift_radius).int()
            while (shower_shift**2).sum() > self.max_shift_radius**2:
                shower_shift = ((torch.rand(2) - 0.5) * 2 * self.max_shift_radius).int()
        else:
            shower_shift = torch.zeros(2)
            
        if self.defer:
            sparse_features = get_sparse_stacked_grid_features(event, self.device)
        else:
            sparse_features = self.sparse_features[idx]
            
        scaled_grid_values_list = list()
        for cid, is_time_average in enumerate(sparse_features.is_time_column):
            if sparse_features.grid_x_list[cid] is None:
                scaled_grid_values_list.append(None)
                continue
            if not self.skip_scaling:
                if is_time_average:
                    local_min = sparse_features.grid_values_list[cid].min().item()
                    scaled_grid_values_list.append((sparse_features.grid_values_list[cid] - local_min) / self.feature_maxes[cid] * 0.5 + 0.5)
                else:
                    scaled_grid_values_list.append(sparse_features.grid_values_list[cid]/self.feature_maxes[cid])
            else:
                scaled_grid_values_list.append(sparse_features.grid_values_list[cid])
        
        try:
            features = sparse_to_dense_grid_features(
                SparseStackedGridFeatures(
                    sparse_features.grid_x_list,
                    sparse_features.grid_y_list,
                    scaled_grid_values_list,
                    sparse_features.grid_length,
                    sparse_features.is_time_column
                ), 
                self.detector_layout, 
                self.downsample_factor,
                shower_shift[0].item(), 
                shower_shift[1].item()
            )
        except DenseFeaturesAllZero:
            return None
        
        if features[0].nonzero().shape[0] <  self.min_nonzero_features:
            return None
        
        if self.restrict_to_pm90 and (event.azimuth < -np.pi/2 or event.azimuth > np.pi/2) and not self.no_azimuth:
            azimuth = ((event.azimuth + np.pi/2) % np.pi) - np.pi/2
            features = torch.rot90(features, 2, dims=[1, 2])
            assert azimuth >= -np.pi/2 and azimuth <= np.pi/2
        else:
            azimuth = event.azimuth
            
        if self.param_weighter == -1 or self.using_subsampling:
            weight = 1 
        else:
            weight = self.param_weighter.get_weight(event.log10_energy_gev, event.zenith)
            
        if self.no_azimuth:
            return {
                "params": torch.tensor([event.log10_energy_gev, event.zenith], dtype=torch.float32, device=self.device),
                "features": features,
                "shift": shower_shift,
                "weights": weight
            }
        else: 
            return {
                "params": torch.tensor([event.log10_energy_gev, event.zenith, azimuth], dtype=torch.float32, device=self.device),
                "features": features,
                "shift": shower_shift,
                "weights": weight
            }
            
    def use_subsampling(self):
        self.internal_train_manifest = self.output_manifest(None)
        self.internal_train_manifest["normalized_weight"] = self.internal_train_manifest["weight"]/self.internal_train_manifest["weight"].max()
        self.internal_train_manifest['subsample_mask'] = np.random.random(len(self.internal_train_manifest)) < self.internal_train_manifest['normalized_weight']
        self.using_subsampling = True
        print("Using Subsampling")
        
        
def get_feature_bounds(ds: CRImageDataset):
    if ds.defer:
        raise Exception("Can't get feature bounds on deferred datasets")
    num_channels = len(ds.sparse_features[0].grid_values_list)
    feature_maxes = [-torch.inf] * num_channels
    for sparse_features in tqdm(ds.sparse_features):
        for cid, is_time_average in enumerate(sparse_features.is_time_column):
            if sparse_features.grid_values_list[cid] is None:
                continue
            if is_time_average:
                feature_maxes[cid] = max(feature_maxes[cid], sparse_features.grid_values_list[cid].max() - sparse_features.grid_values_list[cid].min())
            else:
                feature_maxes[cid] = max(feature_maxes[cid], sparse_features.grid_values_list[cid].max())
    
    return feature_maxes

def scale_batch_params_inplace(batch, param_mins, param_maxes) -> None:
    device = batch['params'].device
    batch['params'] = (batch['params'] - param_mins.to(device))/(param_maxes.to(device) - param_mins.to(device))
                

def get_data(
    data_dir: str, 
    split_name: str, 
    train_ratio_for_val: float, 
    device: torch.device,
    detector_layout: torch.Tensor,
    max_shift_radius: int,
    downsample_factor: int,
    min_features_threshold: int
):
    raise NotImplementedError
    main_data = list()
    val_data = list()

    file_list = glob.glob(data_dir + f"*_{split_name}_*.pkl")
    train_n = int(len(file_list) * train_ratio_for_val)
    for fid, filename in enumerate(tqdm(file_list)):
        with open(filename, 'rb') as file:
            tiled_event_dict = pkl.load(file)
        for event in tiled_event_dict:
            event_features = event['features']
            if event_features is None:
                continue
            if sum([len(val['ti']) for _, val in  event_features['global_features'].items()]) == 0:
                continue
            tiled_event = TiledEvent(
                primary_type=PrimaryParticleId.GAMMA,
                log10_energy_gev=np.log10(event['energy']),
                zenith=event['zenith'],
                azimuth=event['azimuth'],
                time_tail_trims=event_features['time_tail_trims'],
                global_features=event_features['global_features'],
                channel_features=event_features['channel_features'],
                grid_length=event_features["grid_length"],
                device=device
            )
            if fid < train_n:
                main_data.append(tiled_event)
            else:
                val_data.append(tiled_event)

    main_ds = CRImageDataset(main_data, downsample_factor, detector_layout, max_shift_radius, min_features_threshold, device)
    if len(val_data) > 0:
        val_ds = CRImageDataset(val_data, downsample_factor, detector_layout, max_shift_radius, min_features_threshold, device)
    else:        
        val_ds = None 
    return main_ds, val_ds
        