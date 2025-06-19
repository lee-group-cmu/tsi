from dataclasses import dataclass
from typing import List, Callable
import universal
import numpy as np
import pandas as pd
import universal
import matplotlib.pyplot as plt
import util
import glob
import re
import pickle as pkl
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

@dataclass
class RawCosmicRayEvent:
    primary_id: universal.PrimaryParticleId
    azimuth: float
    zenith: float
    energy: float
    secondary_df: pd.DataFrame # columns = [id, x, y, t, momentum]
    

    @classmethod
    def from_h5(cls, source_h5, event_num: int, primary_particle_id: universal.PrimaryParticleId) -> "RawCosmicRayEvent":
        """This transforms coordinates such that positive y is north
        Event data in h5 is 1-indexed. So if there are 100 events in the h5 file, 
            the keys go from "event_1" to "event_100"

        Args:
            source_h5 (_type_): _description_
            event_id (_type_): _description_
            primary_particle_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        raw_event = source_h5[f'event_{event_num}']
        return cls(
            primary_id = primary_particle_id,
            azimuth = source_h5['event_info']['azimuth'][event_num-1],
            zenith = source_h5['event_info']['zenith'][event_num-1],
            energy = source_h5['event_info']['event_energy'][event_num-1],
            secondary_df = pd.DataFrame({
                "x": -np.array(raw_event['y']),
                "y": np.array(raw_event['x']),
                "t": np.array(raw_event['t']),
                "secondary_ids": np.array(raw_event['particle_type']),
                "secondary_momenta": np.array(raw_event['mom'])
            })
        )
    
    x = property(fget = lambda s: s.secondary_df["x"].to_numpy())
    y = property(fget = lambda s: s.secondary_df["y"].to_numpy())
    t = property(fget = lambda s: s.secondary_df["t"].to_numpy())
    secondary_ids = property(fget = lambda s: s.secondary_df["secondary_ids"].to_numpy())
    secondary_momenta = property(fget = lambda s: s.secondary_df["secondary_momenta"].to_numpy())
        
def secondary_id_tabulations(event: RawCosmicRayEvent) -> np.ndarray:
    return np.unique(event.secondary_ids, return_counts=True)


def secondary_positions(event: RawCosmicRayEvent, particle_ids: List[int] = None):
    if particle_ids is not None:
        indices = np.nonzero(np.isin(event.secondary_ids, particle_ids))
        return np.column_stack([event.x[indices], event.y[indices]])
    else:
        return np.column_stack([event.x, event.y])

def secondary_mask(event: RawCosmicRayEvent, particle_ids: List[int]):
    return np.nonzero(np.isin(event.secondary_ids, particle_ids))

def get_arrow_of_time(event: RawCosmicRayEvent, mask=None):
    mask_x, mask_y, mask_t = event.x[mask], event.y[mask], event.t[mask]
      
    location = np.column_stack((mask_x, mask_y))
    coefs = LinearRegression().fit(location, mask_t).coef_
    time_arrow = coefs/np.linalg.norm(coefs, ord=2)
    
    arrow_projection = (location * time_arrow).sum(axis=-1)
    height_slope = LinearRegression().fit(arrow_projection.reshape(-1, 1), mask_t).coef_[0]
    return time_arrow, height_slope

def plot_footprint(
    event: RawCosmicRayEvent, 
    time_trim: float, 
    title_note: str, 
    sample_prop=None,
    manual_title=None, 
    my_ax=None, 
    figsize=(20,20), 
    fontsize=20, 
    marker_size=200,
    ax_lim=2500, 
    secondary_types: List[int]=None, 
    show_lines=True
):
    marker_list = ['v', 's', 'X', 'D']
    
    mask = (event.t < np.quantile(event.t, 1-time_trim/2)) & (event.t > np.quantile(event.t, time_trim/2))
    secondary_types = secondary_types or []
    assert len(secondary_types) <= len(marker_list), "too many secondary types"
      
    if sample_prop is not None:
        mask = mask & np.random.binomial(1, sample_prop, mask.shape).astype(bool)

    if my_ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        ax = my_ax
    ax.set_aspect('equal', adjustable='box')
    
    if mask.sum() == 0:
        print("No secondary particle match filters")
        return

    # ghost plot for cmap
    base_plot = ax.scatter(
        event.x[mask], 
        event.y[mask], 
        c = event.t[mask], 
        s = 0, 
        cmap="plasma",
    )
    for i, secondary in enumerate(secondary_types):
        type_mask = mask & np.isin(event.secondary_ids, secondary)
        event_plot = ax.scatter(
            event.x[type_mask], 
            event.y[type_mask], 
            c = event.t[type_mask], 
            s = marker_size, 
            marker=marker_list[i], 
            norm=base_plot.norm,
            cmap="plasma",
            label=f"Particle id {secondary}"
        )
    type_mask = mask & ~np.isin(event.secondary_ids, secondary_types)
    event_plot = ax.scatter(
        event.x[type_mask], 
        event.y[type_mask], 
        c = event.t[type_mask], 
        s = marker_size, 
        cmap="plasma",
        label="Others",
        norm=base_plot.norm
    )
        
    cbar = plt.colorbar(event_plot, fraction=0.046, pad=0.04, label="Time (ns)", ax=ax)
    cbar.set_label("Time (ns)", fontsize=fontsize)
    
    if show_lines and mask.sum() > 5:
        pca = PCA(n_components=2)
        pca.fit(np.column_stack([event.x[mask], event.y[mask]]))
        pca_com = pca.components_
        pca_var = pca.explained_variance_ratio_
        ax.plot([0, pca_com[0][0] * 4000 * pca_var[0]], [0, pca_com[0][1] * 4000 * pca_var[0]], c="red")
        ax.plot([0, pca_com[1][0] * 4000 * pca_var[0]], [0, pca_com[1][1] * 4000 * pca_var[0]], c="blue")
        
        arrow, _ = get_arrow_of_time(event, mask)
        ax.plot([0, arrow[0] * 4000], [0, arrow[1] * 4000], c="green")
        
    attack_angle = util.angle_line(event.azimuth, event.zenith) * 1000 
    ax.arrow(0, 0, attack_angle[0], attack_angle[1], width=10, head_width=100, color="black")
    
    ax.set_xlabel("x (meters)", fontsize=fontsize)
    ax.set_ylabel("y (meters)", fontsize=fontsize)
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    
    title = f"Position of secondary particles (Azimuth {round(event.azimuth, 3)}, Zenith {round(event.zenith, 3)}, Energy {round(event.energy, 3)})\n" \
                f"Black line shows direction of primary particle. Flat length is 100,000 m\n" \
                f"Red and Blue lines indicate 1st and 2nd PCA components, scaled according to variance explained{title_note}"
    ax.set_title(title if manual_title is None else manual_title, fontsize=fontsize)
    ax.legend()
    fig.tight_layout()
    
def export_tiled_event(
    event:RawCosmicRayEvent, 
    channels: List[List[int]], 
    tile_size: int, 
    time_tail_trims: List[float], 
    include_remainder_channel: bool=False
):
    max_range = max(np.max(np.abs(event.x)), np.max(np.abs(event.y))) # is possibly a float
    grid_length = int(2 * (max_range // tile_size + 1)) # side length in grid squares
    max_grid_range = grid_length // 2 * tile_size # in meters, origin to edge



    min_time = np.min(event.t)
    max_time = np.max(event.t)
    return_dict = {
        "channels": channels,
        "include_remainder_channel": include_remainder_channel,
        "tile_size": tile_size,
        "grid_length": grid_length,
        "num_secondary": len(event.secondary_ids),
        "mom_secondary": event.secondary_momenta.sum(),
        "time_start": min_time,
        "time_end": max_time,
        "time_tail_trims": time_tail_trims,
        "global_features": dict(), # **time_features (bin indices, average time)
        "channel_features": [dict() for _ in channels] + ([dict()] if include_remainder_channel else []) # *channels -> **time_features
    }
    
    # image
    """Bins start from the top left, and go in reading order

    Args:
        p_x (_type_): _description_
        p_y (_type_): _description_

    Returns:
        _type_: _description_
    """
    full_bin_list = np.column_stack((-event.x, event.y)).astype(np.float64)
    full_bin_list = np.floor((max_grid_range - full_bin_list)/tile_size).astype(int)
    full_bin_list = (grid_length * full_bin_list[:, 1] + full_bin_list[:, 0])
    
    lower_time_quantiles = np.quantile(event.t, [t/2 for t in time_tail_trims])
    upper_time_quantiles =  np.quantile(event.t, [1-t/2 for t in time_tail_trims])

    # Global time features
    t = 0
    for min_time, max_time in zip(lower_time_quantiles, upper_time_quantiles):
        time_mask = (event.t >= min_time) & (event.t <= max_time)
        tile_sum = np.bincount(full_bin_list[time_mask], weights=event.t[time_mask])
        tile_count = np.bincount(full_bin_list[time_mask])
        tile_indices = np.flatnonzero(tile_count).astype(np.uint32)
        return_dict['global_features'][f'avg_time_trim_{time_tail_trims[t]}'] = {
            "ti": tile_indices,
            "avg":  (tile_sum[tile_indices]/tile_count[tile_indices]).astype(np.float32)
        }
        
        total_channel_mask = np.zeros(len(event.x)).astype(bool)
        for cid, channel in enumerate(channels):
            channel_mask = np.isin(event.secondary_ids, channel) & time_mask
            if channel_mask.sum() == 0:
                return_dict["channel_features"][cid][f"counts_time_trim_{time_tail_trims[t]}"] = None
                continue
            total_channel_mask = np.logical_or(total_channel_mask, channel_mask)
            
            mask = channel_mask 
            all_tile_count = np.bincount(full_bin_list[mask])
            tile_indices = np.flatnonzero(all_tile_count).astype(np.uint32)
            return_dict['channel_features'][cid][f"counts_time_trim_{time_tail_trims[t]}"] = {
                "ti": tile_indices,
                "tc":  all_tile_count[tile_indices].astype(np.uint32)
            }
        
        if include_remainder_channel:
            mask = (~total_channel_mask) & time_mask
            all_tile_count = np.bincount(full_bin_list[mask])
            tile_indices = np.flatnonzero(all_tile_count).astype(np.uint32)
            return_dict['channel_features'][len(channels)][f"counts_time_trim_{time_tail_trims[t]}"] = {
                "ti": tile_indices,
                "tc":  all_tile_count[tile_indices].astype(np.uint32)
            }
        t += 1

    return return_dict


def generate_unfiltered_manifest(
    save_filename: str,
    folders: List[str],
    overwrite:bool = True
):
    running_manifest_df = None
    processed_files = list()
    if not overwrite:
        try:
            with open(save_filename, 'rb') as file:
                running_manifest_df = pkl.load(file)
                processed_files = running_manifest_df['file'].unique()
        except FileNotFoundError:
            pass
    
    for folder in folders:
        for filename in tqdm(glob.glob(folder + "/" +"DAT10*.pkl"), desc=folder):
            if filename in processed_files:
                continue
            
            manifest_list = list()
            with open(filename, 'rb') as file:
                event_dict = pkl.load(file)
            for event_id in event_dict['events'].keys():
                event = event_dict['events'][event_id]
                if event['features'] is None:
                    num_features = 0
                else:
                    first_trim = event['features']['time_tail_trims'][0]
                    num_features = event['features']['global_features'][f'avg_time_trim_{first_trim}']['ti'].size
                manifest_list.append((
                    filename, 
                    event_id, 
                    event["energy"], 
                    event["zenith"], 
                    event["azimuth"],
                    num_features))
            
            if running_manifest_df is None:
                running_manifest_df = pd.DataFrame(manifest_list, columns=["file", "event_id", "energy", "zenith", "azimuth", "num_features"])
            else:
                new_df = pd.DataFrame(manifest_list, columns=["file", "event_id", "energy", "zenith", "azimuth", "num_features"])
                running_manifest_df = pd.concat([running_manifest_df, new_df]).reset_index(drop=True)
                 
            running_manifest_df['file'] = pd.Categorical(running_manifest_df['file'])
                
            with open(save_filename, 'wb') as file:
                pkl.dump(running_manifest_df, file)
                
                
def generate_filtered_manifest(
    manifest_filename: str,
    filtered_manifest_filename: str,
    reference_ratio: float,
    train_carveout_ratio: float,
    train_log10flux_of_log10energy_gev: Callable[[float], float],
    test_log10flux_of_log10energy_gev: Callable[[float], float],
    astropy_names: List[str],
    observer_latitude: float,
    trajectory_azimuth_epsilon_deg_per_source: List[float],
    energy_bins: int,
    user_min_log10_energy_gev: float = None,
    user_max_log10_energy_gev: float = None,
):
    with open(manifest_filename, 'rb') as file:
        new_manifest = pkl.load(file)
    new_manifest["split"] = 0
    new_manifest["energy_weight"] = 0.0
    new_manifest["source"] = "NA"
    new_manifest["on_trajectory"] = False
    new_manifest["log10_energy"] = np.log10(new_manifest["energy"])
    
    # Calibration Set
    log_energy_bins = np.linspace(
        new_manifest['log10_energy'].min(), 
        new_manifest['log10_energy'].max(), 
        energy_bins
    )
    
    hist, _ = np.histogram(new_manifest['log10_energy'], log_energy_bins)
    bin_widths = log_energy_bins[1:] - log_energy_bins[:-1]
    implied_density = np.min(hist / bin_widths)

    for i in tqdm(range(len(log_energy_bins) - 1), desc="Reference Energy Bins"):
        bin_indexes = new_manifest[(new_manifest['log10_energy'] >= log_energy_bins[i]) & (new_manifest['log10_energy'] < log_energy_bins[i + 1])].index
        assert len(bin_indexes) > int(implied_density * bin_widths[i] * reference_ratio)
        selected = np.random.choice(bin_indexes, int(implied_density * bin_widths[i] * reference_ratio), replace=False)
        new_manifest.loc[selected, "split"] = 2
        
    # Train Test Carvouts
    for loop_i, source_name, split_id, source_flux, sample_ratio in zip(
        list(range(2)),
        astropy_names,
        [1, 3],
        [train_log10flux_of_log10energy_gev, test_log10flux_of_log10energy_gev],
        [train_carveout_ratio, 1.0]
    ):
        available_df = new_manifest[new_manifest['split'] == 0]
        
        print("Trajectory Trimming")
        # Trajectory
        trajectory = get_source_trajectory(source_name, observer_latitude)
        
        azimuth_modded = np.deg2rad(((np.array(trajectory.az) + 180) % 360) - 180)
        right_azimuth = np.max(azimuth_modded)
        left_azimuth = np.min(azimuth_modded)

        right_mask = azimuth_modded > 0
        right_azimuths = azimuth_modded[right_mask]
        right_zeniths = np.pi/2 - np.deg2rad(np.array(trajectory.alt))[right_mask]

        left_mask = azimuth_modded < 0
        left_azimuths = azimuth_modded[left_mask]
        left_zeniths = np.pi/2 - np.deg2rad(np.array(trajectory.alt))[left_mask]

        eps=np.deg2rad(trajectory_azimuth_epsilon_deg_per_source[loop_i])

        azimuth_mask = (
            (available_df['azimuth'] > left_azimuth - eps) & 
            (available_df['azimuth'] < right_azimuth + eps) & 
            (available_df['zenith'] > np.min(np.pi/2 - np.deg2rad(np.array(trajectory.alt))))
        )
        subset = available_df[azimuth_mask]
        subset_indexes = list()
    
        for mask, azimuths, zeniths in zip(
            [subset['azimuth'] > 0, subset['azimuth'] < 0], 
            [right_azimuths, left_azimuths], 
            [right_zeniths, left_zeniths]
        ):
            side_subset = subset[mask]

            zenith_sort = np.argsort(zeniths)
            target_azimuth = np.interp(side_subset['zenith'], zeniths[zenith_sort], azimuths[zenith_sort])
            filter_mask = np.abs(side_subset['azimuth'] - target_azimuth) < eps 
            subset_indexes.append(side_subset[filter_mask].index.to_numpy())

        subset_indexes = np.concatenate(subset_indexes)
        assert np.unique(subset_indexes).size == subset_indexes.size
        new_manifest.loc[subset_indexes, "on_trajectory"] = True
        available_df = new_manifest.loc[subset_indexes, :]
         
        print("Spectrum Trimming")
        
        # upper and lower bounds
        if user_min_log10_energy_gev is None:
            min_log10_energy_gev = available_df["log10_energy"].min()
        else:
            min_log10_energy_gev = user_min_log10_energy_gev
        
        if user_max_log10_energy_gev is None:
            max_log10_energy_gev = available_df["log10_energy"].max()
        else:
            max_log10_energy_gev = user_max_log10_energy_gev
        
        
        # pre-filter and bin
        available_df = available_df[
            (available_df['log10_energy'] >= min_log10_energy_gev) & 
            (available_df['log10_energy'] <= max_log10_energy_gev)
        ]
        log_energy_bins = np.quantile(available_df["log10_energy"], np.linspace(0, 1, energy_bins))
        hist, _ = np.histogram(available_df['log10_energy'], log_energy_bins)
        log_midpoints = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2
        scales = hist / 10**(source_flux(log_midpoints))
        implied_scale = np.min(scales)
        sample_list = list()
        for i in tqdm(range(len(log_energy_bins) - 1)):
            log_mid = log_midpoints[i]
            bin_indexes = available_df[
                (available_df['log10_energy'] >= log_energy_bins[i]) & 
                (available_df['log10_energy'] < log_energy_bins[i + 1])
            ].index
            energy_weight = implied_scale / scales[i]
            assert energy_weight > 0
            new_manifest.loc[bin_indexes, "energy_weight"] = energy_weight
            new_manifest.loc[bin_indexes, "source"] = source_name

        selectable_df = new_manifest.query(f"split == 0 and source == '{source_name}' and on_trajectory")
        permutation = np.random.permutation(selectable_df.index)
        new_manifest.loc[permutation[:int(len(selectable_df) * sample_ratio)], "split"] = split_id
        new_manifest.loc[new_manifest["split"] == 0, "source"] = "NA"
        new_manifest.loc[new_manifest["split"] == 0, "energy_weight"] = 0
        new_manifest.loc[new_manifest["split"] == 0, "on_trajectory"] = False
    
    with open(filtered_manifest_filename, 'wb') as file:
        pkl.dump(new_manifest, file)

def generate_energy_prior_manifest(
    manifest_filename: str,
    filtered_manifest_filename: str,
    min_log10_energy_gev: float,
    max_log10_energy_gev: float,
    train_log10flux_of_log10energy_gev: Callable[[float], float],
    test_log10flux_of_log10energy_gev: Callable[[float], float],
    astropy_names: List[str],
    train_ratio: float,
    test_ratio: float,
    energy_bins: int,
    zenith_cutoffs: bool,
    observer_latitude: float=None
):
    with open(manifest_filename, 'rb') as file:
        new_manifest = pkl.load(file)
    new_manifest["split"] = 0
    new_manifest["energy_weight"] = 0.0
    new_manifest["source"] = "NA"
    new_manifest["on_trajectory"] = True
    new_manifest["log10_energy"] = np.log10(new_manifest["energy"])
    
    # Train Test
    for split_id, ratio, source_flux_func, source_name in zip(
        [1, 3],
        [train_ratio, test_ratio],
        [train_log10flux_of_log10energy_gev, test_log10flux_of_log10energy_gev],
        astropy_names
    ):
        available_df = new_manifest[new_manifest["split"] == 0]
        sample_df = available_df[
            (available_df['log10_energy'] >= min_log10_energy_gev) & 
            (available_df['log10_energy'] <= max_log10_energy_gev)
        ].sample(frac=ratio)
        
        if zenith_cutoffs:
            print("Zenith Thresholding")
            trajectory = get_source_trajectory(source_name, observer_latitude)
            min_zenith = np.deg2rad(90 - np.max(np.array(trajectory.alt)))
            sample_df = sample_df[sample_df["zenith"] >= min_zenith]
        
        print("Spectrum Trimming")  
        # pre-filter and bin
        log_energy_bins = np.quantile(sample_df["log10_energy"], np.linspace(0, 1, energy_bins))
        hist, _ = np.histogram(sample_df['log10_energy'], log_energy_bins)
        log_midpoints = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2
        scales = hist / 10**(source_flux_func(log_midpoints))
        implied_scale = np.min(scales)
        for i in tqdm(range(len(log_energy_bins) - 1)):
            bin_indexes = sample_df[
                (sample_df['log10_energy'] >= log_energy_bins[i]) & 
                (sample_df['log10_energy'] < log_energy_bins[i + 1])
            ].index
            energy_weight = implied_scale / scales[i]
            assert energy_weight > 0
            new_manifest.loc[bin_indexes, "energy_weight"] = energy_weight
            new_manifest.loc[bin_indexes, "source"] = source_name
            new_manifest.loc[bin_indexes, "split"] = split_id
        
        
    # Calibration Set
    available_df = new_manifest[new_manifest["split"] == 0]
    available_df = available_df[
        (available_df['log10_energy'] >= min_log10_energy_gev) & 
        (available_df['log10_energy'] <= max_log10_energy_gev)
    ]
    new_manifest.loc[available_df.index, "split"] = 2
    
    with open(filtered_manifest_filename, 'wb') as file:
        pkl.dump(new_manifest, file)
        
def generate_all_prior_manifest(
    manifest_filename: str,
    filtered_manifest_filename: str,
    min_log10_energy_gev: float,
    max_log10_energy_gev: float,
    min_zenith: float,
    train_log10flux_of_log10energy_gev: Callable[[float], float],
    test_log10flux_of_log10energy_gev: Callable[[float], float],
    astropy_names: List[str],
    cal_sample_ratio: float,
    train_ratio: float,
    test_ratio: float,
    num_calibration_bins: int,
    num_train_test_bins: int,
    observer_latitude: float,
    minimum_weight: float,
    minimum_features: int,
    target_calibration_size: int
):
    with open(manifest_filename, 'rb') as file:
        new_manifest = pkl.load(file)
    new_manifest["split"] = 0
    new_manifest["energy_weight"] = 0.0
    new_manifest["zenith_weight"] = 0.0
    new_manifest["source"] = "NA"
    new_manifest["on_trajectory"] = True
    new_manifest["log10_energy"] = np.log10(new_manifest["energy"])
    
    # Prefilter
    new_manifest = new_manifest[
        (new_manifest["log10_energy"] > min_log10_energy_gev) & 
        (new_manifest["log10_energy"] < max_log10_energy_gev) &
        (new_manifest["zenith"] > min_zenith)
    ]
    
    # Calibration
    print("Calibration Set")
    available_df = new_manifest[
        (new_manifest["split"] == 0) & (new_manifest["num_features"] >= minimum_features)
    ]
    
    energy_hist_counts, energy_hist_bin_edges = np.histogram(available_df['log10_energy'], num_calibration_bins)
    min_energy_bin_index = np.argmin(energy_hist_counts)
    zenith_hist_counts, zenith_hist_bin_edges = np.histogram(available_df['zenith'], num_calibration_bins)
    min_zenith_bin_index = np.argmin(zenith_hist_counts)
    azimuth_hist_counts, azimuth_hist_bin_edges = np.histogram(available_df['azimuth'], num_calibration_bins)
    min_azimuth_bin_index = np.argmin(azimuth_hist_counts)
    
    # bin_counts = list()
    # for energy_bin_lower, energy_bin_upper in zip(energy_hist_bin_edges[:-1], energy_hist_bin_edges[1:]):
    #     for zenith_bin_lower, zenith_bin_upper in zip(zenith_hist_bin_edges[:-1], zenith_hist_bin_edges[1:]):
    #         for azimuth_bin_lower, azimuth_bin_upper in zip(azimuth_hist_bin_edges[:-1], azimuth_hist_bin_edges[1:]):
    #             bin_count = available_df[
    #                 (available_df["log10_energy"] >= energy_bin_lower) & (available_df["log10_energy"] < energy_bin_upper) &
    #                 (available_df["zenith"] >= zenith_bin_lower) & (available_df["zenith"] < zenith_bin_upper) &
    #                 (available_df["azimuth"] >= azimuth_bin_lower) & (available_df["azimuth"] < azimuth_bin_upper)
    #             ].shape[0]
    #             bin_counts.append(bin_count)
    # min_bin_count = sorted(bin_counts)[calibration_min_bin_count_index]         
    # assert min_bin_count > 0
    target_count_per_bin = int(target_calibration_size/(num_calibration_bins**3))
    
    for energy_bin_lower, energy_bin_upper in zip(energy_hist_bin_edges[:-1], energy_hist_bin_edges[1:]):
        for zenith_bin_lower, zenith_bin_upper in zip(zenith_hist_bin_edges[:-1], zenith_hist_bin_edges[1:]):
            for azimuth_bin_lower, azimuth_bin_upper in zip(azimuth_hist_bin_edges[:-1], azimuth_hist_bin_edges[1:]):
                available_to_sample_df = available_df[
                    (available_df["log10_energy"] >= energy_bin_lower) & (available_df["log10_energy"] < energy_bin_upper) &
                    (available_df["zenith"] >= zenith_bin_lower) & (available_df["zenith"] < zenith_bin_upper) &
                    (available_df["azimuth"] >= azimuth_bin_lower) & (available_df["azimuth"] < azimuth_bin_upper)
                ]
                if len(available_to_sample_df) < int(target_count_per_bin * cal_sample_ratio):
                    sampled_indexes = available_to_sample_df.index 
                else:
                    sampled_indexes = available_to_sample_df.sample(n=int(target_count_per_bin * cal_sample_ratio)).index 
                new_manifest.loc[sampled_indexes, "split"] = 2
    
    # Train Test
    for split_id, ratio, source_flux_func, source_name in zip(
        [1, 3],
        [train_ratio, test_ratio],
        [train_log10flux_of_log10energy_gev, test_log10flux_of_log10energy_gev],
        astropy_names
    ):
        print(f"Split {split_id}")
        available_df = new_manifest[new_manifest["split"] == 0]
        trajectory = get_source_trajectory(source_name, observer_latitude)
        source_min_zenith = np.deg2rad(90 - np.max(np.array(trajectory.alt)))
        
        sample_df = available_df[
            (available_df['zenith'] >= source_min_zenith) & (available_df["num_features"] >= minimum_features)
        ].sample(frac=ratio)
        new_manifest.loc[sample_df.index, "source"] = source_name
        new_manifest.loc[sample_df.index, "split"] = split_id

        print("Zenith Weighting")
        global_zenith = new_manifest["zenith"]
        global_zenith_bins = np.quantile(global_zenith, np.linspace(0, 1, num_train_test_bins + 1))
        global_zenith_hist, _ = np.histogram(global_zenith, global_zenith_bins)
        sample_zenith_hist, _ = np.histogram(sample_df["zenith"], global_zenith_bins)
        inverse_zenith_weights = sample_zenith_hist/global_zenith_hist
        nonzero_sum = (1/inverse_zenith_weights[inverse_zenith_weights > 0]).sum()
        inverse_zenith_weights *= nonzero_sum
        # inverse_zenith_weights = inverse_zenith_weights/inverse_zenith_weights.max()
        
        for bin_id, zenith_bin_lower, zenith_bin_upper in zip(range(len(global_zenith_bins) - 1), global_zenith_bins[:-1], global_zenith_bins[1:]):
            assert inverse_zenith_weights[bin_id] >= 0
            if inverse_zenith_weights[bin_id] == 0:
                continue
            
            zenith_weight = 1/inverse_zenith_weights[bin_id]
            bin_indexes = sample_df[
                (sample_df['zenith'] >= zenith_bin_lower) & 
                (sample_df['zenith'] < zenith_bin_upper)
            ].index 
            
            if zenith_weight * len(bin_indexes)/len(sample_df) < minimum_weight:
                continue
            
            new_manifest.loc[bin_indexes, "zenith_weight"] = zenith_weight
        
        print("Spectrum Weighting")  
        # pre-filter and bin
        log_energy_bins = np.quantile(sample_df["log10_energy"], np.linspace(0, 1, num_train_test_bins + 1))
        hist, _ = np.histogram(sample_df['log10_energy'], log_energy_bins)
        log_midpoints = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2
        energy_weights = 10**(source_flux_func(log_midpoints) + 10) / hist
        energy_weights = energy_weights/energy_weights.sum()
        for bin_id, energy_bin_lower, energy_bin_upper in zip(range(len(log_energy_bins) - 1), log_energy_bins[:-1], log_energy_bins[1:]):
            assert energy_weights[bin_id] > 0
            bin_indexes = sample_df[
                (sample_df['log10_energy'] >= energy_bin_lower) & 
                (sample_df['log10_energy'] < energy_bin_upper)
            ].index
            if energy_weights[bin_id] * len(bin_indexes)/len(sample_df) < minimum_weight:
                continue
            new_manifest.loc[bin_indexes, "energy_weight"] = energy_weights[bin_id]
        
        new_manifest.loc[
            (new_manifest["energy_weight"] == 0) & 
            (new_manifest["zenith_weight"] == 0) &
            (new_manifest["split"] == split_id)
        , "split"] = 0

    
    with open(filtered_manifest_filename, 'wb') as file:
        pkl.dump(new_manifest, file)
        
def generate_flexible_manifest(
    manifest_filename: str,
    filtered_manifest_filename: str,
    min_log10_energy_gev: float,
    max_log10_energy_gev: float,
    astropy_names: List[str],
    cal_sample_ratio: float,
    train_ratio: float,
    test_ratio: float,
    num_calibration_bins: int,
    minimum_features: int,
    target_calibration_size: int
):
    with open(manifest_filename, 'rb') as file:
        new_manifest = pkl.load(file)
    new_manifest["split"] = 0
    new_manifest["source"] = "NA"
    new_manifest["on_trajectory"] = True
    new_manifest["log10_energy"] = np.log10(new_manifest["energy"])
    
    # Prefilter
    new_manifest = new_manifest[
        (new_manifest["log10_energy"] > min_log10_energy_gev) & 
        (new_manifest["log10_energy"] < max_log10_energy_gev) &
        (new_manifest["num_features"] >= minimum_features)
    ]
    
    # Calibration
    print("Calibration Set")
    available_df = new_manifest[new_manifest["split"] == 0]
    
    _, energy_hist_bin_edges = np.histogram(available_df['log10_energy'], num_calibration_bins)
    _, zenith_hist_bin_edges = np.histogram(available_df['zenith'], num_calibration_bins)
    _, azimuth_hist_bin_edges = np.histogram(available_df['azimuth'], num_calibration_bins)

    target_count_per_bin = int(target_calibration_size/(num_calibration_bins**3))
    
    for energy_bin_lower, energy_bin_upper in zip(energy_hist_bin_edges[:-1], energy_hist_bin_edges[1:]):
        for zenith_bin_lower, zenith_bin_upper in zip(zenith_hist_bin_edges[:-1], zenith_hist_bin_edges[1:]):
            for azimuth_bin_lower, azimuth_bin_upper in zip(azimuth_hist_bin_edges[:-1], azimuth_hist_bin_edges[1:]):
                available_to_sample_df = available_df[
                    (available_df["log10_energy"] >= energy_bin_lower) & (available_df["log10_energy"] < energy_bin_upper) &
                    (available_df["zenith"] >= zenith_bin_lower) & (available_df["zenith"] < zenith_bin_upper) &
                    (available_df["azimuth"] >= azimuth_bin_lower) & (available_df["azimuth"] < azimuth_bin_upper)
                ]
                if len(available_to_sample_df) < int(target_count_per_bin * cal_sample_ratio):
                    sampled_indexes = available_to_sample_df.index 
                else:
                    sampled_indexes = available_to_sample_df.sample(n=int(target_count_per_bin * cal_sample_ratio)).index 
                new_manifest.loc[sampled_indexes, "split"] = 2
    
    # Train Test
    for split_id, ratio, source_name in zip(
        [1, 3],
        [train_ratio, test_ratio],
        astropy_names
    ):
        print(f"Split {split_id}")
        available_df = new_manifest[new_manifest["split"] == 0]
        sample_df = available_df.sample(frac=ratio)
        
        new_manifest.loc[sample_df.index, "source"] = source_name
        new_manifest.loc[sample_df.index, "split"] = split_id

    
    with open(filtered_manifest_filename, 'wb') as file:
        pkl.dump(new_manifest, file)
        
def generate_flexible_manifest_uniform_test(
    manifest_filename: str,
    filtered_manifest_filename: str,
    min_log10_energy_gev: float,
    max_log10_energy_gev: float,
    test_ratio: float,
    num_uniform_bins: int,
    minimum_features: int,
    target_cal_test_size: int,
    caltest_high_energy_sample_ratio: float
):
    with open(manifest_filename, 'rb') as file:
        new_manifest = pkl.load(file)
    new_manifest["split"] = 0
    new_manifest["source"] = "NA"
    new_manifest["on_trajectory"] = True
    new_manifest["log10_energy"] = np.log10(new_manifest["energy"])
    
    # Prefilter
    new_manifest = new_manifest[
        (new_manifest["log10_energy"] > min_log10_energy_gev) & 
        (new_manifest["log10_energy"] < max_log10_energy_gev) &
        (new_manifest["num_features"] >= minimum_features)
    ]
    
    # Calibration
    print("Calibration and Test Set")
    available_df = new_manifest[new_manifest["split"] == 0]
    
    _, energy_hist_bin_edges = np.histogram(available_df['log10_energy'], num_uniform_bins)
    _, zenith_hist_bin_edges = np.histogram(available_df['zenith'], num_uniform_bins)
    _, azimuth_hist_bin_edges = np.histogram(available_df['azimuth'], num_uniform_bins)
    energy_sample_ratios = np.linspace(1.0, caltest_high_energy_sample_ratio, num_uniform_bins)

    target_count_per_bin = int(target_cal_test_size/(num_uniform_bins**3))
    
    for energy_bin_lower, energy_bin_upper, e_sample_ratio in zip(energy_hist_bin_edges[:-1], energy_hist_bin_edges[1:], energy_sample_ratios):
        for zenith_bin_lower, zenith_bin_upper in zip(zenith_hist_bin_edges[:-1], zenith_hist_bin_edges[1:]):
            for azimuth_bin_lower, azimuth_bin_upper in zip(azimuth_hist_bin_edges[:-1], azimuth_hist_bin_edges[1:]):
                available_to_sample_df = available_df[
                    (available_df["log10_energy"] >= energy_bin_lower) & (available_df["log10_energy"] < energy_bin_upper) &
                    (available_df["zenith"] >= zenith_bin_lower) & (available_df["zenith"] < zenith_bin_upper) &
                    (available_df["azimuth"] >= azimuth_bin_lower) & (available_df["azimuth"] < azimuth_bin_upper)
                ]
                if len(available_to_sample_df) < int(target_count_per_bin):
                    sampled_indexes = available_to_sample_df.sample(frac=e_sample_ratio).index 
                else:
                    sampled_indexes = available_to_sample_df.sample(n=int(target_count_per_bin * e_sample_ratio)).index 
                new_manifest.loc[sampled_indexes, "split"] = 2
                
    caltest_df = new_manifest[new_manifest["split"] == 2]
    test_indexes = caltest_df.sample(frac=test_ratio).index 
    new_manifest.loc[test_indexes, "split"] = 3
    
    new_manifest.loc[new_manifest["split"] == 0, "split"] = 1
    
    with open(filtered_manifest_filename, 'wb') as file:
        pkl.dump(new_manifest, file)

def export_data_from_manifest(
        manifest: pd.DataFrame, 
        train_source: str,
        test_source: str,
        export_dir: str, 
        overwrite: bool = False
    ):
    split_names = ["train", "cal", "test"]
    file_id_regex = re.compile(r".*DAT([0-9]{6})\.pkl")
    
    for split_name, split_df_query in zip(
        split_names,
        [
            f"split == 1 and source == '{train_source}' and on_trajectory",
            "split == 2",
            f"split == 3 and source == '{test_source}' and on_trajectory"
        ],
    ):
        split_df = manifest.query(split_df_query)
        with tqdm(total=len(split_df)) as pbar:
            for source_filename in split_df['file'].unique():
                
                file_id = file_id_regex.search(source_filename).group(1)
                file_subset = split_df[split_df['file'] == source_filename]
                
                search_filename = export_dir + f"*{split_name}*{'small' if 'small' in source_filename else 'big'}*{file_id}*.pkl"
                search_results = glob.glob(search_filename)
                if not overwrite and len(search_results) > 0:
                    print(f"skipping {search_results[0]}")
                    pbar.update(len(file_subset))
                    continue
                
                with open(source_filename, 'rb') as file:
                    tiled_event_dict = pkl.load(file)
                
                file_events = list()
                for row in file_subset.itertuples():
                    pbar.update()
                    event_features = tiled_event_dict['events'][row.event_id]['features']
                    if event_features is None:
                        continue
                    
                    file_events.append(tiled_event_dict['events'][row.event_id])
                    
                export_filename = export_dir + f"vsi_{split_name}_{'small' if 'small' in source_filename else 'big'}_{file_id}_{len(file_subset)}_export.pkl"
                with open(export_filename, "wb") as file:
                    # print(f"Saving {export_filename}")
                    pkl.dump(file_events, file)
            