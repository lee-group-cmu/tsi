import util 
import cr_sources as crs
import numpy as np
import tiled_events as te
import seaborn as sns
import matplotlib.lines as lines
import posteriors as pos
import nde_models
import pandas as pd
import torch
import glob
import matplotlib as mpl
import os
import shutil
from stylesheets.register_roboto import register_roboto
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pickle as pkl
import matplotlib.ticker as ticker
import contextlib

register_roboto()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'''
     \usepackage{amsmath}
     \usepackage{amssymb}
     \usepackage{bm}
     \usepackage{underscore}
 '''

prior_colors = ["magenta", "olive", "dimgrey"]
prior_styles = ["solid", "dashed", "dotted"]
prior_alphas = (1, 0.8, 0.6)
original_data_color = "lightskyblue"
lf2i_color = "seagreen"
hpd_color = "purple"

truth_size = 150
panel_label_size = 22
title_font_size = 14
axis_font_size = 12.5
axis_label_size = 12.5
axis_tick_label_size = 10

def plot_priors(
    ax, 
    unfiltered_manifest, 
    filtered_manifest, 
    precomputed_weighters, 
    observer_latitude,
    skip_mrk, 
    x_axis, 
    y_axis, 
    num_samples, 
    resolution=10,
    legend_outside=False,
    alphas=(0.3, 0.3, 0.3),
    red_point=False,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # with plt.style.context(f"{current_dir}/stylesheets/538-roboto.mplstyle"):
    with contextlib.nullcontext() as _:
        crab_trajectory = crs.get_source_trajectory("crab", observer_latitude)
        crab_min_zenith = np.deg2rad(90 - np.array(crab_trajectory.alt).max())

        mrk_trajectory = crs.get_source_trajectory("mrk421", observer_latitude)
        mrk_min_zenith = np.deg2rad(90 - np.array(mrk_trajectory.alt).max())
        # levels = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
        levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

        for j, split_id, differential_flux, min_zenith, color, style, alpha in zip(
            [0, 1, 2],
            [3, 3, 3], 
            [crs.differential_crab_flux, crs.differential_mrk421_flux, crs.differential_dm_flux],
            [crab_min_zenith, mrk_min_zenith, crab_min_zenith],
            prior_colors,
            prior_styles,
            alphas
        ):
            if j == 1 and skip_mrk:
                continue
            split_df = filtered_manifest[(filtered_manifest["split"] == split_id) & (filtered_manifest["zenith"] >= min_zenith)].copy()
            if precomputed_weighters[j] is None:
                weighter = te.SplitParameterWeights(
                    unfiltered_manifest,
                    split_df,
                    differential_flux,
                    resolution
                )
                precomputed_weighters[j] = weighter
            else:
                weighter = precomputed_weighters[j]

            split_df["weight"] = weighter.get_weight(split_df["log10_energy"], split_df["zenith"])
            kde_df = split_df.sample(n=min(num_samples, len(split_df)), random_state=123)
            sns.kdeplot(
                y = kde_df[y_axis], 
                x = kde_df[x_axis], 
                weights=split_df["weight"],
                ax = ax, 
                levels= levels, 
                alpha = alpha,
                color= color,
                linestyles=style,
            )
        handles = [lines.Line2D([0], [0], color=color, linestyle=style) for color, style in zip(prior_colors, prior_styles)]
        labels = ["Crab", "Mrk421", "Dark Matter"]
        if red_point:
            star_handle = lines.Line2D(
                [0], [0], marker='*', color='red', linestyle='None', markersize=10, label='Example Event'
            )
            handles.append(star_handle)
            labels.append("Example Event")
        if legend_outside:
            legend = ax.legend(
                handles,
                labels,
                loc="upper center",  # Position legend relative to its bounding box
                bbox_to_anchor=(1.2, -0.1),  # Move outside the plot, below the lower-right corner
                frameon=False,  # Optional: remove the legend's frame
                ncol=3,         # Arrange items in a single row
            )
        else:
            legend = ax.legend(handles, labels)

        return legend

def coverage_boxplot(config, ax, coverage_estimator, precomputed_weighters, coverage_param_values, coverage_hits, plot_title, axis_font_size=16, title_font_size=20, major_linewidth=2, minor_linewidth=0.5):
    # with plt.style.context("stylesheets/538-roboto.mplstyle"):
    with contextlib.nullcontext():
        plot_param_grid = pos.get_param_grid(
            config.eval_param_mins,
            config.eval_param_maxes,
            100_00
        )

        crab_trajectory = crs.get_source_trajectory("crab", -15)
        crab_min_zenith = np.deg2rad(90 - np.array(crab_trajectory.alt).max())
        mrk_trajectory = crs.get_source_trajectory("mrk421", -15)
        mrk_min_zentih = np.deg2rad(90 - np.array(mrk_trajectory.alt).max())

        for i, w, min_zenith in zip([0, 1, 2], [0, 1, 2], [crab_min_zenith, mrk_min_zentih, crab_min_zenith]):
            
            zenith_mask = plot_param_grid[:, 1] >= min_zenith
            probs = coverage_estimator.predict_proba(plot_param_grid[zenith_mask].numpy())[:, 1]
            weights = precomputed_weighters[w].get_weight(plot_param_grid[zenith_mask][:, 0], plot_param_grid[zenith_mask][:, 1])
            
            q1 = util.weighted_percentile(probs, weights, 0.25)
            q2 = util.weighted_percentile(probs, weights, 0.5)
            q3 = util.weighted_percentile(probs, weights, 0.75)
            iqr = q3 - q1
            
            bottom_whisker = max(q1 - 1.5 * iqr, probs.min())
            top_whisker = min(q3 + 1.5 * iqr, probs.max())
            
            ax.fill_between([i - 0.1, i + 0.1], [q1, q1], [q3, q3], color='gray', alpha=0.5)
            
            ax.plot([i, i], [q1, bottom_whisker], color='black', linewidth=minor_linewidth)
            ax.plot([i - 0.1, i + 0.1], [bottom_whisker, bottom_whisker], color='black', linewidth=minor_linewidth)
            ax.plot([i, i], [q3, top_whisker], color='black', linewidth=minor_linewidth)
            ax.plot([i - 0.1, i + 0.1], [top_whisker, top_whisker], color='black', linewidth=minor_linewidth)
            
            ax.plot([i - 0.1, i + 0.1], [q2, q2], color='red', linewidth=minor_linewidth)
            
            global_zenith_mask = coverage_param_values[:, 1] >= min_zenith
            global_weights = precomputed_weighters[w].get_weight(coverage_param_values[global_zenith_mask][:, 0], coverage_param_values[global_zenith_mask][:, 1])
            avg_coverage = (coverage_hits[global_zenith_mask] * global_weights).sum() / global_weights.sum()
            ax.plot([i - 0.1, i + 0.1], [avg_coverage, avg_coverage], color='purple', linewidth=major_linewidth)
            

        xticks = [0, 1, 2]
        xlabels = ["Crab", "Mrk421", "Dark Matter"]
        ax.set_xticks(xticks, labels=xlabels)

        handles = [
            lines.Line2D([0], [0], color="black", linestyle="dashed"), 
            lines.Line2D([0], [0], color="red"),
            lines.Line2D([0], [0], color="purple")
        ]
        labels = [
            "90\% Nominal Coverage", 
            "Median Coverage",
            "Average Coverage"
        ]
        ax.legend(handles, labels, loc="lower left")

        ax.axhline(0.9, color='black', linestyle='dashed', linewidth=minor_linewidth)
        ax.set_xlabel(r"\textbf{Source}", fontsize=axis_font_size)
        ax.set_ylabel(r"\textbf{Coverage}", fontsize=axis_font_size)
        ax.set_title(plot_title, fontsize=title_font_size, pad=8)
        ax.set_ylim(-0.02, 1.02)
        
def plot_pointwise_coverage(
    config,
    fig,
    grid_spec,
    hpd_coverage_estimator,
    lf2i_coverage_estimator,
    axis_font_size=16,
    title_font_size=20
):

    cmap = plt.cm.inferno
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the colormap

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    min_color = 70
    bounds = np.linspace(min_color+2.5, 97.5, 6, dtype=float)
    bounds = [70] + list(bounds) + [100]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    grid_resolution = 70
    margin = 0.01

    plot_param_grid = pos.get_param_grid(
        config.eval_param_mins + margin,
        config.eval_param_maxes - margin,
        grid_resolution**3
    )

    plot_param_grid_fixed_az = pos.get_param_grid(
        config.eval_param_mins + margin,
        config.eval_param_maxes - margin,
        grid_resolution**2,
        fixed_azimuth=0
    )
    
    plot_param_grid_fixed_zenith = pos.get_param_grid(
        config.eval_param_mins + margin,
        config.eval_param_maxes - margin,
        grid_resolution**2,
        fixed_zenith=0
    )
    
    plot_param_grid_fixed_energy = pos.get_param_grid(
        config.eval_param_mins + margin,
        config.eval_param_maxes - margin,
        grid_resolution**2,
        fixed_energy=3.5
    )
    
    hpd_coverage_probs = hpd_coverage_estimator.predict_proba(plot_param_grid.numpy())[:, 1]
    lf2i_coverage_probs = lf2i_coverage_estimator.predict_proba(plot_param_grid.numpy())[:, 1]
    
    for ax_i, fixed_dim_index, non_fixed_dim_indexes, fixed_grid, axis_labels in zip(
        [0, 1, 2],
        [2, 1, 0],
        [[0, 1], [2, 0], [2, 1]],
        [plot_param_grid_fixed_az, plot_param_grid_fixed_zenith, plot_param_grid_fixed_energy],
        [
            [r"\textbf{Log$_{10}$ Energy (GeV)}", r"\textbf{Zenith Angle (rad)}"], 
            [r"\textbf{Azimuthal Angle (rad)}", r"\textbf{Log$_{10}$ Energy (GeV)}"], 
            [r"\textbf{Azimuthal Angle (rad)}", r"\textbf{Zenith Angle (rad)}"]
        ]
    ):
        
        ax = fig.add_subplot(grid_spec[0, ax_i])
        
        mean_coverage_probs = hpd_coverage_probs.reshape(grid_resolution, grid_resolution, grid_resolution).mean(axis=fixed_dim_index).flatten()
        ax.scatter(fixed_grid[:, non_fixed_dim_indexes[0]], fixed_grid[:, non_fixed_dim_indexes[1]], c=mean_coverage_probs*100, norm=norm, cmap=cmap)
        ax.set_xlabel(axis_labels[0], fontsize=axis_font_size)
        ax.set_ylabel(axis_labels[1], fontsize=axis_font_size)
        ax.set_xlim(config.eval_param_mins[non_fixed_dim_indexes[0]].item(), config.eval_param_maxes[non_fixed_dim_indexes[0]].item())
        ax.set_ylim(config.eval_param_mins[non_fixed_dim_indexes[1]].item(), config.eval_param_maxes[non_fixed_dim_indexes[1]].item())
        
        if ax_i == 1:
            ax.set_title(r"\textbf{Coverage Diagnostics: 90\% HPD Credible Sets}", fontsize=title_font_size, pad=18)

        ax = fig.add_subplot(grid_spec[1, ax_i])
        mean_coverage_probs = lf2i_coverage_probs.reshape(grid_resolution, grid_resolution, grid_resolution).mean(axis=fixed_dim_index).flatten()
        ax.scatter(fixed_grid[:, non_fixed_dim_indexes[0]], fixed_grid[:, non_fixed_dim_indexes[1]], c=mean_coverage_probs*100, norm=norm, cmap=cmap)
        ax.set_xlabel(axis_labels[0], fontsize=axis_font_size)
        ax.set_ylabel(axis_labels[1], fontsize=axis_font_size)
        ax.set_xlim(config.eval_param_mins[non_fixed_dim_indexes[0]].item(), config.eval_param_maxes[non_fixed_dim_indexes[0]].item())
        ax.set_ylim(config.eval_param_mins[non_fixed_dim_indexes[1]].item(), config.eval_param_maxes[non_fixed_dim_indexes[1]].item())
        
        if ax_i == 1:
            ax.set_title(r"\textbf{Coverage Diagnostics: 90\% FreB Confidence Sets}", fontsize=title_font_size, pad=18)


    # Add a single colorbar for all subplots to the right of the figure
    cbar_ax = fig.add_subplot(grid_spec[:, 3])  # Grid location: rows 2 and 3, last column
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, format='%1.2f', ticks=bounds, boundaries=bounds
    )
    
    # Format colorbar ticks and labels
    ticks = [r"$\leq$" + str(label) + r"\%" if label == min_color else str(label) + r"\%" for label in bounds]
    cbar.ax.yaxis.set_ticklabels(ticks)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.label.set_size(30)
    cbar.ax.set_ylabel(r"\textbf{Estimated Coverage}", fontsize=20)

def plot_paper_figure_1(
    config,
    test_ds,
    unfiltered_manifest,
    filtered_manifest,
    precomputed_weighters,
    hpd_coverage_estimator,
    hpd_param_values,
    joint_hpd_hits,
    lf2i_coverage_estimator,
    lf2i_param_values,
    joint_lf2i_hits,
    high_prior_density_obs_id,
    low_prior_Density_obs_id,
    axis_font_size=16,
    title_font_size=20
):
    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext() as _:
        fig = plt.figure(figsize=(13, 7.8))
        # fig.suptitle("Panel A: Global Coverage Analysis", fontweight="bold", fontsize=24)

        top_level_gridspec = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 0.5, 2])

        top_subgridspec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=top_level_gridspec[0])

        ax1 = fig.add_subplot(top_subgridspec[0])
        ax2 = fig.add_subplot(top_subgridspec[1])
        ax3 = fig.add_subplot(top_subgridspec[2])
    
        div_line = 0.639
        
        fig.add_artist(patches.Rectangle(
            (0, div_line),          # Lower-left corner in figure coordinates
            1, 1-div_line,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="darkgray",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (0, 0),          # Lower-left corner in figure coordinates
            1, div_line - 0.012,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="darkgray",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))

        fig.text(0.007, 0.01, r"\textbf{B}", fontsize=panel_label_size)
        fig.text(0.007, div_line + 0.01, r"\textbf{A}", fontsize=panel_label_size)

        # the following syntax does the same as the GridSpecFromSubplotSpec call above:
        bottom_subgridspec = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=top_level_gridspec[2], width_ratios=[1, 1, 1, 0.1, 0.2], height_ratios=[1, 1], hspace=0.65, wspace=0.40)


        plot_priors(
            ax1, 
            unfiltered_manifest, 
            filtered_manifest, 
            precomputed_weighters, 
            config.train_observer_latitude,
            False, 
            "log10_energy", 
            "zenith",
            10_000, 
            10, 
            alphas=prior_alphas
        )

        ax1.set_ylabel(r"\textbf{Zenith Angle (Rad)}", fontsize=axis_font_size)
        ax1.set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_font_size)
        ax1.set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
        ax1.set_ylim(config.eval_param_mins[1], config.eval_param_maxes[1])
        ax1.set_title(r"\textbf{Gamma Ray Source Distributions}", fontsize=title_font_size)

        for true_obs, true_color in zip([test_ds[high_prior_density_obs_id], test_ds[low_prior_Density_obs_id]], ["blue", "red"]):
            truth_slice = true_obs["params"].cpu().numpy()[:2]
            ax1.scatter(x=truth_slice[0], y=truth_slice[1], alpha=1, color=true_color, marker="*", s=truth_size, zorder=10)
            # ax1.axvline(true_obs["params"][0].item(), color=true_color, linestyle="dashed")
            # ax1.axhline(true_obs["params"][1].item(), color=true_color, linestyle="dashed")

        coverage_boxplot(
            config,
            ax2,
            hpd_coverage_estimator,
            precomputed_weighters,
            hpd_param_values,
            joint_hpd_hits,
            r"\textbf{Coverage of 90\% HPD Sets}",
            axis_font_size=axis_font_size,
            title_font_size=title_font_size
        )

        coverage_boxplot(
            config,
            ax3,
            lf2i_coverage_estimator,
            precomputed_weighters,
            lf2i_param_values,
            joint_lf2i_hits,
            r"\textbf{Coverage of 90\% FreB Sets}",
            axis_font_size=axis_font_size,
            title_font_size=title_font_size
        )

        plot_pointwise_coverage(
            config,
            fig,
            bottom_subgridspec,
            hpd_coverage_estimator,
            lf2i_coverage_estimator,
            axis_font_size=axis_font_size,
            title_font_size=title_font_size
        )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.03)

        return fig

from lf2i.plot.parameter_regions import plot_parameter_region_2D
from matplotlib.gridspec import GridSpec

def plot_paper_figure_2(
    config,
    hpd_lpd_obs_id_pair,
    data_sets, # [(high prior hpd, high prior lf2i), (low prior hpd, low prior lf2i)]
    test_ds, 
    hpd_coverage_estimator,
    lf2i_coverage_estimator,
    test_layout=False
):
    set_colors = [hpd_color, lf2i_color]
    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext() as _:
        hpd_obs, lpd_obs = test_ds[hpd_lpd_obs_id_pair[0]], test_ds[hpd_lpd_obs_id_pair[1]]

        fig, axs = plt.subplots(2, 3, figsize=(13, 5.6))
        
        div_line = 0.51
        fig.add_artist(patches.Rectangle(
            (0, div_line),          # Lower-left corner in figure coordinates
            1, 1-div_line,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="darkgray",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (0, 0),          # Lower-left corner in figure coordinates
            1, div_line - 0.015,           # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="darkgray",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.text(0.007, 0.01, r"\textbf{B}", fontsize=panel_label_size)
        fig.text(0.007, div_line + 0.01, r"\textbf{A}", fontsize=panel_label_size)
        
        for row, test_obs, data_set_pair, truth_color in zip([0, 1], [hpd_obs, lpd_obs], data_sets, ["blue", "red"]):
            hpd_coverage = hpd_coverage_estimator.predict_proba(
                np.array([test_obs["params"].numpy()])
            )[:, 1][0]
            
            waldo_coverage = lf2i_coverage_estimator.predict_proba(
                np.array([test_obs["params"].numpy()])
            )[:, 1][0]

            for point_set, color in zip(reversed(data_set_pair), reversed(set_colors)):
                for col, pair, param_component, slice_component, axis_labels in zip(
                    [0, 1, 2],
                    [["log10_energy", "zenith"], ["azimuth", "log10_energy"], ["azimuth", "zenith"]],
                    [[0, 1], [2, 0], [2, 1]],
                    [2, 1, 0],
                    [[r"\textbf{Log$_{10}$ Energy (GeV)}", r"\textbf{Zenith Angle (Rad)}"], 
                    [r"\textbf{Azimuthal Angle (Rad)}", r"\textbf{Log$_{10}$ Energy (GeV)}"], 
                    [r"\textbf{Azimuthal Angle (Rad)}", r"\textbf{Zenith Angle (Rad)}"]]
                ):
                    ax = axs[row, col] 
                    if not test_layout:
                        _slice, _ = pos.slice_param_set(point_set, slice_component, test_obs["params"][slice_component].item())
                        slice_points = _slice[:, [param_component[0], param_component[1]]]
                        truth_slice = test_obs["params"][[param_component[0], param_component[1]]].numpy()
                        ax.scatter(x=truth_slice.reshape(-1,)[0], y=truth_slice.reshape(-1,)[1], alpha=1, color=truth_color, marker="*", s=250, zorder=10)
                        
                        plot_parameter_region_2D(
                            slice_points,
                            None, 
                            custom_ax=ax,
                            scatter=False,
                            alpha_shape=True,
                            alpha=8,
                            color=color
                        )
                    
                        
                        ax.axvline(test_obs["params"][param_component[0]].item(), color=truth_color, linestyle="dashed")
                        ax.axhline(test_obs["params"][param_component[1]].item(), color=truth_color, linestyle="dashed")
                        ax.set_xlim(config.eval_param_mins[param_component[0]].item(), config.eval_param_maxes[param_component[0]].item())
                        ax.set_ylim(config.eval_param_mins[param_component[1]].item(), config.eval_param_maxes[param_component[1]].item())
                        ax.set_xlabel(axis_labels[0], fontsize=axis_font_size)
                        ax.set_ylabel(axis_labels[1], fontsize=axis_font_size)
                        
                        if col == 1:
                            ax.set_title(r"\textbf{" f"{['High', 'Low'][row]} " r"$\pi(\theta)$" f" - HPD: Coverage = {hpd_coverage * 100:0.0f}\% $\mid$ FreB: Coverage = {waldo_coverage *100:0.0f}\%" r"}", fontsize=title_font_size, pad = 8)
        
        handles = [lines.Line2D([0], [0], color=color) for color in set_colors]
        labels = ["90\% HPD Set", "90\% FreB Set"]
        axs[0, 0].legend(
            handles,
            labels
        )
        
        fig.tight_layout()
        return fig, axs
        
def plot_confidence_sets_internal(
    config, 
    model,
    qr, 
    device, 
    filtered_manifest, 
    unfiltered_manifest,
    precomputed_weighters, 
    test_ds, 
    obs_id,
    plot_param_grid_count, 
    save_dir=None,
    overwrite=True,
):
    
    test_obs = test_ds[obs_id]
    if save_dir is not None:
        save_filename = save_dir + f'sets_{obs_id}_E{test_obs["params"][0].item():0.2f}_Z{test_obs["params"][1].item():0.2f}_A{test_obs["params"][2].item():0.2f}.png'
        if os.path.exists(save_filename) and not overwrite:
            return
    
    is_npse = None 
    if type(model) is nde_models.WeightedFMPE:
        is_npse = False
        
    if type(model) is nde_models.WeightedNPSE:
        is_npse = True
        
    
    param_grid = pos.get_param_grid(
        config.eval_param_mins,
        config.eval_param_maxes,
        plot_param_grid_count
    )
    # high prior density
    hpd_set, _ = pos.get_hpd_set(
        model,
        param_grid,
        0.9,
        test_obs['features'],
        config.train_param_mins,
        config.train_param_maxes,
        hpd_set_type="joint",
        is_npse=is_npse
    )

    waldo_set, _, grid_crits = pos.lf2i_confidence_set(
        qr,
        model,
        config.calibration_num_posterior_samples,
        param_grid,
        config.train_param_mins,
        config.train_param_maxes,
        test_obs['features'].to(device),
        is_npse,
        config.use_posterior
    )

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for r in range(2):
        for i, ax, pair, param_component, slice_component in zip(
            [0, 1, 2],
            axs[r],
            [["log10_energy", "zenith"], ["azimuth", "log10_energy"], ["azimuth", "zenith"]],
            [[0, 1], [2, 0], [2, 1]],
            [2, 1, 0]
        ):
            plot_priors(ax, 
                unfiltered_manifest,
                filtered_manifest,
                precomputed_weighters,
                config.train_observer_latitude,
                True, pair[0], pair[1], 2_000)
            if r == 0:
                hpd_slice, _ = pos.slice_param_set(hpd_set, slice_component, test_obs["params"][slice_component].item())
                ax.scatter(
                    hpd_slice[:, param_component[0]],
                    hpd_slice[:, param_component[1]],
                    s=10,
                    color="orange",
                    alpha=0.5
                )
                ax.set_title("90% HPD Set")
            else:
                waldo_slice, _ = pos.slice_param_set(waldo_set, slice_component, test_obs["params"][slice_component].item())
                ax.scatter(
                    waldo_slice[:, param_component[0]],
                    waldo_slice[:, param_component[1]],
                    s=10,
                    color="green",
                    alpha=0.5
                )
                ax.set_title("90% LF2I (Waldo) Set")
            

            ax.scatter(
                test_obs["params"][param_component[0]].item(), 
                test_obs["params"][param_component[1]].item(),
                marker="x",
                color="red",
                s=50
            )
            
            ax.axvline(test_obs["params"][param_component[0]].item(), color="red", linestyle="dashed")
            ax.axhline(test_obs["params"][param_component[1]].item(), color="red", linestyle="dashed")
            ax.set_xlim(config.eval_param_mins[param_component[0]].item(), config.eval_param_maxes[param_component[0]].item())
            ax.set_ylim(config.eval_param_mins[param_component[1]].item(), config.eval_param_maxes[param_component[1]].item())
            
            if i == 0:
                colors = ["cornflowerblue",  "grey"]
                handles = [lines.Line2D([0], [0], color=color) for color in colors]
                labels = ["Crab", "Dark Matter"]
                ax.legend(handles, labels)
    
    if save_dir is not None:
        fig.savefig(save_filename)
        plt.close(fig)

def generate_examples(
    config, 
    config_name,
    out_dir, 
    model,
    qr,
    device,
    filtered_manifest,
    unfiltered_manifest,
    precomputed_weighters,
    test_ds, 
    test_manifest,
    plot_param_grid_count,
    debug=False,
    random_state=123
):
    # Number of bins for each column
    n_azimuth, n_zenith, n_log10_energy = 5, 5, 6  # Example bin counts

    # Define bin edges
    azimuth_bins = np.linspace(config.eval_param_mins[2], config.eval_param_maxes[2], n_azimuth + 1)
    zenith_bins = np.linspace(config.eval_param_mins[1], config.eval_param_maxes[1], n_zenith + 1)
    log10_energy_bins = np.linspace(config.eval_param_mins[0], config.eval_param_maxes[0], n_log10_energy + 1)

    # Find bin indices for each value
    test_manifest['azimuth_bin'] = np.digitize(test_manifest['azimuth_pm90'], azimuth_bins) - 1
    test_manifest['zenith_bin'] = np.digitize(test_manifest['zenith'], zenith_bins) - 1
    test_manifest['log10_energy_bin'] = np.digitize(test_manifest['log10_energy'], log10_energy_bins) - 1

    # Correct bins that fall on the rightmost edge
    test_manifest['azimuth_bin'] = test_manifest['azimuth_bin'].clip(0, n_azimuth - 1)
    test_manifest['zenith_bin'] = test_manifest['zenith_bin'].clip(0, n_zenith - 1)
    test_manifest['log10_energy_bin'] = test_manifest['log10_energy_bin'].clip(0, n_log10_energy - 1)

    # Calculate centroids
    azimuth_centroids = (azimuth_bins[:-1] + azimuth_bins[1:]) / 2
    zenith_centroids = (zenith_bins[:-1] + zenith_bins[1:]) / 2
    log10_energy_centroids = (log10_energy_bins[:-1] + log10_energy_bins[1:]) / 2

    # Map bin indices to centroids
    test_manifest['azimuth_centroid'] = test_manifest['azimuth_bin'].map(lambda x: azimuth_centroids[x])
    test_manifest['zenith_centroid'] = test_manifest['zenith_bin'].map(lambda x: zenith_centroids[x])
    test_manifest['log10_energy_centroid'] = test_manifest['log10_energy_bin'].map(lambda x: log10_energy_centroids[x])

    # Define a function to select a random row from a bin
    def select_random_row(bin_rows):
        if len(bin_rows) > 0:
            random_row = bin_rows.sample(1, random_state=random_state)  # Select a random row
            random_row['original_index'] = random_row.index  # Save the original index
            return random_row
        return None

    # Iterate over all bin combinations
    random_rows = []
    for i in range(n_azimuth):
        for j in range(n_zenith):
            for k in range(n_log10_energy):
                # Filter rows within the current bin
                bin_rows = test_manifest[
                    (test_manifest['azimuth_bin'] == i) &
                    (test_manifest['zenith_bin'] == j) &
                    (test_manifest['log10_energy_bin'] == k)
                ]
                
                if not bin_rows.empty:
                    # Select a random row from this bin
                    random_row = select_random_row(bin_rows)
                    if random_row is not None:
                        random_rows.append(random_row)

    # Combine all selected rows into a single DataFrame
    selected_rows_df = pd.concat(random_rows, ignore_index=True)
    selected_rows_df["plot_bin"] = selected_rows_df["zenith_bin"] * n_log10_energy + selected_rows_df["log10_energy_bin"]
    
    # Cheat Sheet
    fig, ax = plt.subplots()
    legend = plot_priors(
        ax, 
        unfiltered_manifest,
        filtered_manifest,
        precomputed_weighters=precomputed_weighters,
        observer_latitude=config.train_observer_latitude,
        skip_mrk=False,
        x_axis="log10_energy", 
        y_axis="zenith",
        num_samples=10_000,
        legend_outside=False
    )
    ax.scatter(selected_rows_df["log10_energy"], selected_rows_df["zenith"], marker='x', color='black')
    ax.set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
    ax.set_ylim(config.eval_param_mins[1], config.eval_param_maxes[1])

    for energy in log10_energy_bins:
        ax.axvline(energy, color='black', linestyle='--')

    for zenith in zenith_bins:
        ax.axhline(zenith, color='black', linestyle='--')

    j = 0
    # Iterate over cartesian product of energy and zenith grid
    for zenith, energy in torch.cartesian_prod(
        torch.tensor(zenith_centroids),
        torch.tensor(log10_energy_centroids), 
    ):
        ax.text(energy + 0.03, zenith - 0.04, str(j), fontsize=14, color='red')
        j += 1


    ax.set_xlabel("Log10 Energy (GeV)")
    ax.set_ylabel("Zenith (rad)")
    ax.set_title("Examples Cheat Sheet")
    if not os.path.exists(f"{out_dir}example_sets/"):
        os.makedirs(f"{out_dir}example_sets/")
    fig.savefig(f"{out_dir}example_sets/observations.png", bbox_extra_artists=(legend,))
    print(f"Saved Cheat Sheet to {out_dir}example_sets/observations.png")

    for row in tqdm(selected_rows_df.iterrows(), desc="Generating Example Sets", total=len(selected_rows_df)):
        save_dir = f"{out_dir}example_sets/{row[1]['plot_bin']:02.0f}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        test_filename = f"{save_dir}test_sets_{row[1]['original_index']:.0f}_E{row[1]['log10_energy']:0.2f}_Z{row[1]['zenith']:0.2f}_A{row[1]['azimuth_pm90']:0.2f}.empty"
        if os.path.exists(test_filename):
            os.remove(test_filename)
        
        if debug:
            with open(test_filename, "wb") as f:
                pass
        else:
            if test_ds[int(row[1]['original_index'])] is not None:
                plot_confidence_sets_internal(
                    config,
                    model,
                    qr,
                    device,
                    filtered_manifest,
                    unfiltered_manifest,
                    precomputed_weighters,
                    test_ds,
                    int(row[1]['original_index']),
                    plot_param_grid_count,
                    save_dir,
                    overwrite=False
                )
            else:
                print(f"Observation {row[1]['original_index']} is None")
            
    with open(f"{out_dir}example_sets/{config_name}.name", "wb") as f:
        pass
    
def fix_folder_structure(config, out_dir):
    # Number of bins for each column
    n_azimuth, n_zenith, n_log10_energy = 5, 5, 6  # Example bin counts

    # Define bin edges
    azimuth_bins = np.linspace(config.eval_param_mins[2], config.eval_param_maxes[2], n_azimuth + 1)
    zenith_bins = np.linspace(config.eval_param_mins[1], config.eval_param_maxes[1], n_zenith + 1)
    log10_energy_bins = np.linspace(config.eval_param_mins[0], config.eval_param_maxes[0], n_log10_energy + 1)

    file_list = glob.glob(f"{out_dir}example_sets/*/sets_*.png")
    file_data = list()
    for filename in file_list:
        file_params = filename.split("/")[-1].split("_")
        file_data.append({
            "filename": filename,
            "obs_id": int(file_params[1]),
            "log10_energy": float(file_params[2][1:]),
            "zenith": float(file_params[3][1:]),
            "azimuth_pm90": float(file_params[4][1:-4])
        })
        
    file_df = pd.DataFrame(file_data) 
    file_df['azimuth_bin'] = np.digitize(file_df['azimuth_pm90'], azimuth_bins) - 1
    file_df['zenith_bin'] = np.digitize(file_df['zenith'], zenith_bins) - 1
    file_df['log10_energy_bin'] = np.digitize(file_df['log10_energy'], log10_energy_bins) - 1

    # Correct bins that fall on the rightmost edge
    file_df['azimuth_bin'] = file_df['azimuth_bin'].clip(0, n_azimuth - 1)
    file_df['zenith_bin'] = file_df['zenith_bin'].clip(0, n_zenith - 1)
    file_df['log10_energy_bin'] = file_df['log10_energy_bin'].clip(0, n_log10_energy - 1)
    
    file_df["plot_bin"] = file_df["zenith_bin"] * n_log10_energy + file_df["log10_energy_bin"]
    
    if not os.path.exists(f"{out_dir}example_sets_fixed/"):
        os.makedirs(f"{out_dir}example_sets_fixed/")
        
    for row in file_df.iterrows():
        save_dir = f"{out_dir}example_sets_fixed/{row[1]['plot_bin']:02.0f}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        new_filename = f"{save_dir}test_sets_{row[1]['obs_id']:.0f}_E{row[1]['log10_energy']:0.2f}_Z{row[1]['zenith']:0.2f}_A{row[1]['azimuth_pm90']:0.2f}.png"
        shutil.copy(row[1]["filename"], new_filename)
        
        
def plot_appendix_1(plot_event):
    cbar_tick_label_size = 10
    # axis_label_size = 16
    # title_font_size = 20
    cbar_shrink = 0.66
    
    padding = 8
    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext():
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))

        # First Plot: Timing of Secondary Hits
        time_features = plot_event['features'][0]
        min_time = time_features[time_features.nonzero()[:, 0], time_features.nonzero()[:, 1]].min()
        max_time = time_features.max()
        time_features = torch.where(time_features == 0, min_time - 1, time_features)

        bounds = np.linspace(min_time, max_time, 6)

        the_plot = ax[0].imshow(time_features, cmap="inferno", vmin=min_time, vmax=max_time)
        cbar = fig.colorbar(the_plot, ticks=bounds, shrink=cbar_shrink)
        ticks = [fr"{label/1000:.0f}" for label in bounds]
        cbar.ax.yaxis.set_ticklabels(ticks)
        cbar.ax.tick_params(labelsize=cbar_tick_label_size)
        cbar.ax.set_ylabel(r"\textbf{Time since shower start ($\mu$s)}", fontsize=axis_label_size)
        ax[0].set_title(r"\textbf{Shower Arrival Time at Detectors}", fontsize=title_font_size, pad=padding)
        ax[0].tick_params(labelsize=axis_tick_label_size)

        # Second Plot
        max_count = np.log(31)
        bounds = np.linspace(0, max_count, 6)

        # the_plot = ax[1].imshow(torch.exp(plot_event['features'][1]/4 * 100) - 1, cmap="inferno", vmax=max_count)
        the_plot = ax[1].imshow(torch.log((torch.exp(plot_event['features'][1]/4) - 1) * 100 + 1), cmap="inferno", vmax=max_count)
        cbar = fig.colorbar(the_plot, ticks=bounds, shrink=cbar_shrink)
        ticks = [r"$\geq$" + fr"{np.exp(label) - 1:.1f}" if label == max_count else fr"{np.exp(label) - 1:.1f}" for label in bounds]
        cbar.ax.yaxis.set_ticklabels(ticks)
        cbar.ax.tick_params(labelsize=cbar_tick_label_size)
        cbar.ax.set_ylabel(r"\textbf{Hits per 100$m^2$}", fontsize=axis_label_size)
        ax[1].set_title(r"\textbf{Detector Counts: Main Group}", fontsize=title_font_size, pad=padding)
        ax[1].tick_params(labelsize=axis_tick_label_size)

        # Third Plot
        max_count = 0.1
        bounds = np.linspace(0, max_count, 6)

        the_plot = ax[2].imshow(torch.exp(plot_event['features'][2]/4 * 100) - 1, cmap="inferno", vmax=max_count)
        cbar = fig.colorbar(the_plot, ticks=bounds, shrink=cbar_shrink)
        ticks = [r"$\geq$" + fr"{label:.3f}" if label == max_count else fr"{label:.3f}" for label in bounds]
        cbar.ax.yaxis.set_ticklabels(ticks)
        cbar.ax.tick_params(labelsize=cbar_tick_label_size)
        cbar.ax.set_ylabel(r"\textbf{Hits per 100$m^2$}", fontsize=axis_label_size)
        ax[2].set_title(r"\textbf{Detector Counts: Secondary Group}", fontsize=title_font_size, pad=padding)
        ax[2].tick_params(labelsize=axis_tick_label_size)

        # Set 7 evenly spaced ticks on both x and y axes
        x_ticks = np.linspace(0, time_features.shape[1] - 1, 7)  # 7 ticks from start to end
        y_ticks = np.linspace(0, time_features.shape[0] - 1, 7)

        for i in range (3):
            ax[i].set_xticks(x_ticks)
            ax[i].set_xticklabels([fr"{tick * 40 - 1980:.0f}" for tick in x_ticks])  # Multiply labels by 4
            ax[i].set_yticks(y_ticks)
            ax[i].set_yticklabels([fr"{-tick * 40 + 1980:.0f}" for tick in y_ticks])
            ax[i].set_xlabel(r"\textbf{Easting (m)}", fontsize=axis_label_size)
            ax[i].set_ylabel(r"\textbf{Northing (m)}", fontsize=axis_label_size)

        fig.tight_layout()
        return fig



def plot_appendix_2(config, out_dir):
    # Define font sizes
    # title_font_size = 16  # Font size for plot titles
    # axis_label_size = 16  # Font size for axis labels
    # tick_label_size = 14  # Font size for axis tick labels

    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext():
        fig = plt.figure(figsize=(13, 11))
        num_bins = 10
        
        div_line1 = 0.655
        fig.add_artist(patches.Rectangle(
            (0, div_line1),          # Lower-left corner in figure coordinates
            1, 1-div_line1 - 0.035,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="darkgray",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (0.715, div_line1),          # Lower-left corner in figure coordinates
            0.01, 1-div_line1 - 0.035,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            linewidth=2,
            facecolor="darkgray",
            zorder=0  # Place the rectangle behind everything
        ))
        
        div_line2 = 0.33
        fig.add_artist(patches.Rectangle(
            (0, div_line2),          # Lower-left corner in figure coordinates
            1, div_line1 - div_line2 - 0.015,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="darkgray",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (0.715, div_line2),          # Lower-left corner in figure coordinates
            0.01, div_line1 - div_line2 - 0.015,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            linewidth=2,
            facecolor="darkgray",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (0, 0),          # Lower-left corner in figure coordinates
            1, div_line2 - 0.015,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="darkgray",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (0.715, 0),          # Lower-left corner in figure coordinates
            0.01, div_line2 - 0.015,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            linewidth=2,
            facecolor="darkgray",
            zorder=0  # Place the rectangle behind everything
        ))
        
        gs = gridspec.GridSpec(3, 5, figure=fig, width_ratios = [1, 1, 1, 0.01, 1])
        ax = np.zeros((3, 5), dtype=object)

        for i, manifest_file, split, text_offset in zip(
            range(3), 
            ["train_manifest.pkl", "cal_manifest.pkl", "test_manifest.pkl"], 
            ["Train", "Calibration", "Evaluation"],
            [0.30, 0.06, 0.08]
        ):
            for j in range(4):
                ax[i, j] = fig.add_subplot(gs[i, j if j < 3 else j + 1])
            with open(out_dir + manifest_file, "rb") as file:
                split_df = pkl.load(file)
            split_df = split_df.query(f"num_features >= {config.min_features_threshold}")
            if config.use_subsampling and i == 0:
                split_df = split_df[split_df["subsample_mask"]]
            n, _, _ = ax[i, 0].hist(split_df["log10_energy"], bins=num_bins, label="Data Distribution", color=original_data_color)
            print(f"{split}: {len(split_df)}")
            ax[i, 0].set_ylim(bottom=1)
            
            
            
            ax[i, 1].hist(split_df["zenith"], bins=num_bins, color=original_data_color)
             
           
            
            if i == 0:
                padding = 15
                ax[i, 1].legend(loc="lower left")
                ax[i, 0].set_title(r"\textbf{Histogram of Energy", fontsize=title_font_size, pad=padding)
                ax[i, 1].set_title(r"\textbf{Histogram of Zenith", fontsize=title_font_size, pad=padding)
                ax[i, 2].set_title(r"\textbf{Histogram of Azimuth", fontsize=title_font_size, pad=padding)
                ax[i, 3].set_title(r"\textbf{Histogram of Zenith/Energy", fontsize=title_font_size, pad=padding)
            
            if True:
                ax[i, 0].set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size)
                ax[i, 1].set_xlabel(r"\textbf{Zenith Angle (Rad)}", fontsize=axis_label_size)   
                ax[i, 2].set_xlabel(r"\textbf{Azimuthal Angle (Rad)}", fontsize=axis_label_size)
            
            ax[i, 2].hist(split_df["azimuth"], bins=num_bins, label="Original Data Distribution", color=original_data_color)
            


                

            num_2d_bins = 10
            x_bins = np.linspace(config.eval_param_mins[0], config.eval_param_maxes[0], num_2d_bins)  # Adjust number of bins if needed
            y_bins = np.linspace(config.eval_param_mins[1], config.eval_param_maxes[1], num_2d_bins)

            # Plot the histogram
            hist, xedges, yedges, img = ax[i, 3].hist2d(
                split_df['log10_energy'], 
                split_df['zenith'], 
                bins=[x_bins, y_bins], 
                cmap='inferno',
                vmin=0
            )

            # Add a colorbar
            cbar = fig.colorbar(img)
            cbar.set_label(r"\textbf{Count}", fontsize=axis_label_size)

            # Label the axes
            ax[i, 3].set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size)
            ax[i, 3].set_ylabel(r"\textbf{Zenith Angle (rad)}", fontsize=axis_label_size)
            # Set a title
            
            ax[i, 0].set_ylabel(r"\textbf{Count (1000s)}", fontsize=axis_label_size)
            ax[i, 0].text(-0.41, text_offset, r"\textbf{" + split + r"}", transform=ax[i, 0].transAxes, fontsize=30, rotation=90)
            
            for j in range(4):  # Iterate over all columns
                ax[i, j].tick_params(axis='both', labelsize=axis_tick_label_size)
                if j != 3:  # Skip the 2D histogram column for ylabel
                    ax[i, j].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: fr'{x / 1000:.0f}'))

        fig.tight_layout()
        return fig

        
def plot_appendix_3(
    config,
    unfiltered_manifest,
    filtered_manifest,
    train_manifest_df,
    precomputed_weighters
):
    # Define font sizes
    # title_font_size = 16  # Font size for plot titles
    # axis_label_size = 16  # Font size for axis labels
    # tick_label_size = 14  # Font size for axis tick labels

    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    padding = 20
    with contextlib.nullcontext():
        flux_bins = 100
        flux_grid = np.linspace(config.eval_param_mins[0], config.eval_param_maxes[0], flux_bins)
        crab_flux = crs.differential_crab_flux(flux_grid)
        mrk_flux = crs.differential_mrk421_flux(flux_grid)
        dm_flux = crs.differential_dm_flux(flux_grid)
        num_bins = 50

        fig, ax = plt.subplots(1, 3, figsize=(13, 5))

        plot_priors(
            ax[0],
            unfiltered_manifest,
            filtered_manifest,
            precomputed_weighters,
            config.train_observer_latitude,
            False,
            "log10_energy",
            "zenith",
            40_000,
            20,
            alphas=prior_alphas
        )
        ax[0].set_ylim(0, np.deg2rad(65))
        ax[0].set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
        ax[0].set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size)
        ax[0].set_ylabel(r"\textbf{Zenith Angle (Rad)}", fontsize=axis_label_size)
        ax[0].set_title(r"\begin{center}\textbf{Energy/Zenith Distribution \\ (3 Sources)}\end{center}", fontsize=title_font_size, pad=padding)

        ax[1].hist(train_manifest_df["log10_energy"], bins=num_bins, label="Unweighted Train Data Distribution", density=True, color=original_data_color)
        for flux_values, labels, color, style, alpha in zip(
            [crab_flux, mrk_flux, dm_flux],
            ["Crab Spectrum", "Mrk421 Spectrum", "DM Spectrum"],
            prior_colors,
            prior_styles,
            [0.3, 1, 1]
        ):
            ax[1].plot(flux_grid, flux_values/flux_values.sum() * flux_bins, label=labels, color=color, linestyle=style, alpha=alpha)

        ax[1].set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
        ax[1].set_title(r"\begin{center}\textbf{Energy Distribution \\ (Data vs Sources)}\end{center}", fontsize=title_font_size, pad=padding)
        ax[1].legend()
        ax[1].set_ylabel(r"\textbf{Normalized Density}", fontsize=axis_label_size)
        ax[1].set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size)

        crab_trajectory = crs.get_source_trajectory("crab", config.train_observer_latitude)
        crab_min_zenith = np.deg2rad(90 - np.max(np.array(crab_trajectory.alt)))
        mrk_trajectory = crs.get_source_trajectory("mrk421", config.train_observer_latitude)
        mrk_min_zenith = np.deg2rad(90 - np.max(np.array(mrk_trajectory.alt)))
        
        uniform_grid = np.linspace(0, np.deg2rad(65), 100)
        uniform_density = np.sin(uniform_grid)**2/(np.sin(uniform_grid)**2).sum() * 100

        ax[2].hist(train_manifest_df["zenith"], bins=num_bins, density=True, label="Unweighed Train Data Distribution", color=original_data_color)
        ax[2].hist(unfiltered_manifest["zenith"], bins=num_bins, density=True, alpha=0.5, label="Reference Distribution", color="grey")
        ax[2].plot(uniform_grid, uniform_density, label="Uniform Over Sphere", color="black")
        ax[2].axvline(crab_min_zenith, label="Crab Minimum Zenith", color=prior_colors[0], linestyle=prior_styles[0])
        ax[2].axvline(mrk_min_zenith, label="Mrk421 Minimum Zenith", color=prior_colors[1], linestyle=prior_styles[1])
        ax[2].set_xlim(0, np.deg2rad(65))
        ax[2].set_title(r"\begin{center}\textbf{Zenith Distribution \\ (Data vs Sources)} \end{center}", fontsize=title_font_size, pad=padding)
        ax[2].legend()
        ax[2].set_ylabel(r"\textbf{Normalized Density}", fontsize=axis_label_size)
        ax[2].set_xlabel(r"\textbf{Zenith Angle (rad)}", fontsize=axis_label_size)

        # Adjust tick label sizes for all axes
        for a in ax:
            a.tick_params(axis='both', labelsize=axis_tick_label_size)
        
        fig.tight_layout()
        
        return fig
    
def plot_appendix_4(out_dir):
    # Define font sizes
    # title_font_size = 20  # Font size for plot titles
    # axis_label_size = 16  # Font size for axis labels
    # tick_label_size = 14  # Font size for axis tick labels

    with open(out_dir + "sbi_train_summary.pkl", "rb") as file:
        sbi_train_summary = pkl.load(file)

    # Use a neutral style context for consistent formatting
    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext():
        fig, ax = plt.subplots(figsize=(13, 5))

        # Plot training and validation loss
        ax.plot(sbi_train_summary["training_loss"], label="Training Loss", color="tab:blue", linewidth=2)
        ax.plot(sbi_train_summary["validation_loss"], label="Validation Loss", color="tab:orange", linewidth=2, linestyle="--")

        # Set labels and title with specified font sizes
        ax.set_ylabel(r"\textbf{Weighted Conditional\newline Flow Matching Loss (MSE)}", fontsize=axis_label_size)
        ax.set_xlabel(r"\textbf{Epoch}", fontsize=axis_label_size)
        ax.set_title(r"\textbf{Training Loss: Flow Matching Posterior Estimator}", fontsize=title_font_size)

        # Adjust tick label size
        ax.tick_params(axis='both', labelsize=axis_tick_label_size)

        # Add legend with appropriate font size
        ax.legend(fontsize=axis_label_size)

        # Ensure layout is tight
        fig.tight_layout()

        return fig

    
    
def plot_appendix_5(config, model, device, plot_grid_length, dot_size, event_1, event_2):
    # Define adjustable parameters
    # title_font_size = 22  # Font size for plot titles
    # axis_label_size = 20  # Font size for axis labels
    # tick_label_size = 16  # Font size for axis tick labels
    # truth_marker_size = 300  # Size of the truth marker
    # colorbar_label_size = 16  # Font size for colorbar labels
    title_pad = 10

    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext():
        is_npse = None 
        if type(model) is nde_models.WeightedFMPE:
            is_npse = False
        elif type(model) is nde_models.WeightedNPSE:
            is_npse = True

        def get_log_probs(unscaled_param_grid, features):
            if config.use_sbi: 
                posterior = nde_models.SbiPosteriorLF2IWrapper(
                    model.build_posterior(
                        prior=torch.distributions.Uniform(-99, 99)
                    ),
                    3,
                    device,
                    is_npse
                )
                return posterior.log_prob(unscaled_param_grid, features)
            else:
                scaled_param_grid = (unscaled_param_grid.cpu() - config.train_param_mins) / (
                    config.train_param_maxes - config.train_param_mins
                )
                log_probs, _ = nde_models.log_probs(model, features, scaled_param_grid.to(device)) 
                return log_probs
        
        fig, ax = plt.subplots(3, 3, figsize=(13, 8))
        
        # div_line = 0.51
        # fig.add_artist(patches.Rectangle(
        #     (0, div_line),          # Lower-left corner in figure coordinates
        #     1, 1-div_line,            # Width and height in figure coordinates (normalized)
        #     transform=fig.transFigure,  # Use figure coordinates
        #     edgecolor="darkgray",
        #     linewidth=2,
        #     facecolor="gainsboro",
        #     zorder=0  # Place the rectangle behind everything
        # ))
        
        # fig.add_artist(patches.Rectangle(
        #     (0, 0),          # Lower-left corner in figure coordinates
        #     1, div_line - 0.015,           # Width and height in figure coordinates (normalized)
        #     transform=fig.transFigure,  # Use figure coordinates
        #     edgecolor="darkgray",
        #     linewidth=2,
        #     facecolor="gainsboro",
        #     zorder=0  # Place the rectangle behind everything
        # ))
        
        param_grid = pos.get_param_grid(
            config.eval_param_mins,
            config.eval_param_maxes,
            plot_grid_length**3
        ).to(device)
        
        for r, plot_obs, truth_color, truth_label, posterior_label, point_label in zip(
            [0, 1], 
            [event_1, event_2], 
            ["blue", "red"],
            ["True Value $(x_H)$", "True Value $(x_L)$"],
            ["Posterior $(x_H)$", "Posterior $(x_L)$"],
            ["$x_H$", "$x_L$"]
        ):
            
            posterior = get_log_probs(param_grid, plot_obs['features'].to(device)[None])
            
            # Energy Marginal
            just_energy_slice = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                plot_grid_length,
                fixed_zenith=plot_obs['params'][1].item(),
                fixed_azimuth=plot_obs['params'][2].item()
            ).to(device)
            just_energy_posterior = torch.log(torch.exp(posterior).reshape(plot_grid_length, plot_grid_length, plot_grid_length).mean(dim=2).mean(dim=1))
            ax[0, 0].plot(
                just_energy_slice[:, 0].cpu(),
                torch.exp(just_energy_posterior.cpu()),
                color=truth_color,
                label=posterior_label,
                alpha=0.7
            )
            ax[0, 0].fill_between(
                just_energy_slice[:, 0].cpu(),
                torch.exp(just_energy_posterior.cpu()),
                0,
                color=truth_color,
                alpha=0.3
            )
            ax[0, 0].axvline(
                plot_obs['params'][0].item(),
                color=truth_color,
                linestyle="dashed",
                label=truth_label,
                linewidth=2
            )
            ax[0, 0].set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
            ax[0, 0].set_ylim(0, None)



            # Zenith Marginal
            just_zenith_slice = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                plot_grid_length,
                fixed_energy=plot_obs['params'][0].item(),
                fixed_azimuth=plot_obs['params'][2].item()
            ).to(device)
            just_zenith_posterior = torch.log(torch.exp(posterior).reshape(plot_grid_length, plot_grid_length, plot_grid_length).mean(dim=2).mean(dim=0))
            ax[1, 1].plot(
                just_zenith_slice[:, 1].cpu(),
                torch.exp(just_zenith_posterior.cpu()),
                color=truth_color,
                label=posterior_label
            )
            ax[1, 1].fill_between(
                just_zenith_slice[:, 1].cpu(),
                torch.exp(just_zenith_posterior.cpu()),
                0,
                color=truth_color,
                alpha=0.3
            )
            ax[1, 1].axvline(
                plot_obs['params'][1].item(),
                color=truth_color,
                linestyle="dashed",
                label=truth_label,
                linewidth=2
            )
            ax[1, 1].set_xlim(config.eval_param_mins[1], config.eval_param_maxes[1])
            ax[1, 1].set_ylim(0, None)

            # Azimuth Marginal
            just_azimuth_slice = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                plot_grid_length,
                fixed_energy=plot_obs['params'][0].item(),
                fixed_zenith=plot_obs['params'][1].item(),
            ).to(device)
            just_azimuth_posterior = torch.log(torch.exp(posterior).reshape(plot_grid_length, plot_grid_length, plot_grid_length).mean(dim=1).mean(dim=0))
            ax[2, 2].plot(
                just_azimuth_slice[:, 2].cpu(),
                torch.exp(just_azimuth_posterior.cpu()),
                color=truth_color,
                label=posterior_label
            )
            ax[2, 2].fill_between(
                just_azimuth_slice[:, 2].cpu(),
                torch.exp(just_azimuth_posterior.cpu()),
                0,
                color=truth_color,
                alpha=0.3
            )
            ax[2, 2].axvline(
                plot_obs['params'][2].item() + (r*0.015),
                color=truth_color,
                linestyle="dashed",
                label=truth_label,
                linewidth=2
            )
            ax[2, 2].set_xlim(config.eval_param_mins[2], config.eval_param_maxes[2])
            ax[2, 2].set_ylim(0, None)

            # Fixed azimuth
            fixed_a_slice = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                plot_grid_length**2,
                fixed_azimuth=plot_obs['params'][2].item()
            ).to(device)
            fixed_a_posterior = torch.log(torch.exp(posterior).reshape(plot_grid_length, plot_grid_length, plot_grid_length).mean(dim=2))
            plotted = ax[1-r, r].scatter(
                fixed_a_slice[:, r].cpu(),
                fixed_a_slice[:, 1-r].cpu(),
                c=fixed_a_posterior.cpu(),
                s=dot_size,
                marker="s",
                cmap="inferno",
            )
            ax[1-r, r].scatter(
                plot_obs['params'][r].item(),
                plot_obs['params'][1-r].item(),
                marker="*",
                color=truth_color,
                s=truth_size
            )
            ax[1-r, r].axvline(plot_obs['params'][r].item(), color=truth_color, linestyle="dashed")
            ax[1-r, r].axhline(plot_obs['params'][1-r].item(), color=truth_color, linestyle="dashed")

            ax[1-r, r].set_xlim(config.eval_param_mins[r], config.eval_param_maxes[r])
            ax[1-r, r].set_ylim(config.eval_param_mins[1-r], config.eval_param_maxes[1-r])
            ax[1-r, r].set_xlabel([r"\textbf{Log$_{10}$ Energy (GeV)}", r"\textbf{Zenith Angle (rad)}"][r], fontsize=axis_label_size)
            ax[1-r, r].set_ylabel([r"\textbf{Zenith Angle (rad)}", r"\textbf{Log$_{10}$ Energy (GeV)}"][r], fontsize=axis_label_size)
            ax[1-r, r].tick_params(axis='both', labelsize=axis_tick_label_size)
            cbar = fig.colorbar(plotted, ax=ax[1-r, r])
            cbar.set_label(r"\textbf{Log density}", fontsize=axis_label_size)
            ax[1-r, r].set_title(r"\textbf{$" + ["(E, Z)", "(Z, E)"][r] +  r"$ Estimated Posterior (" + point_label + r")}", fontsize=title_font_size, pad=title_pad)

            # Fixed zenith
            fixed_z_slice = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                plot_grid_length**2,
                fixed_zenith=plot_obs['params'][1].item()
            ).to(device)
            fixed_z_posterior = torch.log(torch.exp(posterior).reshape(plot_grid_length, plot_grid_length, plot_grid_length).mean(dim=1))
            plotted = ax[2*(1-r), 2*r].scatter(
                fixed_z_slice[:, 2*r].cpu(),
                fixed_z_slice[:, 2*(1-r)].cpu(),
                c=fixed_z_posterior.cpu(),
                s=dot_size,
                marker="s",
                cmap="inferno"
            )
            ax[2*(1-r), 2*r].scatter(
                plot_obs['params'][2*r].item(),
                plot_obs['params'][2*(1-r)].item(),
                marker="*",
                color=truth_color,
                s=truth_size
            )
            ax[2*(1-r), 2*r].axvline(plot_obs['params'][2*r].item(), color=truth_color, linestyle="dashed")
            ax[2*(1-r), 2*r].axhline(plot_obs['params'][2*(1-r)].item(), color=truth_color, linestyle="dashed")

            ax[2*(1-r), 2*r].set_xlim(config.eval_param_mins[2*r], config.eval_param_maxes[2*r])
            ax[2*(1-r), 2*r].set_ylim(config.eval_param_mins[2*(1-r)], config.eval_param_maxes[2*(1-r)])
            ax[2*(1-r), 2*r].set_xlabel([r"\textbf{Log$_{10}$ Energy (GeV)}", r"\textbf{Azimuthal Angle (rad)}"][r], fontsize=axis_label_size)
            ax[2*(1-r), 2*r].set_ylabel([r"\textbf{Azimuthal Angle (rad)}", r"\textbf{Log$_{10}$ Energy (GeV)}"][r], fontsize=axis_label_size)
            ax[2*(1-r), 2*r].tick_params(axis='both', labelsize=axis_tick_label_size)
            cbar = fig.colorbar(plotted, ax=ax[2*(1-r), 2*r])
            cbar.set_label(r"\textbf{Log density}", fontsize=axis_label_size)
            ax[2*(1-r), 2*r].set_title(r"\textbf{$" + ["(E, A)", "(A, E)"][r] + r"$ Estimated Posterior (" + point_label + r")}", fontsize=title_font_size, pad=title_pad)

            # Fixed energy
            fixed_e_slice = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                plot_grid_length**2,
                fixed_energy=plot_obs['params'][0].item()
            ).to(device)
            fixed_e_posterior = torch.log(torch.exp(posterior).reshape(plot_grid_length, plot_grid_length, plot_grid_length).mean(dim=0))
            plotted = ax[2-r, r+1].scatter(
                fixed_e_slice[:, r+1].cpu(),
                fixed_e_slice[:, 2-r].cpu(),
                c=fixed_e_posterior.cpu(),
                s=dot_size,
                marker="s",
                cmap="inferno",
            )
            ax[2-r, r+1].scatter(
                plot_obs['params'][r+1].item(),
                plot_obs['params'][2-r].item(),
                marker="*",
                color=truth_color,
                s=truth_size
            )
            ax[2-r, r+1].axvline(plot_obs['params'][r+1].item(), color=truth_color, linestyle="dashed")
            ax[2-r, r+1].axhline(plot_obs['params'][2-r].item(), color=truth_color, linestyle="dashed")

            ax[2-r, r+1].set_xlim(config.eval_param_mins[r+1], config.eval_param_maxes[r+1])
            ax[2-r, r+1].set_ylim(config.eval_param_mins[2-r], config.eval_param_maxes[2-r])
            ax[2-r, r+1].set_xlabel([r"\textbf{Zenith Angle (rad)}", r"\textbf{Azimuthal Angle (rad)}"][r], fontsize=axis_label_size)
            ax[2-r, r+1].set_ylabel([r"\textbf{Azimuthal Angle (rad)}", r"\textbf{Zenith Angle (rad)}"][r], fontsize=axis_label_size)
            ax[2-r, r+1].tick_params(axis='both', labelsize=axis_tick_label_size)
            cbar = fig.colorbar(plotted, ax=ax[2-r, r+1])
            cbar.set_label(r"\textbf{Log density}", fontsize=axis_label_size)
            ax[2-r, r+1].set_title(r"\textbf{$" + ["(Z, A)", "(A, Z)"][r] + r"$ Estimated Posterior (" + point_label + r")}", fontsize=title_font_size, pad=title_pad)

        
        ax[0, 0].legend()
        ax[0, 0].set_title(r"\textbf{Energy Estimated Posterior}", fontsize=title_font_size, pad=title_pad)
        ax[0, 0].set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size)
        ax[0, 0].set_ylabel(r"\textbf{Posterior Density}", fontsize=axis_label_size)
        ax[1, 1].legend()
        ax[1, 1].set_title(r"\textbf{Zenith Estimated Posterior}", fontsize=title_font_size, pad=title_pad)
        ax[1, 1].set_xlabel(r"\textbf{Zenith Angle (rad)}", fontsize=axis_label_size)
        ax[1, 1].set_ylabel(r"\textbf{Posterior Density}", fontsize=axis_label_size)
        ax[2, 2].legend()
        ax[2, 2].set_title(r"\textbf{Azimuth Estimated Posterior}", fontsize=title_font_size, pad=title_pad)
        ax[2, 2].set_xlabel(r"\textbf{Azimuthal Angle (rad)}", fontsize=axis_label_size)
        ax[2, 2].set_ylabel(r"\textbf{Posterior Density}", fontsize=axis_label_size)

        fig.tight_layout()
        return fig


    
def plot_appendix_6(config, qr, test_obs, lf2i_set):
    # Define adjustable parameters
    # title_font_size = 20  # Font size for plot titles
    # axis_label_size = 18  # Font size for axis labels
    # tick_label_size = 16  # Font size for axis tick labels
    # colorbar_label_size = 18  # Font size for colorbar labels
    marker_dot_size = 20  # Default marker dot size for scatter plot points

    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext():
        fig = plt.figure(figsize=(13, 10))
        ax = np.zeros(9, dtype=object)
        
        gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

        azimuth_grid = np.linspace(
            config.eval_param_mins[2] + 0.01,
            config.eval_param_maxes[2] - 0.01,
            ax.shape[0]
        )

        for i, azimuth in enumerate(azimuth_grid):
            ax[i] = fig.add_subplot(gs[i // 3, i % 3])
            fixed_azimuth = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                10_000,
                fixed_azimuth=azimuth
            )
            critical_values = qr.predict(fixed_azimuth.numpy())
            plot = ax[i].scatter(
                fixed_azimuth[:, 0],
                fixed_azimuth[:, 1],
                c=critical_values,
                s=marker_dot_size,
                vmax=0.9,
                vmin=-8,
                cmap="inferno"
            )
        
            ax[i].set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
            ax[i].set_ylim(config.eval_param_mins[1], config.eval_param_maxes[1])
            if i == 7:
                ax[i].set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size + 6)
            if i == 3:
                ax[i].set_ylabel(r"\textbf{Zenith Angle (rad)}", fontsize=axis_label_size + 6)
            ax[i].tick_params(axis='both', labelsize=axis_tick_label_size)
            if i == 4:
                ax[i].set_title(r"\t" f"extbf{{$\mathbf{{A = {azimuth:.2f}}}$ rad ($\mathbf{{A_i = {test_obs['params'][2]:0.2f}}}$)}}", fontsize=title_font_size)
            else:
                ax[i].set_title(r"\t" f"extbf{{$\mathbf{{A = {azimuth:.2f}}}$ rad }}", fontsize=title_font_size)
            
            truth_color = "red"
            _slice, _ = pos.slice_param_set(lf2i_set, 2, azimuth)
            slice_points = _slice[:, [0, 1]]
            truth_slice = test_obs["params"][[0, 1]].numpy()
            if i == 4:
                ax[i].scatter(x=truth_slice.reshape(-1,)[0], y=truth_slice.reshape(-1,)[1], alpha=1, color=truth_color, marker="*", s=250, zorder=10)
            
            if slice_points.shape[0] > 2:
                plot_parameter_region_2D(
                    slice_points,
                    None, 
                    custom_ax=ax[i],
                    scatter=False,
                    alpha_shape=True,
                    alpha=8,
                    color=lf2i_color
                )
        
            alpha = 1 if i == 4 else 0.3
            ax[i].axvline(test_obs["params"][0].item(), color=truth_color, linestyle="dashed", alpha=alpha)
            ax[i].axhline(test_obs["params"][1].item(), color=truth_color, linestyle="dashed", alpha=alpha)
            
        cbar_ax = fig.add_subplot(gs[:, 3])
        cbar = fig.colorbar(plot, cax=cbar_ax)
        cbar.set_label(r"\textbf{Critical Value}", fontsize=axis_label_size)

        # Modify the color bar tick labels
        ticks = cbar.ax.get_yticks()  # Get the tick positions
        tick_labels = [f"{tick:.0f}" for tick in ticks]  # Create labels
        tick_labels[0] = r"$\leq$" f" {tick_labels[0]}"  # Add "<=" to the first label
        cbar.ax.set_yticklabels(tick_labels, fontsize=axis_tick_label_size)  # Set the modified labels

        fig.suptitle(r"\textbf{90\% FreB Critical Values}", fontsize=title_font_size + 6)
        fig.tight_layout()
    return fig


def plot_appendix_7(config, hpd_coverage_estimator):
    # Adjustable parameters
    # title_font_size = 18  # Font size for plot titles
    # axis_label_size = 16  # Font size for axis labels
    # tick_label_size = 16  # Font size for axis tick labels
    # colorbar_label_size = 18  # Font size for colorbar labels
    # colorbar_tick_size = 14  # Font size for colorbar ticks
    marker_dot_size = 20  # Default marker dot size for scatter plot points
    figsize = (13, 11)  # Figure size

    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext():
        # Customize colormap and normalization
        cmap = plt.cm.inferno
        cmaplist = [cmap(i) for i in range(cmap.N)]  # Extract all colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

        min_color = 70
        bounds = np.linspace(min_color + 2.5, 97.5, 6, dtype=float)
        bounds = [70] + list(bounds) + [100]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # Initialize figure and layout
        fig = plt.figure(figsize=figsize)
        grid_spec = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

        azimuth_grid = np.linspace(config.eval_param_mins[2] + 0.01, config.eval_param_maxes[2] - 0.01, 9)

        # Iterate through azimuths and create scatter plots
        for i, azimuth in enumerate(azimuth_grid):
            ax = fig.add_subplot(grid_spec[i // 3, i % 3])
            fixed_azimuth = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                10_000,
                fixed_azimuth=azimuth
            )
            coverage_values = hpd_coverage_estimator.predict_proba(fixed_azimuth.numpy())[:, 1] * 100
            plot = ax.scatter(
                fixed_azimuth[:, 0],
                fixed_azimuth[:, 1],
                c=coverage_values,
                s=marker_dot_size,
                norm=norm,
                cmap="inferno"
            )
            ax.set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
            ax.set_ylim(config.eval_param_mins[1], config.eval_param_maxes[1])
            if i == 7:
                ax.set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size + 6)
            if i == 3:
                ax.set_ylabel(r"\textbf{Zenith Angle (rad)}", fontsize=axis_label_size + 6)
            ax.tick_params(axis='both', labelsize=axis_tick_label_size)
            ax.set_title(r"\t" f"extbf{{$\mathbf{{A = {azimuth:.2f}}}$ rad }}", fontsize=title_font_size)

        # Add a single colorbar for the entire figure
        cbar_ax = fig.add_subplot(grid_spec[:, 3])
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            format='%1.2f',
            ticks=bounds,
            boundaries=bounds
        )
        ticks = [r"$\leq$" + str(label) + "%" if label == min_color else str(label) + "%" for label in bounds]
        cbar.ax.yaxis.set_ticklabels(ticks)

        cbar.ax.set_ylabel(r"\textbf{Estimated Coverage}", fontsize=axis_label_size)
        

        # Adjust layout
        fig.suptitle(r"\textbf{Coverage of 90\% HPD Sets (Unaveraged)}", fontsize=title_font_size + 6)
        fig.tight_layout()
    return fig


def plot_appendix_8(config, lf2i_coverage_estimator):
    # Adjustable parameters
    # title_font_size = 18  # Font size for plot titles
    # axis_label_size = 16  # Font size for axis labels
    # tick_label_size = 16  # Font size for axis tick labels
    # colorbar_label_size = 18  # Font size for colorbar labels
    # colorbar_tick_size = 14  # Font size for colorbar ticks
    marker_dot_size = 20  # Default marker dot size for scatter plot points
    figsize = (13, 11)  # Figure size

    # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):
    with contextlib.nullcontext():
        # Customize colormap and normalization
        cmap = plt.cm.inferno
        cmaplist = [cmap(i) for i in range(cmap.N)]  # Extract all colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

        min_color = 70
        bounds = np.linspace(min_color + 2.5, 97.5, 6, dtype=float)
        bounds = [70] + list(bounds) + [100]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # Initialize figure and layout
        fig = plt.figure(figsize=figsize)
        grid_spec = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

        azimuth_grid = np.linspace(config.eval_param_mins[2] + 0.01, config.eval_param_maxes[2] - 0.01, 9)

        # Iterate through azimuths and create scatter plots
        for i, azimuth in enumerate(azimuth_grid):
            ax = fig.add_subplot(grid_spec[i // 3, i % 3])
            fixed_azimuth = pos.get_param_grid(
                config.eval_param_mins,
                config.eval_param_maxes,
                10_000,
                fixed_azimuth=azimuth
            )
            coverage_values = lf2i_coverage_estimator.predict_proba(fixed_azimuth.numpy())[:, 1] * 100
            plot = ax.scatter(
                fixed_azimuth[:, 0],
                fixed_azimuth[:, 1],
                c=coverage_values,
                s=marker_dot_size,
                norm=norm,
                cmap="inferno"
            )
            ax.set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
            ax.set_ylim(config.eval_param_mins[1], config.eval_param_maxes[1])
            
            if i == 7:
                ax.set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_label_size + 6)
            if i == 3:
                ax.set_ylabel(r"\textbf{Zenith Angle (rad)}", fontsize=axis_label_size + 6)
            ax.tick_params(axis='both', labelsize=axis_tick_label_size)
            ax.set_title(r"\t" f"extbf{{$\mathbf{{A = {azimuth:.2f}}}$ rad }}", fontsize=title_font_size)
        
        # Add a single colorbar for the entire figure
        cbar_ax = fig.add_subplot(grid_spec[:, 3])
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            format='%1.2f',
            ticks=bounds,
            boundaries=bounds
        )
        ticks = [r"$\leq$" + str(label) + "%" if label == min_color else str(label) + "%" for label in bounds]
        cbar.ax.yaxis.set_ticklabels(ticks)
        cbar.ax.set_ylabel(r"\textbf{Estimated Coverage}", fontsize=axis_label_size)

        # Adjust layout
        fig.suptitle(r"\textbf{Coverage of 90\% FreB Sets (Unaveraged)}", fontsize=title_font_size + 6)
        fig.tight_layout()
    return fig

def plot_new_fig_1(
    config, 
    unfiltered_manifest, 
    filtered_manifest, 
    precomputed_weighters,
    plot_event,
    hpd_coverage_estimator,
    hpd_param_values,
    joint_hpd_hits,
    lf2i_coverage_estimator,
    lf2i_param_values,
    joint_lf2i_hits,
    test_obs,
    data_set_pair, # (low prior hpd, low prior lf2i)
    test_obs_hp,
    data_set_pair_hp,
    test_layout_only=False,
    pickled_objects_file=None,
    overwrite_pickle=False
):
    if pickled_objects_file is not None:
        if not os.path.exists(pickled_objects_file) or overwrite_pickle:
            assert unfiltered_manifest is not None
            assert filtered_manifest is not None
            assert precomputed_weighters is not None
            assert plot_event is not None
            assert hpd_coverage_estimator is not None
            assert hpd_param_values is not None
            assert joint_hpd_hits is not None
            assert lf2i_coverage_estimator is not None
            assert lf2i_param_values is not None
            assert joint_lf2i_hits is not None
            assert test_obs is not None
            assert data_set_pair is not None
            assert test_obs_hp is not None
            assert data_set_pair_hp is not None

            # Save the objects to a pickle file
            with open(pickled_objects_file, 'wb') as f:
                pkl.dump({
                    "unfiltered_manifest": unfiltered_manifest,
                    "filtered_manifest": filtered_manifest,
                    "precomputed_weighters": precomputed_weighters,
                    "plot_event": plot_event,
                    "hpd_coverage_estimator": hpd_coverage_estimator,
                    "hpd_param_values": hpd_param_values,
                    "joint_hpd_hits": joint_hpd_hits,
                    "lf2i_coverage_estimator": lf2i_coverage_estimator,
                    "lf2i_param_values": lf2i_param_values,
                    "joint_lf2i_hits": joint_lf2i_hits,
                    "test_obs": test_obs,
                    "data_set_pair": data_set_pair,
                    "test_obs_hp": test_obs_hp,
                    "data_set_pair_hp": data_set_pair_hp
                }, f)
        elif os.path.exists(pickled_objects_file):
            # Load the objects from the pickle file
            with open(pickled_objects_file, 'rb') as f:
                data = pkl.load(f)
                unfiltered_manifest = data["unfiltered_manifest"]
                filtered_manifest = data["filtered_manifest"]
                precomputed_weighters = data["precomputed_weighters"]
                plot_event = data["plot_event"]
                hpd_coverage_estimator = data["hpd_coverage_estimator"]
                hpd_param_values = data["hpd_param_values"]
                joint_hpd_hits = data["joint_hpd_hits"]
                lf2i_coverage_estimator = data["lf2i_coverage_estimator"]
                lf2i_param_values = data["lf2i_param_values"]
                joint_lf2i_hits = data["joint_lf2i_hits"]
                test_obs = data["test_obs"]
                data_set_pair = data["data_set_pair"]
                test_obs_hp = data["test_obs_hp"]
                data_set_pair_hp = data["data_set_pair_hp"]

    with contextlib.nullcontext():
        fig = plt.figure(figsize=(13, 10))
        
        v_div = 0.39
        h_div1 = 0.37
        h_div2 = 0.70
        spacing = 0.012
        bigspacing = 0.02
        fig.add_artist(patches.Rectangle(
            (0, v_div),          # Lower-left corner in figure coordinates
            h_div1 - spacing, 1-v_div,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="black",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (h_div1, v_div),          # Lower-left corner in figure coordinates
            h_div2 - h_div1 - bigspacing, 1-v_div,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="black",
            linewidth=2,
            facecolor="gainsboro", #"lightcoral",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (h_div2, v_div),          # Lower-left corner in figure coordinates
            1 - h_div2, 1-v_div,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="black",
            linewidth=2,
            facecolor="gainsboro", #"lightgreen",
            zorder=0  # Place the rectangle behind everything
        ))
        
        fig.add_artist(patches.Rectangle(
            (0, 0),          # Lower-left corner in figure coordinates
            1, v_div - spacing,            # Width and height in figure coordinates (normalized)
            transform=fig.transFigure,  # Use figure coordinates
            edgecolor="black",
            linewidth=2,
            facecolor="gainsboro",
            zorder=0  # Place the rectangle behind everything
        ))

        fig.text(0.007, v_div + 0.01, r"\textbf{A}", fontsize=panel_label_size)
        fig.text(h_div1 + 0.007, v_div + 0.01, r"\textbf{B}", fontsize=panel_label_size)
        fig.text(h_div2 + 0.007, v_div + 0.01, r"\textbf{C}", fontsize=panel_label_size)
        fig.text(0.007, 0.01, r"\textbf{D}", fontsize=panel_label_size)
        
        top_level_gridspec = gridspec.GridSpec(
            3, 3, 
            width_ratios=[1, 0.01, 2], 
            height_ratios=[2, 0.01, 1]
        )

        fig.text(0.16, 0.97, r"\textbf{Data}", fontsize=title_font_size+3)
        fig.text(0.378, 0.97, r"\textbf{Local Coverage: 90\% HPD Sets}", fontsize=title_font_size+3)
        fig.text(0.710, 0.97, r"\textbf{Local Coverage: 90\% FreB Sets}", fontsize=title_font_size+3)
        
        height_ratios = [0.3, 1, 0.45, 1]
        panel_a_spec = gridspec.GridSpecFromSubplotSpec(
            4, 1, 
            subplot_spec=top_level_gridspec[0, 0],
            height_ratios=height_ratios,
            hspace=0
        )
        priors_ax = fig.add_subplot(panel_a_spec[1, 0])
        data_ax = fig.add_subplot(panel_a_spec[3, 0])
        if not test_layout_only:
            truth_color = "red"
            priors_ax.scatter(
                x=test_obs['params'][0], 
                y=test_obs['params'][1], 
                alpha=1, 
                color=truth_color, 
                marker="*", 
                s=250, 
                zorder=10,
                label="Example Event"
            )
            
            plot_priors(
                priors_ax, 
                unfiltered_manifest, 
                filtered_manifest, 
                precomputed_weighters, 
                config.train_observer_latitude,
                False, 
                "log10_energy", 
                "zenith",
                10_000, 
                10, 
                alphas=prior_alphas,
                red_point=True
            )      
        
        priors_ax.set_ylabel(r"\textbf{Zenith Angle (Rad)}", fontsize=axis_font_size)
        priors_ax.set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_font_size)
        priors_ax.set_xlim(config.eval_param_mins[0], config.eval_param_maxes[0])
        priors_ax.set_ylim(config.eval_param_mins[1], config.eval_param_maxes[1])
        priors_ax.set_title(r"\textbf{Gamma Ray Source Distributions}", fontsize=title_font_size)

        
        cbar_tick_label_size = 10
        # axis_label_size = 16
        # title_font_size = 20
        cbar_shrink = 0.9
        
        padding = 8
        # with plt.style.context("stylesheets/538-roboto-nogrid.mplstyle"):

        time_features = plot_event['features'][0]
        min_time = time_features[time_features.nonzero()[:, 0], time_features.nonzero()[:, 1]].min()
        max_time = time_features.max()
        time_features = torch.where(time_features == 0, min_time - 1, time_features)

        bounds = np.linspace(min_time, max_time, 6)
        if not test_layout_only:
            the_plot = data_ax.imshow(time_features, cmap="inferno", vmin=min_time, vmax=max_time)
            cbar = fig.colorbar(the_plot, ticks=bounds, shrink=cbar_shrink)
            ticks = [fr"{label/1000:.0f}" for label in bounds]
            cbar.ax.yaxis.set_ticklabels(ticks)
            cbar.ax.tick_params(labelsize=cbar_tick_label_size)
            cbar.ax.set_ylabel(r"\textbf{Time since start ($\mu$s)}", fontsize=axis_label_size)
        
        data_ax.set_title(r"\textbf{Example Event Data: Detector Timing}", fontsize=title_font_size, pad=padding)
        data_ax.tick_params(labelsize=axis_tick_label_size)

        # Set 7 evenly spaced ticks on both x and y axes
        x_ticks = np.linspace(0, time_features.shape[1] - 1, 7)  # 7 ticks from start to end
        y_ticks = np.linspace(0, time_features.shape[0] - 1, 7)

        data_ax.set_xticks(x_ticks)
        data_ax.set_xticklabels([fr"{tick * 40 - 1980:.0f}" for tick in x_ticks])  # Multiply labels by 4
        data_ax.set_yticks(y_ticks)
        data_ax.set_yticklabels([fr"{-tick * 40 + 1980:.0f}" for tick in y_ticks])
        data_ax.set_xlabel(r"\textbf{Easting (m)}", fontsize=axis_label_size)
        data_ax.set_ylabel(r"\textbf{Northing (m)}", fontsize=axis_label_size)


        panel_b_spec = gridspec.GridSpecFromSubplotSpec(
            4, 3,
            subplot_spec=top_level_gridspec[0, 2],
            height_ratios=height_ratios,
            width_ratios=[1, 0.4, 1],
            hspace=0
        )
        
        before_boxplot_ax = fig.add_subplot(panel_b_spec[3, 0])
        after_boxplot_ax = fig.add_subplot(panel_b_spec[3, 2])
        if not test_layout_only:
            coverage_boxplot(
                config,
                before_boxplot_ax,
                hpd_coverage_estimator,
                precomputed_weighters,
                hpd_param_values,
                joint_hpd_hits,
                r"\begin{center}\textbf{Per Gamma-Ray Source}\end{center}",
                axis_font_size=axis_font_size,
                title_font_size=title_font_size
            )
        
            coverage_boxplot(
                config,
                after_boxplot_ax,
                lf2i_coverage_estimator,
                precomputed_weighters,
                lf2i_param_values,
                joint_lf2i_hits,
                r"\begin{center} \textbf{Per Gamma-Ray Source} \end{center}",
                axis_font_size=axis_font_size,
                title_font_size=title_font_size
            )
        
        # Diagnostics
        cmap = plt.cm.jet
        cmaplist = list(reversed([cmap(i) for i in range(cmap.N)])) # extract all colors from the colormap

        top_trim = 20
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist[:-top_trim], cmap.N-top_trim)

        min_color = 76
        # bounds = np.linspace(min_color+2.5, 97.5, 6, dtype=float)
        # bounds = [70] + list(bounds) + [100]
        
        # Create uneven spacing to put target_middle in the middle position
        # For example, if you want 7 colors total (6 boundaries between):
        bounds = [
            min_color,  # 76
            80,         # Values below middle
            84,
            88,  # 80 - middle value
            92,         # Values above middle
            96,
            100  # 100
        ]
        centers = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds)-1)]
        center_labels = [
            r"$\leq$" + str(int(label)) + r"\%" if label == centers[0] else 
            r"\textbf{" + str(int(label)) + r"\%}" if label == centers[3] else
            str(int(label)) + r"\%" for label in centers
        ]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N-top_trim)

        before_diagnostics_ax = fig.add_subplot(panel_b_spec[1, 0])
        after_diagnostics_ax = fig.add_subplot(panel_b_spec[1, 2])
        
        coverage_grid_resolution = 70
        margin = 0.01

        if not test_layout_only:
            plot_param_grid = pos.get_param_grid(
                config.eval_param_mins + margin,
                config.eval_param_maxes - margin,
                coverage_grid_resolution**3
            )
            
            fixed_grid = pos.get_param_grid(
                config.eval_param_mins + margin,
                config.eval_param_maxes - margin,
                coverage_grid_resolution**2,
                fixed_azimuth=0
            )
            
            hpd_coverage_probs = hpd_coverage_estimator.predict_proba(plot_param_grid.numpy())[:, 1]
            lf2i_coverage_probs = lf2i_coverage_estimator.predict_proba(plot_param_grid.numpy())[:, 1]
            # cbar_shrink = 0.78
            
            for ax, probs in zip(
                [before_diagnostics_ax, after_diagnostics_ax],
                [hpd_coverage_probs, lf2i_coverage_probs]
            ):
                mean_coverage_probs = probs.reshape(coverage_grid_resolution, coverage_grid_resolution, coverage_grid_resolution).mean(axis=2).flatten()
                the_plot = ax.scatter(fixed_grid[:, 0], fixed_grid[:, 1], c=mean_coverage_probs*100, norm=norm, cmap=cmap)
                ax.set_xlabel(r"\textbf{Log$_{10}$ Energy (GeV)}", fontsize=axis_font_size)
                ax.set_ylabel(r"\textbf{Zenith Angle (rad)}", fontsize=axis_font_size)
                ax.set_xlim(config.eval_param_mins[0].item(), config.eval_param_maxes[0].item())
                ax.set_ylim(config.eval_param_mins[1].item(), config.eval_param_maxes[1].item())
                cbar = fig.colorbar(
                    the_plot,
                    ax=ax,
                    ticks=centers,
                    format='%1.2f',
                    boundaries=bounds,
                    shrink=cbar_shrink,
                )
                # ticks = [r"$\leq$" + str(label) + r"\%" if label == min_color else str(label) + r"\%" for label in bounds]
                cbar.ax.yaxis.set_ticks(bounds, minor=True)
                cbar.ax.yaxis.set_ticklabels(center_labels)
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.yaxis.label.set_size(axis_label_size)
                cbar.ax.set_ylabel(r"\textbf{Estimated Coverage}", fontsize=axis_label_size)
                cbar.ax.tick_params(which='minor', length=2, color='black')
                cbar.ax.tick_params(which='major', length=5, color='black', labelsize=12)

        before_diagnostics_ax.set_title(r"\textbf{Individual Events}", fontsize=title_font_size)#, pad=24)
        after_diagnostics_ax.set_title(r"\textbf{Individual Events}", fontsize=title_font_size) #, pad=24)

        # fig.add_artist(lines.Line2D(
        #     [0.595, 0.615],  # x coordinates (figure-relative)
        #     [0.83, 0.83],  # y coordinates (above the top of the figure)
        #     transform=fig.transFigure,  # Use figure coordinates
        #     color='black',
        #     linestyle='dotted',
        #     linewidth=2
        # ))
    
        panel_c_spec = gridspec.GridSpecFromSubplotSpec(
            1, 3,
            subplot_spec=top_level_gridspec[2, :],
        )
        
        if not test_layout_only:
            hpd_coverage = hpd_coverage_estimator.predict_proba(
                np.array([test_obs["params"].numpy()])
            )[:, 1][0]
            
            waldo_coverage = lf2i_coverage_estimator.predict_proba(
                np.array([test_obs["params"].numpy()])
            )[:, 1][0]

            row = 1
            test_layout = False
            set_colors = [hpd_color, lf2i_color]
            sets_ax = [
                fig.add_subplot(panel_c_spec[0, c]) for c in range(3)
            ]
            for point_set, color in zip(reversed(data_set_pair), reversed(set_colors)):
                for col, pair, param_component, slice_component, axis_labels in zip(
                    [0, 1, 2],
                    [["log10_energy", "zenith"], ["azimuth", "log10_energy"], ["azimuth", "zenith"]],
                    [[0, 1], [2, 0], [2, 1]],
                    [2, 1, 0],
                    [[r"\textbf{Log$_{10}$ Energy (GeV)}", r"\textbf{Zenith Angle (Rad)}"], 
                    [r"\textbf{Azimuthal Angle (Rad)}", r"\textbf{Log$_{10}$ Energy (GeV)}"], 
                    [r"\textbf{Azimuthal Angle (Rad)}", r"\textbf{Zenith Angle (Rad)}"]]
                ):
                    ax = sets_ax[col]
                    if not test_layout:
                        _slice, _ = pos.slice_param_set(point_set, slice_component, test_obs["params"][slice_component].item())
                        slice_points = _slice[:, [param_component[0], param_component[1]]]
                        truth_slice = test_obs["params"][[param_component[0], param_component[1]]].numpy()
                        ax.scatter(x=truth_slice.reshape(-1,)[0], y=truth_slice.reshape(-1,)[1], alpha=1, color=truth_color, marker="*", s=250, zorder=10)
                        
                        plot_parameter_region_2D(
                            slice_points,
                            None, 
                            custom_ax=ax,
                            scatter=False,
                            alpha_shape=True,
                            alpha=8,
                            color=color
                        )
                    
                        
                        ax.axvline(test_obs["params"][param_component[0]].item(), color=truth_color, linestyle="dashed")
                        ax.axhline(test_obs["params"][param_component[1]].item(), color=truth_color, linestyle="dashed")
                        ax.set_xlim(config.eval_param_mins[param_component[0]].item(), config.eval_param_maxes[param_component[0]].item())
                        ax.set_ylim(config.eval_param_mins[param_component[1]].item(), config.eval_param_maxes[param_component[1]].item())
                        ax.set_xlabel(axis_labels[0], fontsize=axis_font_size)
                        ax.set_ylabel(axis_labels[1], fontsize=axis_font_size)
                        
                        if col == 1:
                            ax.set_title(r"\textbf{"  f"HPD: Actual Coverage = {hpd_coverage * 100:0.0f}\% $\mid$ FreB: Actual Coverage = {waldo_coverage *100:0.0f}\%" r"}", fontsize=title_font_size, pad = 10)
            
            handles = [lines.Line2D([0], [0], color=color) for color in set_colors]
            star_handle = lines.Line2D(
                [0], [0], marker='*', color='red', linestyle='None', markersize=10, label='Special Point'
            )
            handles = [star_handle] + handles
            labels = ["Example Event", "90\% HPD Set", "90\% FreB Set"]
            sets_ax[0].legend(
                handles,
                labels
            )
        

    return fig
