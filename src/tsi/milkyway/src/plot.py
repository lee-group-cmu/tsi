import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import matplotlib.lines as mlines

from lf2i.plot.miscellanea import PolygonPatchFixed

from matplotlib.colors import to_rgba
import alphashape

def plot_kiel_diagram(
        target_x, # teff
        target_y, # logg
        prior_x=None,
        prior_y=None,
        ax=None,
        title=r"\textbf{No Selection Bias}",
        xlabel=r"\textbf{T}$\mathbf{_{\mathrm{eff}}}$ \textbf{(K)}",
        ylabel=r"\textbf{log }$\mathbf{g}$ \textbf{(dex)}",
        target_color="gray",
        target_alpha=0.25,
        target_label=r"$p_{\mathrm{target}}(\theta)$",
        target_label_linewidth=4,
        plot_prior=False,
        prior_filter=True,
        prior_color="blue",
        prior_alpha=0.25,
        prior_label=r"$\pi(\theta)$",
        prior_label_linewidth=4,
        separation_line_color="black",
        separation_line_style="--",
        separation_line_linewidth=2,
        kde_levels=25,
        kde_thresh=0.01,
        kde_linewidths=2,
        plot_reference_star=True,
        reference_star_teff=4750,
        reference_star_logg=4.5,
        reference_star_marker="*",
        reference_star_color="red",
        reference_star_size=400,
        reference_star_edgecolor="black",
        reference_star_label="Reference Star",
        reference_star_legend_size=15,
        crosshair_style=":",
        crosshair_color="red",
        crosshair_linewidth=2,
        kiel_xlim_low=2750,
        kiel_xlim_high=6750,
        kiel_ylim_low=-0.5,
        kiel_ylim_high=5,
        title_fontsize=36,
        label_fontsize=20,
        tick_labelsize=16,
        legend_fontsize=16,
        legend_location="upper left",
        figsize_x=15,
        figsize_y=12
):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Target Distribution (full data region/parameter space)
    sns.kdeplot(x=target_x,
                y=target_y,
                fill = True,
                color=target_color,
                alpha=target_alpha,
                thresh=kde_thresh,
                levels=kde_levels,
                linewidths=kde_linewidths,
                zorder=1)
    
    # Prior Distribution (train data)
    if plot_prior:
        if prior_filter:
            separation_logg = 5.9 - 0.4 * prior_x / 1000
            mask = prior_y < separation_logg
            prior_x = prior_x[mask]
            prior_y = prior_y[mask]

        if len(prior_x) > 0:
            sns.kdeplot(x=prior_x,
                        y=prior_y,
                        fill = True,
                        color=prior_color,
                        alpha=prior_alpha,
                        thresh=kde_thresh,
                        levels=kde_levels,
                        linewidths=kde_linewidths,
                        zorder=2)
            
    # Separation line
    x_line = np.array([2750, 6750])
    y_line = 5.9 - 0.4 * x_line / 1000
    ax.plot(x_line, 
            y_line, 
            color=separation_line_color, 
            linestyle=separation_line_style, 
            linewidth=separation_line_linewidth,
            zorder=3)

    # Reference Star Marker and Crosshair
    if plot_reference_star:
        ax.axhline(reference_star_logg,
                   ls=crosshair_style,
                   color=crosshair_color,
                   lw=crosshair_linewidth,
                   zorder=4)
        ax.axvline(reference_star_teff,
                   ls=crosshair_style,
                   color=crosshair_color,
                   lw=crosshair_linewidth,
                   zorder=4)
        ax.scatter([reference_star_teff],
                   [reference_star_logg],
                   marker=reference_star_marker,
                   s=reference_star_size,
                   color=reference_star_color,
                   edgecolor=reference_star_edgecolor,
                   zorder=5)
        
    # Axes styling
    ax.set_xlim(kiel_xlim_low, kiel_xlim_high)
    ax.invert_xaxis()
    ax.set_ylim(kiel_ylim_low, kiel_ylim_high)
    ax.invert_yaxis()
    ax.tick_params(labelsize=tick_labelsize)

    # Title and Labels
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Legend
    handles = [mlines.Line2D([], [], 
                             color=target_color, 
                             linewidth=target_label_linewidth, 
                             label=target_label)]

    if plot_prior:
        handles.insert(0, mlines.Line2D([], [], 
                                        color=prior_color, 
                                        linewidth=prior_label_linewidth, 
                                        label=prior_label))

    if plot_reference_star:
        handles.insert(len(handles), mlines.Line2D([], [], 
                                        marker=reference_star_marker, 
                                        markersize=reference_star_legend_size, 
                                        markerfacecolor=reference_star_color, 
                                        markeredgecolor=reference_star_edgecolor, 
                                        linestyle="None", 
                                        label=reference_star_label))
        
    ax.legend(handles=handles, fontsize=legend_fontsize, loc=legend_location)

def plot_star_spectrum(
        bprp_coeffs,
        xp_design_matrices,
        xp_merge,
        xp_sampling_grid,
        reference_star_logg,
        reference_star_teff,
        reference_star_feh,
        ax=None,
        title=r"\textbf{Spectrum of Sun-like Star}",
        xlabel=r"\textbf{Wavelength (nm)}",
        ylabel=r"\textbf{Flux (10$^{-16}$ W nm$^{-1}$ m$^{-2}$)}",
        spectra_color="red",
        spectra_linewidth=4,
        reference_star_marker="*",
        reference_star_color="red",
        reference_star_edgecolor="black",
        reference_star_label="Reference Star",
        reference_star_legend_size=15,
        title_fontsize=36,
        label_fontsize=20,
        tick_labelsize=16,
        legend_fontsize=16,
        legend_location="upper right",
        figsize_x=15,
        figsize_y=12
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
                               
    # reconstruct physical flux
    bp_spec = bprp_coeffs["bp"].dot(xp_design_matrices["bp"])
    rp_spec = bprp_coeffs["rp"].dot(xp_design_matrices["rp"])
    flux = bp_spec * xp_merge["bp"] + rp_spec * xp_merge["rp"]

    # spectrum
    sns.lineplot(x=xp_sampling_grid,
                 y=flux/1e-16,
                 color=spectra_color,
                 linewidth=spectra_linewidth,
                 ax=ax,
                 zorder=1)
    
    # titles and labels
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_labelsize)

    # reference star information
    reference_star_information_label = rf"""
{reference_star_label}
$\log g$ = {reference_star_logg:.2f}
T$_{{\mathrm{{eff}}}}$ = {reference_star_teff:.0f} K
[Fe/H] = {reference_star_feh:.2f}
"""
    
    handle = mlines.Line2D(
        [], [], linestyle="None", marker=reference_star_marker, markersize=reference_star_legend_size,
        markerfacecolor=reference_star_color, markeredgecolor=reference_star_edgecolor,
        label=reference_star_information_label
    )

    ax.legend(handles=[handle], fontsize=legend_fontsize, loc=legend_location)

def plot_hpd_freb_region(
        region_data,
        target_x, # teff
        target_y, # logg
        prior_x=None,
        prior_y=None,
        ax=None,
        title=r"90\% HPD Set",
        xlabel=r"\textbf{T}$\mathbf{_{\mathrm{eff}}}$ \textbf{(K)}",
        ylabel=r"\textbf{log }$\mathbf{g}$ \textbf{(dex)}",
        region_label=r"90\% HPD Set",
        region_color="orange",
        region_edge_alpha=1.0,
        region_fill_alpha=0.25,
        region_alpha_shape_param=0.1,
        region_linewidth=2,
        region_linestyle="-",
        region_plot_legend=True,
        target_color="gray",
        target_alpha=0.25,
        plot_prior=False,
        prior_filter=True,
        prior_color="blue",
        prior_alpha=0.25,
        kde_levels=25,
        kde_thresh=0.01,
        kde_linewidths=2,
        plot_reference_star=True,
        reference_star_teff=4750,
        reference_star_logg=4.5,
        reference_star_marker="*",
        reference_star_color="red",
        reference_star_size=400,
        reference_star_edgecolor="black",
        crosshair_style=":",
        crosshair_color="red",
        crosshair_linewidth=2,
        kiel_xlim_low=2750,
        kiel_xlim_high=6750,
        kiel_ylim_low=-0.5,
        kiel_ylim_high=5,
        title_fontsize=36,
        label_fontsize=20,
        tick_labelsize=16,
        legend_fontsize=16,
        legend_location="upper left",
        figsize_x=15,
        figsize_y=12
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Target Distribution (full data region/parameter space)
    sns.kdeplot(x=target_x,
                y=target_y,
                fill = True,
                color=target_color,
                alpha=target_alpha,
                thresh=kde_thresh,
                levels=kde_levels,
                linewidths=kde_linewidths,
                zorder=1)
    
    # Prior Distribution (train data)
    if plot_prior:
        if prior_filter:
            separation_logg = 5.9 - 0.4 * prior_x / 1000
            mask = prior_y < separation_logg
            prior_x = prior_x[mask]
            prior_y = prior_y[mask]

        if len(prior_x) > 0:
            sns.kdeplot(x=prior_x,
                        y=prior_y,
                        fill = True,
                        color=prior_color,
                        alpha=prior_alpha,
                        thresh=kde_thresh,
                        levels=kde_levels,
                        linewidths=kde_linewidths,
                        zorder=2)
            
    # 90% HPD or FreB Region
    shape = alphashape.alphashape(region_data, alpha=region_alpha_shape_param)
    patch = PolygonPatchFixed(shape,
                              fc=to_rgba(region_color, region_fill_alpha),
                              ec=to_rgba(region_color, region_edge_alpha),
                              lw=region_linewidth, linestyle=region_linestyle)
    ax.add_patch(patch)

    # Reference Star Marker and Crosshair
    if plot_reference_star:
        ax.axhline(reference_star_logg,
                   ls=crosshair_style,
                   color=crosshair_color,
                   lw=crosshair_linewidth,
                   zorder=4)
        ax.axvline(reference_star_teff,
                   ls=crosshair_style,
                   color=crosshair_color,
                   lw=crosshair_linewidth,
                   zorder=4)
        ax.scatter([reference_star_teff],
                   [reference_star_logg],
                   marker=reference_star_marker,
                   s=reference_star_size,
                   color=reference_star_color,
                   edgecolor=reference_star_edgecolor,
                   zorder=5)
        
    # Axes styling
    ax.set_xlim(kiel_xlim_low, kiel_xlim_high)
    ax.invert_xaxis()
    ax.set_ylim(kiel_ylim_low, kiel_ylim_high)
    ax.invert_yaxis()
    
    # Title and Labels
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_labelsize)
    
    # Legend
    if region_plot_legend:
        handle = mlines.Line2D([], [], color=region_color, linewidth=region_linewidth, label=region_label)

        ax.legend(handles=[handle], fontsize=legend_fontsize, loc=legend_location)
                              

def plot_local_diagnostics(
        coverage_data,
        target_x, # teff
        target_y, # logg
        confidence_level=0.90,
        prior_x=None,
        prior_y=None,
        ax=None,
        title=r"\textbf{90\% HPD Local Coverage}",
        xlabel=r"\textbf{T}$\mathbf{_{\mathrm{eff}}}$ \textbf{(K)}",
        ylabel=r"\textbf{log }$\mathbf{g}$ \textbf{(dex)}",
        cmap_top_trim=20,
        cmap_bounds=[76, 80, 84, 88, 92, 96, 100],
        cmap_tick_color="black",
        cmap_extend="both",
        cmap_extend_frac=0.05,
        cmap_format="%1.1f",
        target_color="gray",
        target_alpha=0.25,
        plot_prior=False,
        prior_filter=True,
        prior_color="blue",
        prior_alpha=0.25,
        kde_levels=25,
        kde_thresh=0.01,
        kde_linewidths=2,
        metal_tracks = False,
        metal_tracks_catalog_data = None,
        metal_track_color="purple",
        metal_track_size=100,
        metal_track_linewidth=4,
        metal_track_label_flag=True,
        metal_track_label=-1,
        metal_track_legend_size=4,
        plot_reference_star=True,
        reference_star_teff=4750,
        reference_star_logg=4.5,
        reference_star_marker="*",
        reference_star_color="red",
        reference_star_size=400,
        reference_star_edgecolor="black",
        crosshair_style=":",
        crosshair_color="red",
        crosshair_linewidth=2,
        kiel_xlim_low=2750,
        kiel_xlim_high=6750,
        kiel_ylim_low=-0.5,
        kiel_ylim_high=5,
        title_fontsize=36,
        label_fontsize=20,
        tick_labelsize=16,
        legend_fontsize=16,
        legend_location="upper left",
        figsize_x=15,
        figsize_y=12  
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Target Distribution (full data region/parameter space)
    sns.kdeplot(x=target_x,
                y=target_y,
                fill = True,
                color=target_color,
                alpha=target_alpha,
                thresh=kde_thresh,
                levels=kde_levels,
                linewidths=kde_linewidths,
                zorder=1)
    
    # Prior Distribution (train data)
    if plot_prior:
        if prior_filter:
            separation_logg = 5.9 - 0.4 * prior_x / 1000
            mask = prior_y < separation_logg
            prior_x = prior_x[mask]
            prior_y = prior_y[mask]

        if len(prior_x) > 0:
            sns.kdeplot(x=prior_x,
                        y=prior_y,
                        fill = True,
                        color=prior_color,
                        alpha=prior_alpha,
                        thresh=kde_thresh,
                        levels=kde_levels,
                        linewidths=kde_linewidths,
                        zorder=2)
    
    # heatmap setup
    x_bins = np.histogram_bin_edges(coverage_data["parameter2_teff"], bins='auto')
    y_bins = np.histogram_bin_edges(coverage_data["parameter1_logg"], bins='auto')
    bsum, xedges, yedges = np.histogram2d(
        coverage_data["parameter2_teff"],
        coverage_data["parameter1_logg"],
        bins=[x_bins, y_bins],
        weights=np.round(coverage_data["mean_proba"] * 100, 2)
    )
    counts, _, _ = np.histogram2d(
        coverage_data["parameter2_teff"],
        coverage_data["parameter1_logg"],
        bins=[x_bins, y_bins]
    )
    heatmap_values = bsum / counts

    # colormap
    base = plt.cm.jet
    cmaplist = [base(i) for i in range(base.N)]
    # reversed & drop top_trim entries
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap",
                                                        list(reversed(cmaplist))[:-cmap_top_trim],
                                                        N=base.N - cmap_top_trim
                                                        )
    
    # colormap setup
    bounds = cmap_bounds
    centers = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds)-1)]
    center_labels = [
        r"$\leq$" + str(int(label)) + r"\%" if label == centers[0] else 
        r"\textbf{" + str(int(label)) + r"\%}" if label == centers[3] else
        str(int(label)) + r"\%" for label in centers
    ]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # heatmap
    ax.imshow(heatmap_values.T[::-1, :],
              cmap=cmap,
              norm=norm,
              aspect="auto",
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    # metal tracks
    if metal_tracks == True and metal_tracks_catalog_data is not None:
        sns.scatterplot(x=metal_tracks_catalog_data["TEFF"], 
                        y = metal_tracks_catalog_data["LOGG"], 
                        color=metal_track_color, ax=ax, s=metal_track_size,
                        legend=False)
        sns.lineplot(x=metal_tracks_catalog_data["TEFF"], 
                     y =metal_tracks_catalog_data["LOGG"], 
                     color=metal_track_color, ax=ax,
                     sort = False, linewidth=metal_track_linewidth,
                     legend=False)

   # Reference Star Marker and Crosshair
    if plot_reference_star:
        ax.axhline(reference_star_logg,
                   ls=crosshair_style,
                   color=crosshair_color,
                   lw=crosshair_linewidth,
                   zorder=4)
        ax.axvline(reference_star_teff,
                   ls=crosshair_style,
                   color=crosshair_color,
                   lw=crosshair_linewidth,
                   zorder=4)
        ax.scatter([reference_star_teff],
                   [reference_star_logg],
                   marker=reference_star_marker,
                   s=reference_star_size,
                   color=reference_star_color,
                   edgecolor=reference_star_edgecolor,
                   zorder=5)
        
    # Axes styling
    ax.set_xlim(kiel_xlim_low, kiel_xlim_high)
    ax.invert_xaxis()
    ax.set_ylim(kiel_ylim_low, kiel_ylim_high)
    ax.invert_yaxis()
    ax.tick_params(labelsize=tick_labelsize)

    # Title and Labels
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
        
    if metal_track_label_flag==True:
        track_handle = mlines.Line2D([], [], color=metal_track_color,
                                marker='o', linestyle='-',
                                markersize=metal_track_legend_size, linewidth=metal_track_linewidth, label=f"[Fe/H] = {metal_track_label}")
        ax.legend(handles=[track_handle], loc="upper left", fontsize=legend_fontsize)