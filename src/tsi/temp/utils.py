from typing import Optional, Tuple, Union, List, Sequence

from itertools import cycle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as  mpatches
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import gaussian_kde


def kdeplots2D(
    samples: Sequence[Union[np.ndarray, torch.Tensor]],
    true_theta: Optional[np.ndarray] = None,
    plot_marginals: bool = True,
    ignore_lower_than: Optional[float] = 1e-10,  
    xlim: Optional[Tuple[float]] = None,
    ylim: Optional[Tuple[float]] = None,
    names: Optional[Tuple[str]] = None,
    axis_labels: Optional[Tuple[str]] = None,
    **kwargs
) -> None:
    if not isinstance(samples, (Tuple, List)):
        samples = [samples]
    xlim, ylim = xlim or (-10, 10), ylim or (-10, 10)
    axis_labels = axis_labels or (r"$\theta_1$", r"$\theta_2$")
    names = names or [f'Distr {i}' for i in range(len(samples))]
    linestyles = cycle(['-', '--', '-.', ':'])
    marginal_colors = cycle(['blue', 'red', 'purple', 'green', 'orange'])
    fill_palettes = cycle(['Blues', 'Reds', 'Purples', 'Greens', 'Oranges'])
    grid = sns.JointGrid(x=samples[0][:, 0], y=samples[0][:, 1])
    
    for i, samples_distr_i in enumerate(samples):
        fill_palette = next(fill_palettes)
        linestyle = next(linestyles)

        kde = gaussian_kde(samples_distr_i.T)
        x = np.linspace(*xlim, 100)
        y = np.linspace(*ylim, 100)
        xx, yy = np.meshgrid(x, y)
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(grid_coords).reshape(xx.shape)

        if ignore_lower_than is not None:
            contour_levels = [lvl for lvl in np.linspace(density.min(), density.max(), 9) if lvl > ignore_lower_than]
        else:
            contour_levels = 9
        grid.ax_joint.contourf(xx, yy, density, levels=contour_levels, cmap=sns.color_palette(fill_palette, as_cmap=True), alpha=0.8, zorder=10*(i+1), locator=ticker.MaxNLocator(prune = 'lower'))
        grid.ax_joint.contour(xx, yy, density, levels=contour_levels, colors='black', linewidths=0.5, linestyles=linestyle, zorder=10*(i+1)+1)
        if plot_marginals:
            marg_c = f'tab:{next(marginal_colors)}'
            sns.kdeplot(x=samples_distr_i[:, 0], ax=grid.ax_marg_x, color=marg_c, zorder=10*(i+1), **kwargs)
            sns.kdeplot(y=samples_distr_i[:, 1], ax=grid.ax_marg_y, color=marg_c, zorder=10*(i+1), **kwargs)

    if true_theta is not None:
        grid.ax_joint.scatter(x=true_theta[0, 0].item(), y=true_theta[0, 1].item(),marker='*', s=100, color='black', zorder=100)

    grid.ax_joint.set_xlabel(axis_labels[0], fontsize=15)
    grid.ax_joint.set_ylabel(axis_labels[1], fontsize=15, rotation=0)
    grid.ax_joint.set_xlim(*xlim)
    grid.ax_joint.set_ylim(*ylim)
    handles_palettes = cycle([plt.cm.Blues(100), plt.cm.Reds(100), plt.cm.Purples(100), plt.cm.Greens(100), plt.cm.Oranges(100)])
    handles = [mpatches.Patch(facecolor=next(handles_palettes), label=names[i]) for i in range(len(samples))]
    plt.legend(handles=handles)
    plt.show()


def pinball_loss_heatmap(
    eval_grid: np.ndarray,
    pinball_loss: np.ndarray,
) -> None:
    x_bins = np.histogram_bin_edges(eval_grid[:, 0].numpy(), bins='auto')
    y_bins = np.histogram_bin_edges(eval_grid[:, 1].numpy(), bins='auto')
    binned_sum_loss, xedges, yedges = np.histogram2d(
        eval_grid[:, 0].numpy(), eval_grid[:, 1].numpy(), bins=[x_bins, y_bins], weights=pinball_loss
    )
    bin_counts, _, _ = np.histogram2d(
        eval_grid[:, 0].numpy(), eval_grid[:, 1].numpy(), bins=[x_bins, y_bins]
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_values = np.divide(binned_sum_loss, bin_counts, where=bin_counts != 0)
    heatmap_values[heatmap_values <= 0] = 1e-10

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(
        heatmap_values.T[::-1, :], 
        cmap='Blues', 
        aspect='auto', 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
        norm=LogNorm(vmin=heatmap_values.min(), vmax=heatmap_values.max())
    )
    plt.colorbar(heatmap, ax=ax, label='Pinball Loss (Log scale)')
    ax.set_xlabel(r"$\theta_1$", fontsize=15)
    ax.set_ylabel(r"$\theta_2$", fontsize=15, rotation=0)
    plt.show()
