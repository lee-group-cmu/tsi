from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch.distributions import Distribution, MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.axes import Axes
import alphashape

from lf2i.simulator import Simulator
from lf2i.plot.miscellanea import PolygonPatchFixed


def plot_region_2D(
    region_points,
    truth,
    color,
    alpha,  # for alphashape
    parameter_space_bounds,
    labels,
    figsize,
    save_fig_path: Optional[str] = None,
    custom_ax: Optional[Axes] = None,
    **kwargs
) -> Optional[Axes]:  
    if custom_ax is None:
        plt.figure(figsize=figsize if figsize is not None else (10, 10))
        ax = plt.gca()
    else:
        ax = custom_ax
    labels = [r"$\theta_{{(1)}}$", r"$\theta_{{(2)}}$"] if labels is None else labels
    
    alpha_shape = alphashape.alphashape(region_points, alpha=alpha)
    patch = PolygonPatchFixed(alpha_shape, fc=to_rgba(color, 0.2), ec=to_rgba(color, 1), lw=5, zorder=10, linestyle=kwargs.pop('linestyle', '-'))#, label=region_name)
    ax.add_patch(patch)
    ax.scatter(x=truth.reshape(-1,)[0], y=truth.reshape(-1,)[1], alpha=1, marker=kwargs.get('truth_marker'), facecolor=kwargs.get('truth_color'), edgecolor=kwargs.get('truth_color'), s=1000, linewidth=2, zorder=10)
    
    ax.set_xlim(parameter_space_bounds[labels[0]]['low'], parameter_space_bounds[labels[0]]['high'])
    ax.set_ylim(parameter_space_bounds[labels[1]]['low'], parameter_space_bounds[labels[1]]['high'])
    #ax.set_xlabel(labels[0], fontsize=45)
    #ax.set_ylabel(labels[1], fontsize=45, rotation=0, labelpad=30)
    ax.set_xticks(ticks=[-10, -5, 0, 5, 10], labels=[])
    ax.set_yticks(ticks=[-10, -5, 0, 5, 10], labels=[])
    ax.set_title(kwargs.pop('title'), size=45, pad=15)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
        ax.spines[axis].set_color('black')

    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    
    if custom_ax is None:
        plt.show()
    else:
        return ax


class GaussianMean(Simulator):
    
    def __init__(
        self,
        poi_dim: int,
        prior: Distribution,
        reference: Distribution
    ) -> None:
        super().__init__(poi_dim=poi_dim, data_dim=poi_dim, batch_size=1, nuisance_dim=0)
        
        self.prior = prior
        self.reference = reference
        self.likelihood = lambda theta: MultivariateNormal(loc=theta, covariance_matrix=torch.eye(poi_dim))

    def simulate_for_test_statistic(self, size: int) -> Tuple[torch.Tensor]:
        theta = self.prior.sample(sample_shape=(size, )).reshape(size, 1)
        x = self.likelihood(theta).sample(sample_shape=(1, )).squeeze(0)
        return theta, x
    
    def simulate_for_critical_values(self, size: int) -> Tuple[torch.Tensor]:
        theta = self.reference.sample(sample_shape=(size, )).reshape(size, 1)
        x = self.likelihood(theta).sample(sample_shape=(1, )).squeeze(0)
        return theta, x
    
    def simulate_for_diagnostics(self, size: int) -> Tuple[torch.Tensor]:
        return self.simulate_for_critical_values(size)


# Custom legend handler for N divided patches
class DividedPatchHandler(HandlerPatch):
    def __init__(self, edgecolors, facecolors, num_patches, **kwargs):
        self.edgecolors = edgecolors  # List of edge colors
        self.facecolors = facecolors  # List of face colors
        self.num_patches = num_patches  # Number of patches
        super().__init__(**kwargs)  # Initialize the base class properly

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Define the space between rectangles
        gap = width * 0.15  # 15% of the total width as gap between each patch
        
        # Adjust the width of each rectangle to account for the gaps
        rect_width = (width - (self.num_patches - 1) * gap) / self.num_patches

        patches = []
        for i in range(self.num_patches):
            patch = mpatches.Rectangle(
                [xdescent + i * (rect_width + gap), ydescent],  # Position shifts for each patch
                rect_width, height, transform=trans,
                edgecolor=self.edgecolors[i], facecolor=self.facecolors[i], linewidth=2
            )
            patches.append(patch)

        return patches