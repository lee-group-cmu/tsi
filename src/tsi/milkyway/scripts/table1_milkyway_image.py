import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances

# Add source directory
src_subdir = "../src/"
if src_subdir not in sys.path:
    sys.path.insert(1, src_subdir)

# Imports from your project
from data import read_catalog, generate_splits, process_all_splits
from utils import setup_paths
from vsi import VSI

from astropy import units as u
from astropy.coordinates import SkyCoord
from mw_plot import MWSkyMap

data_paths = setup_paths(notebook=True)
catalog_path = os.path.join(data_paths["data"], "xp_apogee_cat.h5")
catalog = read_catalog(catalog_path, filter="full", normalize=False, labels=["GLON", "GLAT", "TEFF", "LOGG", "FE_H"])

catalog_pristine_path = os.path.join(data_paths["data"], "xp_apogee_cat.h5")
catalog_pristine = read_catalog(catalog_path, filter="pristine", normalize=False, labels=["GLON", "GLAT"])

# Sun-like stars
SUNLIKE_PARAMS = [4.43775, 5772, 0.0]

catalog_features = np.vstack([
    catalog["LOGG"],
    catalog["TEFF"],
    catalog["FE_H"]
]).T

N_HOLDOUT_sub = 1

sun_vec = np.array([SUNLIKE_PARAMS])  # LOGG, TEFF, FE_H

distances = pairwise_distances(catalog_features, sun_vec).flatten()

sunlike_indices = np.argsort(distances)[:N_HOLDOUT_sub]

sunlike_star_info = catalog[sunlike_indices]

print(sunlike_star_info["GLON", "GLAT", "LOGG", "TEFF", "FE_H",])

glon_sun = sunlike_star_info["GLON"]
glat_sun = sunlike_star_info["GLAT"]

galactic_coords_sun = SkyCoord(l=glon_sun * u.deg, b=glat_sun * u.deg, frame='galactic')

equatorial_coords_sun = galactic_coords_sun.icrs
ra_sun = equatorial_coords_sun.ra   # Right ascension
dec_sun = equatorial_coords_sun.dec # Declination

glon = catalog["GLON"]  # in degrees
glat = catalog["GLAT"]  # in degrees

# Create an Astropy SkyCoord object using the galactic coordinates
galactic_coords = SkyCoord(l=glon * u.deg, b=glat * u.deg, frame='galactic')

# Convert to ICRS (i.e., right ascension and declination) for proper alignment on the background
equatorial_coords = galactic_coords.icrs
ra = equatorial_coords.ra   # Right ascension
dec = equatorial_coords.dec # Declination

# PRISTINE DATA
np.random.seed(42)  # Set seed for reproducibility (optional)
n_pristine = len(catalog_pristine)
sample_indices = np.random.choice(n_pristine, size=int(0.2 * n_pristine), replace=False)

# Sample the catalog
catalog_pristine_sampled = catalog_pristine[sample_indices]

glon_prist = catalog_pristine_sampled["GLON"]  # in degrees
glat_prist = catalog_pristine_sampled["GLAT"]  # in degrees

# Create an Astropy SkyCoord object using the galactic coordinates
galactic_coords_pristine = SkyCoord(l=glon_prist * u.deg, b=glat_prist * u.deg, frame='galactic')

# Convert to ICRS (i.e., right ascension and declination) for proper alignment on the background
equatorial_coords_pristine = galactic_coords_pristine.icrs
ra_pristine = equatorial_coords_pristine.ra   # Right ascension
dec_pristine = equatorial_coords_pristine.dec # Declination

# Milkway map
mw_map = MWSkyMap(projection="aitoff", grayscale=False, grid="galactic", background="infrared")

mw_map.scatter(ra, dec, c="orange", s=0.03, alpha=0.95)

mw_map.scatter(ra_pristine, dec_pristine, c="blue", s=0.05, alpha=0.8)

mw_map.scatter(ra_sun, dec_sun, c="yellow", s=50, alpha=1, edgecolor="black")

ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])

# plt.show()
save_path = os.path.join(data_paths["figures"], "table1_milkyway_observations.png")
plt.savefig(save_path, bbox_inches='tight', dpi=600)


