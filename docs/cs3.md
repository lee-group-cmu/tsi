---
layout: default
---

# Supplement on Case Study III

## Experimental Setup

Large astronomical flagship surveys like *Gaia* photometrically observe sources (e.g., stars and galaxies) and rely on other—often smaller—higher-resolution spectroscopic surveys to accurately determine the properties (i.e., "labels") of these sources. The resulting survey catalogs are then "cross-matched" to correctly match the same sources observed with different instruments, at different times, and in different wavelengths. However, due to observational limitations and label systematics, neither the large flagship surveys nor higher-resolution surveys uniformly observe (i.e, randomly sample) sources across the sky, resulting in labeled data that are often biased and do not match the target data of interest for inference.

To illustrate this common challenge of *data set shift* due to selection bias, we design an inference setup faced by astronomers using cross-matched astronomical catalogs. We generate two "proof-of-concept" data settings: one where we have no selection bias and the training data match the target distribution, and one where we have a pre-trained model on "censored" data to represent selection bias. We use the censored data as a follow-up survey for our calibration set for FreB. This setup replicates a common scenario where synthetic or broad survey data are available to pre-train an initial model, possibly a large "foundation" model like SpectraFM. Later, new follow-up survey data targeted at regions of interest in parameter space become available, allowing us to use FreB to adjust the initial model for valid and precise parameter estimation.

In this case study, using a cross-match of stellar labels from APOGEE Data Release 17 and stellar spectra from Gaia Data Release 3, we estimate the parameter vector $\theta = (\log g, \; T_{\text{eff}}, \; [Fe/H])$ from data $X$ consisting of 110 Gaia BP/RP spectra coefficients. The stellar labels refer to stellar properties like effective temperature ($T_{\text{eff}}$), surface gravity ($\log g$), and metallicity ($[Fe/H]$).

## Data

The data consist of a set of 202,970 Gaia XP spectra cross-matched with APOGEE derived stellar labels. The Gaia XP spectra are represented as Hermite polynomial coefficients that compress the low-resolution blue photometer (BP) and red photometer (RP) spectra into a 110-dimensional vector. The stellar labels in our case study are the star's effective temperature ($T_{\text{eff}}$), surface gravity ($\log g$), and metallicity ($[Fe/H]$)—all of which were derived from the high-resolution APOGEE spectra.

This cross-match between the two catalogs was originally compiled by Laroche et al. (2025) to train a scatter variational auto-encoder that was used to denoise and generate XP spectra. The "full" cross-match catalog contained 502,311 stars, but after implementing filters to ensure a high signal-to-noise ratio for reliable labels for training, we were left with the "good" labels set of 202,970 stars. These filter ranges for signal-to-noise ratios and measurement errors were placed on measurements including $T_{\text{eff}}$, $\log g$, metallicity, and $BP - RP$ (see Laroche et al. 2025 for details).

To generate our two data settings, we censor in a way that reflects possible observational limitations encountered in surveys. For the no selection bias setting, we train across the full parameter space of the stars and do not censor. For the selection bias setting, we train only on a subset of stars in the train data from the no selection bias setting that are observed to have "pristine" quality labels (see Laroche et al. 2025 for details)—all of which happen to be stars in the giant branch of the Kiel diagram. This division allows us to simulate selection effects commonly encountered in astronomical surveys, where certain stellar types are preferentially observed due to selection effects or systematics. We then conduct diagnostics to assess local coverage, and as an example, generate HPD and FreB sets for a Sun-like star held out from the train and calibration data. The information for this target Sun-like star are listed in Table 1. Table 2 details the data splits for both data settings.

**Table 1: Information for the target Sun-like star in Figure 3 (main text)**

| **Gaia DR3 Source ID** | **Distance** [pc] | **log g** [dex] | **T<sub>eff</sub>** [K] | **[Fe/H]** [dex] |
|---|---|---|---|---|
| 4660210013529490176 | 334.15 | -4.26 | 5772 | -0.02 |

**Table 2: Data splits for the two data settings in Section 2.3 (main text)**

| **Data Setting** | **Train** [N] | **Calibration** [N] | **Target** [N] |
|---|---|---|---|
| No selection bias | 101,481 | 40,592 | 60,889 |
| Selection bias & follow-up survey | 61,859 | 40,592 | 60,889 |

## Details on Training

We estimate the posterior distribution $\pi(\theta\mid X)$ with masked autoregressive flows as implemented in the `SBI` package and construct 90% HPD and FreB sets in both data settings with and without selection bias. For FreB, we performed quantile regression with Python's `CatBoost` package. For various metallicity ranges, we provide side-by-side box plots for local coverage for both HPD and FreB sets under the two data settings in Figures 7 and 8 (main text).