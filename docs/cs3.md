---
layout: default
title: Supplement on Case Study III
---

# Supplement on case study III

## Experimental setup

Large astronomical flagship surveys like *Gaia* [[1](#ref1)] photometrically observe sources (e.g., stars and galaxies) and rely on other—often smaller—higher-resolution spectroscopic surveys to accurately determine the properties (i.e., "labels") of these sources. The resulting survey catalogs are then "cross-matched" to correctly match the same sources observed with different instruments, at different times, and in different wavelengths [[2](#ref2)]. However, due to observational limitations and label systematics, neither the large flagship surveys nor higher-resolution surveys uniformly observe (i.e, randomly sample) sources across the sky, resulting in labeled data that are often biased and do not match the target data of interest for inference [[3](#ref3)].

To illustrate this common challenge of *data set shift* due to selection bias, we design an inference setup faced by astronomers using cross-matched astronomical catalogs. We generate two "proof-of-concept" data settings: one where we have no selection bias and the training data match the target distribution, and one where we have a pre-trained model on "censored" data to represent selection bias. We use the censored data as a follow-up survey for our calibration set for FreB. This setup replicates a common scenario where synthetic or broad survey data are available to pre-train an initial model, possibly a large "foundation" model like SpectraFM [[4](#ref4)]. Later, new follow-up survey data targeted at regions of interest in parameter space become available, allowing us to use FreB to adjust the initial model for valid and precise parameter estimation.

In this case study, using a cross-match of stellar labels from APOGEE Data Release 17 [[5](#ref5)] and stellar spectra from Gaia Data Release 3 [[1](#ref1)], we estimate the parameter vector $$\theta = (\log g, \; T_{\text{eff}}, \; [Fe/H])$$ from data $$X$$ consisting of 110 Gaia BP/RP spectra coefficients [[6](#ref6)]. The stellar labels refer to stellar properties like effective temperature ($$T_{\text{eff}}$$), surface gravity ($$\log g$$), and metallicity ($$[Fe/H]$$).

## Data

The data consist of a set of 202,970 Gaia XP spectra cross-matched with APOGEE derived stellar labels. The Gaia XP spectra are represented as Hermite polynomial coefficients that compress the low-resolution blue photometer (BP) and red photometer (RP) spectra into a 110-dimensional vector. The stellar labels in our case study are the star's effective temperature ($$T_{\text{eff}}$$), surface gravity ($$\log g$$), and metallicity ($$[Fe/H]$$)—all of which were derived from the high-resolution APOGEE spectra.

This cross-match between the two catalogs was originally compiled by [[4](#ref4)] to train a scatter variational auto-encoder that was used to denoise and generate XP spectra. The "full" cross-match catalog contained 502,311 stars, but after implementing filters to ensure a high signal-to-noise ratio for reliable labels for training, we were left with the "good" labels set of 202,970 stars. These filter ranges for signal-to-noise ratios and measurement errors were placed on measurements including $$T_{\text{eff}}$$, $$\log g$$, metallicity, and $$BP - RP$$ (see [[4](#ref4)] for details).

To generate our two data settings, we censor in a way that reflects possible observational limitations encountered in surveys. For the no selection bias setting, we train across the full parameter space of the stars and do not censor. For the selection bias setting, we train only on a subset of stars in the train data from the no selection bias setting that are observed to have "pristine" quality labels (see [[4](#ref4)] for details)—all of which happen to be stars in the giant branch of the Kiel diagram. This division allows us to simulate selection effects commonly encountered in astronomical surveys, where certain stellar types are preferentially observed due to selection effects or systematics. We then conduct diagnostics to assess local coverage, and as an example, generate HPD and FreB sets for a Sun-like star held out from the train and calibration data. The information for this target Sun-like star are listed in Table 1. Table 2 details the data splits for both data settings.

**Table 1: Information for the target Sun-like star in Figure 3 (main text)**

| **Gaia DR3 Source ID** | **Distance** [pc] | **log g** [dex] | **T<sub>eff</sub>** [K] | **[Fe/H]** [dex] |
|---|---|---|---|---|
| 4660210013529490176 | 334.15 | -4.26 | 5772 | -0.02 |

**Table 2: Data splits for the two data settings in Section 2.3 (main text)**

| **Data Setting** | **Train** [N] | **Calibration** [N] | **Target** [N] |
|---|---|---|---|
| No selection bias | 101,481 | 40,592 | 60,889 |
| Selection bias & follow-up survey | 61,859 | 40,592 | 60,889 |

## Details on training

We estimate the posterior distribution $$\pi(\theta\mid X)$$ with masked autoregressive flows [[7](#ref7)] as implemented in the `SBI` package [[8](#ref8)] and construct 90% HPD and FreB sets in both data settings with and without selection bias. For FreB, we performed quantile regression with Python's `CatBoost` package [[9](#ref9)]. For various metallicity ranges, we provide side-by-side box plots for local coverage for both HPD and FreB sets under the two data settings in Figures 7 and 8 (main text).

---

## References

<a id="ref1"></a>
**[1]** Gaia Collaboration, Vallenari, A., Brown, A. G. A., Prusti, T., de Bruijne, J. H. J., Arenou, F., et al. (2023). Gaia Data Release 3: Summary of the content and survey properties. *Astronomy & Astrophysics*, *674*, A1. [https://doi.org/10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940)

<a id="ref2"></a>
**[2]** Salvato, M., Buchner, J., Budavári, T., Dwelly, T., Merloni, A., Brusa, M., et al. (2018). Finding counterparts for all-sky X-ray surveys with nway: a Bayesian algorithm for cross-matching multiple catalogues. *Monthly Notices of the Royal Astronomical Society*, *473*(4), 4937–4955. [https://doi.org/10.1093/mnras/stx2651](https://doi.org/10.1093/mnras/stx2651)

<a id="ref3"></a>
**[3]** Tak, H., Mandel, K., van Dyk, D. A., Kashyap, V. L., Meng, X.-L., & Siemiginowska, A. (2024). Six Maxims of Statistical Acumen for Astronomical Data Analysis. *arXiv preprint arXiv:2404.13998*. [https://doi.org/10.48550/arXiv.2404.13998](https://doi.org/10.48550/arXiv.2404.13998)

<a id="ref4"></a>
**[4]** Laroche, A., & Speagle, J. S. (2025). Closing the Stellar Labels Gap: Stellar Label independent Evidence for [α/M] Information in Gaia BP/RP Spectra. *The Astrophysical Journal*, *979*(1), 5. [https://doi.org/10.3847/1538-4357/ad9607](https://doi.org/10.3847/1538-4357/ad9607)

<a id="ref5"></a>
**[5]** Majewski, S. R., Schiavon, R. P., Frinchaboy, P. M., Allende Prieto, C., Barkhouser, R., Bizyaev, D., et al. (2017). The Apache Point Observatory Galactic Evolution Experiment (APOGEE). *The Astronomical Journal*, *154*(3), 94. [https://doi.org/10.3847/1538-3881/aa784d](https://doi.org/10.3847/1538-3881/aa784d)

<a id="ref6"></a>
**[6]** De Angeli, F., Weiler, M., Montegriffo, P., Evans, D. W., Riello, M., Andrae, R., et al. (2023). Gaia Data Release 3. Processing and validation of BP/RP low-resolution spectral data. *Astronomy and Astrophysics*, *674*, A2. [https://doi.org/10.1051/0004-6361/202243680](https://doi.org/10.1051/0004-6361/202243680)

<a id="ref7"></a>
**[7]** Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. *Advances in Neural Information Processing Systems*, *30*.

<a id="ref8"></a>
**[8]** Tejero-Cantero, A., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonçalves, P. J., Greenberg, D. S., & Macke, J. H. (2020). sbi: A toolkit for simulation-based inference. *Journal of Open Source Software*, *5*(52), 2505. [https://doi.org/10.21105/joss.02505](https://doi.org/10.21105/joss.02505)

<a id="ref9"></a>
**[9]** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, *31*.