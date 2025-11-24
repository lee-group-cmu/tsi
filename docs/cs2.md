---
layout: default
title: Supplement on Case Study II
---

# Supplement on Case Study II

## Experimental Set-up

In this case study, we identify stars along the $$(\ell, b)=(70^\circ, 30^\circ)$$ (in Galactic coordinates) line of sight because it amply includes both disk and halo components. To obtain **Model H**, we decrease the default mean and increase the default variance of the age distribution in the galactic halo component from `brutus`. To obtain **Model D**, we increase the mean of the conditional metallicity-given-age distribution according to Table 1. These hyperparameters affect the `brutus` model which is encoded as a collection of PDFs which can be evaluated directly. See Sec 2.4 in [[1](#ref1)] for further details on the `brutus` prior.

The true parameter values of the star displayed in Figure 6 (main text) are given in Table 2.

**Table 1: Galactic model hyperparameters**

| | **[Fe/H] halo mean, std. dev.** | **[Fe/H]\|Age ctr., scale** |
|---|---|---|
| Model H | -2.25, 0.5 | 0.0, 0.4 |
| Model D | -0.6, 0.2 | -0.72, 0.58 |

**Table 2: True stellar parameters for the example star in Section 2.2**

| $$T_{\text{eff}}$$ [$$10^3K$$] | $$\log g$$ [cgs] | $$[Fe/H]$$ [dex] | $$[Fe/H]_{\text{surf}}$$ [dex] | $$L$$ [$$L_{\odot}$$] |
|---|---|---|---|---|
| 7.13 | 2.85 | -2.80 | -2.76 | 7.87 |

| Dist. [kpc] | $$M_{\text{ini}}$$ [$$M_{\odot}$$] | Age [Gyr] | EEP |
|---|---|---|---|
| 0.842 | 1.30 | 2.48 | 696 |

## Data

`Brutus` [[1](#ref1)] is an open-source Python package designed to quickly estimate stellar properties, distances, and reddening based on photometric and astrometric data. It operates using grids of stellar models within a Bayesian framework that incorporates Galactic models, enabling efficient parameter estimation. Brutus accepts photometric and astrometric data as inputs, and it outputs derived stellar properties, including 3D positions, effective temperatures, distances, and extinction values. It uses empirical corrections for better accuracy and can rapidly process large data sets, making it suitable for studies requiring quick stellar parameter recovery.

Our data set consists of a large number of labeled stellar objects drawn from a prior over the log-scale surface gravity ($$\log g$$), effective temperature ($$T_{\rm{effective}}$$), surface metallicity ($$[Fe/H]_{\rm{surface}}$$), luminosity ($$L$$), distance ($$d$$), dust extinction ($$A_V$$), and differential extinction ($$R_V$$). The parameter of interest of the model is

$$\theta=(\log g, T_{\rm{effective}}, [Fe/H]_{\rm{surface}}, L)\in\mathbb{R}^5.$$

Note that we treat $$A_V$$, $$R_V$$, and $$d$$ as nuisance components, i.e. unavailable for inference in this setting. To report our inference on $$\theta$$, $$d$$ is included along with $$\theta$$ in posterior estimation as it is known to be strongly informative of the expected measurements whereas $$A_V$$ and $$R_V$$ are not estimated.

The estimated photometry for those objects are then hypothetically obtained under the Two Micron All-Sky Survey (2MASS) [[2](#ref2)] $$J$$, $$H$$, and $$K_S$$ filters and the Panoramic Survey Telescopic And Rapid Response System (PS) [[3](#ref3)] 'grizy' filters. Our likelihood processes the raw magnitudes $$m_i$$ of these filtered spectra with noiseless and noisy components. First, the magnitudes $$m_i$$ for the eight photometric bands are estimated noiselessly,

$$m_i := f_i(\theta) + \mu(d) + A_V \cdot (R_i(\theta) + R_V \cdot R_i'(\theta)),$$

where $$\mu(d)=5\log(d/10)$$ is the distance modulus in parsecs (pc) and $$f$$, $$R$$, and $$R'$$ are deterministic functions available in the `brutus` library parameterizing photometry generation and reddening. Then some random noise is added to the flux scale,

$$$$F_i \sim \mathcal{N}\left(\exp\left(-\frac{2}{5} m_i\right), 0.2\right).$$$$

Lastly, the final noised magnitudes $$M_i=-\frac{5}{2} \log(F_i)$$ are decomposed into relative and absolute components, i.e.

$$$$X = (\tilde{M}_1, \tilde{M}_2, \ldots, \tilde{M}_8, M)\in\mathbb{R}^9,$$$$

where $$M$$ is such that $$\tilde{M}_i = M_i / M$$, to help with the stability of network training.

## Details on Training

We trained a posterior estimator $$\hat{\pi}(\theta\mid X)$$ using a normalizing flow model with the masked autoregressive flow [[4](#ref4)] architecture as implemented in the `SBI` library [[5](#ref5)] with 50 hidden features over five hidden layers. Quantile regression for calibration of the FreB method was implemented using Python's `CatBoost` library [[6](#ref6)]. We used $$B=500{,}000$$ for training, $$B'=500{,}000$$ for calibration, and $$B''=25{,}000$$ for evaluation.

---

## References

<a id="ref1"></a>
**[1]** Speagle, J. S., Zucker, C., Beane, A., Cargile, P. A., Dotter, A., Finkbeiner, D. P., et al. (2025). Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS. *arXiv preprint arXiv:2503.02227*. [https://doi.org/10.48550/arXiv.2503.02227](https://doi.org/10.48550/arXiv.2503.02227)

<a id="ref2"></a>
**[2]** Skrutskie, M. F., Cutri, R. M., Stiening, R., Weinberg, M. D., Schneider, S., Carpenter, J. M., et al. (2006). The Two Micron All Sky Survey (2MASS). *The Astronomical Journal*, *131*(2), 1163–1183. [https://doi.org/10.1086/498708](https://doi.org/10.1086/498708)

<a id="ref3"></a>
**[3]** Chambers, K. C., Magnier, E. A., Metcalfe, N., Flewelling, H. A., Huber, M. E., Waters, C. Z., et al. (2016). The Pan-STARRS1 Surveys. *arXiv preprint arXiv:1612.05560*. [https://doi.org/10.48550/arXiv.1612.05560](https://doi.org/10.48550/arXiv.1612.05560)

<a id="ref4"></a>
**[4]** Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. *Advances in Neural Information Processing Systems*, *30*.

<a id="ref5"></a>
**[5]** Tejero-Cantero, A., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonçalves, P. J., Greenberg, D. S., & Macke, J. H. (2020). sbi: A toolkit for simulation-based inference. *Journal of Open Source Software*, *5*(52), 2505. [https://doi.org/10.21105/joss.02505](https://doi.org/10.21105/joss.02505)

<a id="ref6"></a>
**[6]** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, *31*.