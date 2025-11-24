---
layout: default
title: Supplement on 1D Synthetic Example
---

# Supplement on 1D synthetic example

## 1D synthetic example in Figure 3

The synthetic example in Figure 3 leverages a simple setting to showcase the main components of our framework for trustworthy scientific inference. We assume that all data are generated from an (unknown) Gaussian likelihood $p(X \mid \theta) = \mathcal{N}(\theta, 1)$ and proceed as follows:

1. We construct a training set $\mathcal{T}_{\text{train}} = \{(\theta_i, X_i)\}_{i=1}^B \sim p(X \mid \theta)\pi(\theta)$ with $B=100{,}000$ and $\pi(\theta) = \mathcal{N}(0, 1)$ to learn $\hat{\pi}(\theta \mid X)$ through a generative model. For this example, we use a simple masked autoregressive flow [[1](#ref1), [2](#ref2)] as implemented in the `SBI` library [[3](#ref3)], using default hyper-parameters;

2. We construct a calibration set $\mathcal{T}_{\text{cal}} = \{(\theta_i, X_i)\}_{i=1}^{B^\prime} \sim p(X \mid \theta)r(\theta)$ with $B^{\prime}=50{,}000$ and $r(\theta) = \mathcal{U}(-10, 10)$ to learn a monotonic transformation $\hat{F}(\hat{\pi}(\theta \mid X);\theta)$ of the estimated posterior. Here, we estimate an amortized p-value function $P_{X \mid \theta}\left( \hat{\pi}(\theta \mid X) < \hat{\pi}(\theta_0 \mid X) \right)$ by setting the number of resampled cutoffs to $K=10$ and leveraging a tree-based gradient-boosted probabilistic classifier as implemented in the `CatBoost` library [[4](#ref4)]. We only optimize the number of trees and the maximum depth, which are finally set to $1000$ and $9$, respectively;

3. We generate $X_{\text{target}} \sim p(X \mid \theta^\star = 4)$ and construct an HPD set and a FreB set. Note that we only observe a single sample to infer $\theta^\star$, i.e., $n=1$;

4. Finally, we check local coverage by first generating a diagnostic set $\mathcal{T}_{\text{diagn}} = \{(\theta_i, X_i)\}_{i=1}^{B^{\prime\prime}} \sim p(X \mid \theta)r(\theta)$ with $B^{\prime\prime}=50{,}000$ and $r(\theta) = \mathcal{U}((-10, 10))$ and then learning a probabilistic classifier via a univariate Generalized Additive Model (GAM) with thin plate splines as implemented in the `MGCV` library in `R` [[5](#ref5)].

---

## References

<a id="ref1"></a>
**[1]** Papamakarios, G., & Murray, I. (2016). Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems 29* (pp. 1028–1036). Curran Associates, Inc.

<a id="ref2"></a>
**[2]** Lueckmann, J.-M., Goncalves, P. J., Bassetto, G., Öcal, K., Nonnenmacher, M., & Macke, J. H. (2017). Flexible statistical inference for mechanistic models of neural dynamics. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems 30* (pp. 1289–1299). Curran Associates, Inc. [http://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics.pdf](http://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics.pdf)

<a id="ref3"></a>
**[3]** Tejero-Cantero, A., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonçalves, P. J., Greenberg, D. S., & Macke, J. H. (2020). sbi: A toolkit for simulation-based inference. *Journal of Open Source Software*, *5*(52), 2505. [https://doi.org/10.21105/joss.02505](https://doi.org/10.21105/joss.02505)

<a id="ref4"></a>
**[4]** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, *31*.

<a id="ref5"></a>
**[5]** Wood, S., & Wood, M. S. (2015). Package 'mgcv'. *R Package Version*, *1*(29), 729.