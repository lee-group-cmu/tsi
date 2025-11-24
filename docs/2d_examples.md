---
layout: default
title: Supplement on 2D Synthetic Examples
---

# Supplement on 2D synthetic examples

## 2D synthetic example in Figure 4

The synthetic example in Figure 4 showcases the main properties of our framework—i.e., reliability (in the form of correct coverage) and precision (in the form of optimal constraining power)—for an inference task that was introduced in [[1](#ref1)] and has become a standard benchmark in the SBI literature. It consists of estimating the (common) mean of the components of a two-dimensional Gaussian mixture, with one component having much broader covariance:

$$
X \mid \theta \sim \frac{1}{2}\mathcal{N}(\theta, I) + \frac{1}{2}\mathcal{N}(\theta, 0.01\cdot I),
$$

where $$\theta \in \mathbb{R}^2$$ and $$n=1$$. The misspecified forward model is

$$
X \mid \theta \sim \frac{1}{2}\mathcal{N}((1-\delta)\theta, I) + \frac{1}{2}\mathcal{N}((1-\delta)\theta, 0.01\cdot I),
$$

with $$\delta=0.25$$. We proceed as follows:

1. We construct a training set $$\mathcal{T}_{\text{train}} = \{(\theta_i, X_i)\}_{i=1}^B \sim \hat{p}(X \mid \theta)\pi(\theta)$$ with $$B=50{,}000$$ and $$\pi(\theta) = \mathcal{N}(0, 2I)$$ to learn $$\hat{\pi}(\theta \mid X)$$ through a generative model. For this example, we use a flow matching posterior estimator, whose idea was first introduced in [[2](#ref2)] and then adapted for simulation-based inference settings in [[3](#ref3)]. We leverage the implementation available in the `SBI` library [[4](#ref4)], using default hyper-parameters;

2. We construct a calibration set $$\mathcal{T}_{\text{cal}} = \{(\theta_i, X_i)\}_{i=1}^{B^\prime} \sim p(X \mid \theta)r(\theta)$$ with $$B^{\prime}=30{,}000$$ and $$r(\theta) = \mathcal{N}(0, 36 I)$$ to learn a monotonic transformation $$\hat{F}(\hat{\pi}(\theta \mid X);\theta)$$ of the estimated posterior. Here, we again estimate an amortized p-value function $$P_{X \mid \theta}\left( \hat{\pi}(\theta \mid X) < \hat{\pi}(\theta_0 \mid X) \right)$$ according to Algorithm 1 by setting the number of resampled cutoffs to $$K=10$$ and using a monotone neural network whose implementation is available in our code repository;

3. We then generate one observation to represent poor alignment with the prior distribution—

$$
X_{1, \text{target}} \sim p(X \mid \theta^\star = [-8.5, -8.5]),
$$$

—and one observation to represent good alignment with the prior distribution— 

$$
X_{2, \text{target}} \sim p(X \mid \theta^\star = [0, 0])
$$

—for which we again construct HPD and FreB sets. We only observe a single sample to infer $$\theta^\star$$, i.e., $$n=1$$;

4. We check local coverage as detailed in Appendix A.8 by first generating a diagnostic set $$\mathcal{T}_{\text{diagn}} = \{(\theta_i, X_i)\}_{i=1}^{B^{\prime\prime}} \sim p(X \mid \theta)q(\theta)$$ with $$B^{\prime\prime}=20{,}000$$ and $$q(\theta) = \mathcal{U}([-10, 10] \times [-10, 10])$$ and then learning a probabilistic classifier via a tree-based gradient-boosted probabilistic classifier as implemented in the CatBoost library [[5](#ref5)].

## 2D synthetic example in Figure 8

For this example, we construct a training set $$\mathcal{T}_{\text{train}} = \{(\theta_i, X_i)\}_{i=1}^B \sim p(X \mid \theta)\pi(\theta)$$ with $$B=50{,}000$$ and $$\pi(\theta) = \mathcal{N}(0, 2I)$$ (the true data generating process). Specification of data generation and models are otherwise identical

---

## References

<a id="ref1"></a>
**[1]** Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). Sequential monte carlo without likelihoods. *Proceedings of the National Academy of Sciences*, *104*(6), 1760–1765.

<a id="ref2"></a>
**[2]** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. [https://openreview.net/forum?id=PqvMRDCJT9t](https://openreview.net/forum?id=PqvMRDCJT9t)

<a id="ref3"></a>
**[3]** Wildberger, J., Dax, M., Buchholz, S., Green, S., Macke, J. H., & Schölkopf, B. (2023). Flow Matching for Scalable Simulation-Based Inference. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, & S. Levine (Eds.), *Advances in Neural Information Processing Systems 36* (pp. 16837–16864). Curran Associates, Inc. [https://proceedings.neurips.cc/paper_files/paper/2023/file/3663ae53ec078860bb0b9c6606e092a0-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/3663ae53ec078860bb0b9c6606e092a0-Paper-Conference.pdf)

<a id="ref4"></a>
**[4]** Tejero-Cantero, A., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonçalves, P. J., Greenberg, D. S., & Macke, J. H. (2020). sbi: A toolkit for simulation-based inference. *Journal of Open Source Software*, *5*(52), 2505. [https://doi.org/10.21105/joss.02505](https://doi.org/10.21105/joss.02505)

<a id="ref5"></a>
**[5]** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, *31*.