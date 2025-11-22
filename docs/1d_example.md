---
layout: default
---

# Supplement on 1D synthetic example

## 1D synthetic example in Figure~1c-d

The synthetic example in Figure~1c-d leverages a simple setting to showcase the main components of our framework for trustworthy scientific inference. We assume that all data are generated from an (unknown) Gaussian likelihood $$p(X \mid \theta) = \mathcal{N}(\theta, 1)$$ and proceed as follows:

1. We construct a training set $$\mathcal{T}_{\text{train}} = \{(\theta_i, X_i)\}_{i=1}^B \sim p(X \mid \theta)\pi(\theta)$$ with $$B=100{,}000$$ and $$\pi(\theta) = \mathcal{N}(0, 1)$$ to learn $$\hat{\pi}(\theta \mid X)$$ through a generative model. For this example, we use a simple masked autoregressive flow \citepsupp{papamakarios_fast_2016,lueckmann_flexible_2017} as implemented in the \texttt{SBI} library \citepsupp{tejero-cantero_sbi_2020}, using default hyper-parameters;
2. We construct a calibration set $$\mathcal{T}_{\text{cal}} = \{(\theta_i, X_i)\}_{i=1}^{B^\prime} \sim p(X \mid \theta)r(\theta)$$ with $$B^{\prime}=50{,}000$$ and $$r(\theta) = \mathcal{U}(-10, 10)$$ to learn a monotonic transformation $$\hat{F}(\hat{\pi}(\theta \mid X);\theta)$$ of the estimated posterior. Here, we estimate an amortized p-value function $$\P_{X \mid \theta}\left( \hat{\pi}(\theta \mid X) < \hat{\pi}(\theta_0 \mid X) \right)$$ according to Algorithm~\ref{algo:rejection_prob0} by setting the number of resampled cutoffs to $$K=10$$ and leveraging a tree-based gradient-boosted probabilistic classifier as implemented in the \texttt{CatBoost} library \citepsupp{prokhorenkova_catboost_2018}. We only optimize the number of trees and the maximum depth, which are finally set to $$1000$$ and $$9$$, respectively;
3. We generate $$X_{\text{target}} \sim p(X \mid \theta^\star = 4)$$ and construct an HPD set according to Equation~\ref{eq:hpd_def} and a FreB set as shown in Appendix~\ref{sec:p-values}. Note that we only observe a single sample to infer $$\theta^\star$$, i.e., $$n=1$$;
4. Finally, we check local coverage as detailed in Appendix~\ref{sec:diagnostics} by first generating a diagnostic set $$\mathcal{T}_{\text{diagn}} = \{(\theta_i, X_i)\}_{i=1}^{B^{\prime\prime}} \sim p(X \mid \theta)r(\theta)$$ with $$B^{\prime\prime}=50{,}000$$ and $$r(\theta) = \mathcal{U}((-10, 10))$$ and then learning a probabilistic classifier via a univariate Generalized Additive Model (GAM) with thin plate splines as implemented in the \texttt{MGCV} library in \texttt{R} \citesupp{wood_package_2015}.