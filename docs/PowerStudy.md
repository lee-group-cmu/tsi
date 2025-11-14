# Synthetic example: Two Moons benchmark

We provide an empirical demonstration of the implication of Theorem 6 — that the confidence sets produced by the FreB framework are, **on average over the target population**, the smallest among all locally valid confidence procedures. To emphasize this point, we compare FreB with another likelihood-free confidence procedure, **Waldo** [cite], on the classic *Two Moons* task.

For this task, data are generated from a bimodal likelihood of the form

[
p(X\mid\theta) =
\begin{pmatrix}
r\cos(\alpha) + 0.25 \
r\sin(\alpha)
\end{pmatrix}
+
\begin{pmatrix}
-\lvert \theta_1 + \theta_2 \rvert / \sqrt{2} \
(-\theta_1 + \theta_2)/\sqrt{2}
\end{pmatrix},
]

with (\alpha\sim\mathcal{U}(-\pi/2,\pi/2)) and (r\sim\mathcal{N}(0.1,0.01^2)).

We study the comparative performance of FreB and **Waldo** under two priors:

1. a uniform prior on (\Theta=[-1,1]^2), and
2. a strong Gaussian prior.

In each case, we use a flow-matching posterior estimator from the **SBI** library [cite], using default hyperparameters. We generate one observation from each mode of the likelihood:

* (X_{1,\text{target}} \sim p(X\mid\theta^\star = [-0.5,-0.5]))
* (X_{2,\text{target}} \sim p(X\mid\theta^\star = [0.5,0.5]))

As in our other examples, we assume only a single sample to infer (\theta^\star), i.e., (n=1).

**Experimental setup (common across prior settings):**

1. **Training set:**
   Construct (\mathcal{T}*{\text{train}} = {(\theta_i, X_i)}*{i=1}^B \sim p(X\mid\theta)\pi(\theta))
   with (B=50{,}000) and (\pi(\theta)=\mathcal{N}([0.5,0.5]^T, 0.5I))
   to learn (\hat{\pi}(\theta\mid X)) via flow matching.

2. **Calibration set:**
   Construct (\mathcal{T}*{\text{cal}} = {(\theta_i, X_i)}*{i=1}^{B'} \sim p(X\mid\theta) r(\theta))
   with (B'=50{,}000) and (r(\theta)=\mathcal{U}([-1,1]^2)),
   and estimate critical values via quantile regression using **CatBoost** [cite].

3. **Diagnostic set / local size estimation:**
   Construct (\mathcal{T}*{\text{diagn}} = {(\theta_i, X_i)}*{i=1}^{B''} \sim p(X\mid\theta) r(\theta)),
   with (B''=10{,}000) and (r(\theta)=\mathcal{U}([-1,1]^2)).
   A probabilistic classifier is then trained using CatBoost to estimate local set size.

---

## Case 1: Uniform prior

Figure 1 shows results under the uniform prior. The primary factor influencing the size of **Waldo**’s confidence sets is their dependence on the posterior variance. Because the posterior is bimodal, this variance is globally large, even though each mode is individually sharp. FreB, on the other hand, preserves the posterior multimodality while maintaining valid coverage on each mode without merging them.

![FreB vs Waldo with uniform prior](assets/freb_v_waldo_uniform_prior.png)

**Figure 1 — FreB confidence sets are precise (uniform prior).**
With the two-moons task, FreB yields substantially more precise confidence sets than **Waldo**, which tends to merge the two modes and become conservative as a result.

---

## Case 2: Strong prior

Figure 2 displays results under the strong prior. Both FreB and Waldo maintain local coverage despite being trained under a strongly informative prior. In this case, Waldo’s confidence sets can also be multimodal (see Panel a), but they are less constrained beyond the prior support than FreB's. This behavior may arise because Waldo’s test statistic depends more heavily on the posterior estimator’s variance than the posterior density itself.

![FreB vs Waldo with strong prior](assets/freb_v_waldo_strong_prior.png)

**Figure 2 — FreB confidence sets are precise (strong prior).**
Under the strong prior, FreB again provides tighter confidence sets than Waldo.
