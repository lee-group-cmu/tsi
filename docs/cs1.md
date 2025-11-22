---
layout: default
---

# Supplement on Case Study I

## Experimental Setup

Training and target data sets for this case study have been created as a proof-of-concept. We are not simulating a full-fledged ground-based gamma-ray experiment as it would not change our main results and the applicability of FreB.

We base the parameter distributions of the simulated air showers on the following three gamma-ray sources:

- **Crab Nebula:** A pulsar-wind nebula emitting the brightest and stable TeV signal in the northern hemisphere sky, for the past 970 years.
- **Markarian 421 (Mrk421):** A blazar located about 397 million light years from earth. Blazars and other active galactic nuclei emit intense electromagnetic radiation, facilitating the discovery of otherwise faint distant galaxies.
- **Dark Matter (DM) annihilation:** Similar to matter-antimatter annihilation, some theories of dark matter propose an annihilation mechanism for dark matter particles, which emit gamma rays following a certain energy spectrum. Gamma-ray measurements from regions of space thought to contain dark matter (e.g. around galaxies) can put these theories to the test.

Note that Mrk421 is a point source much like the Crab Nebula, but the DM Annihilation source is a theorized mechanism that could happen anywhere in the cosmos. As such, we treat DM as a diffuse source of gamma ray events that hit the Earth from all directions. We only consider the zenith component of the point source trajectories, azimuth distributed uniformly, for direct comparison between sources.

The zenith distribution along the Crab and Mrk 421 trajectories relative to the zenith distribution in the pre-simulated CORSIKA data is used to assign weights to individual gamma ray events. All trajectory calculations are performed using `astropy`. Each source's theoretical energy spectrum assign weights to individual gamma ray events in the pre-simulated set. For the Crab Nebula, we use the log-parabola fit proposed by Aleksić et al. (2015). For Mrk 421, we perform a custom fit to observational data that accounts for attenuation of gamma-ray flux due to extragalactic background light (EBL). For the DM source, we use `gammapy` to generate the dark-matter annihilation spectrum for very heavy DM particles (100 TeV). We do not attenuate this spectrum using EBL.

High-energy gamma rays must be discriminated from the much more abundant charged cosmic rays (protons and heavier hadrons) hitting the Earth atmosphere. Because hadrons also produce an atmospheric shower observable by ground detectors, a preliminary step in reconstructing gamma-ray events from ground detector data is to first determine if an observed shower is a gamma ray or a hadron. We do not perform this initial classification step in this case study and focus only on the reconstruction of gamma-ray events. This assumption does not affect the results obtained in this work since we are concentrating on individual events rather than attempting a full reconstruction.

## Data

Our data set consists of a large number of labeled gamma-ray events $(E_i, Z_i, A_i, X_i)$. For each event $i$:

1. $E_i$ is the energy of the original gamma ray in GeV
2. $Z_i$ is the zenith/polar angle
3. $A_i$ is the azimuthal angle
4. $X_i$ represents the data collected by ground detectors by the resulting atmospheric shower

Our data come from the CORSIKA simulator. We make three splits from the data:

1. **Training set** ($B=1{,}072{,}821$) used to train our posterior estimator $p(\theta_i \mid X_i)$
2. **Calibration set** ($B'=98{,}765$) used to train our FreB quantile regression
3. **Diagnostic set** ($B''=42{,}270$) used to evaluate the performance of our confidence set procedures

For observed detector data $X_i$, we assume (an unrealistic, but practical for this work) full ground coverage in a 4km × 4km square, where each detector is 2m × 2m. For a given shower, we assume that each detector is capable of recording the identity and timing of every secondary particle that passes through it. The number of secondary particles per shower can range from less than 10 for low-energy gamma rays to up to 100 million for very high-energy gamma rays. Although many types of secondary particles may appear in an atmospheric shower, we consider only two broad groups (photons/electrons/positrons versus everything else) for ease of analysis.

We remove all gamma-ray events in all data splits where less than 10 ground detectors recorded secondary particle hits. We weight our filtered training data to resemble the Crab Nebula in terms of its energy spectrum. We also weight the training data to resemble a fixed reference distribution in zenith. This reference distribution is a combination of a uniform distribution over the sphere and atmospheric effects at high zenith angles. We assume that $p(X_i \mid \theta_i)$ exhibits azimuthal symmetry.

We place our observer at 19 degrees north for definitiveness. This latitude corresponds to the current location of the operational HAWC observatory.

## Details on Training

We train our posterior estimator using a the flow matching architecture, a diffusion-based model with training-based acceleration, to obtain an estimate of the posterior $\hat p(\theta_i \mid X_i)$. We use the `SBI` Python package v0.23.2 to implement the flow matching model. We use the default model architecture in `SBI`, but use a custom context model to convert our high-dimensional $X_i$ into a low-dimensional context vector:

1. $X_i$ has initial shape 3×2000×2000
2. Max pooling for timing channel and Average pooling for counts channels with kernel size/stride of 20
3. 2D Convolution with max pooling and batch normalization
4. 2D Convolution with max pooling and batch normalization
5. Flattening and fully connected layer to a fixed sized context vector

Additional hyperparameters can be found on the `SBI` GitHub repository.