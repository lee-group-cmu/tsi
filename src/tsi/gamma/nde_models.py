import torch 
import numpy as np
import normflows as nf
import tiled_events as te
from typing import List, Union, Optional, Any, Callable
from torch.utils.data import default_collate
from tqdm import tqdm
from torch import Tensor 
from sbi.inference.trainers.npse import NPSE 
from sbi.inference.trainers.fmpe import FMPE
from sbi.inference.posteriors.score_posterior import ScorePosterior



class SinhArcsinhMDN1D(torch.nn.Module):
    
    def __init__(
        self,
        device: torch.device,
        context_size: int,
        num_components: int,
        gaussian: bool = False,
    ):
        super().__init__()
        self.mu = torch.nn.Linear(context_size, num_components)
        self.sigma = torch.nn.Linear(context_size, num_components) # sigma is logged
        
        self.gaussian = gaussian
        if self.gaussian:
            self.nu = None 
            self.tau = None 
        else:
            self.nu = torch.nn.Linear(context_size, num_components) 
            self.tau = torch.nn.Linear(context_size, num_components) # tau is logged
        
        if num_components > 1:
            self.pi = torch.nn.Sequential(
                torch.nn.Linear(context_size, num_components),
                torch.nn.Softmax(dim=1)
            )
        else:
            self.pi = None
            
        self.device = device 
        self.to(device)
        self.num_components = num_components
        
    def forward(self, context_batch):
        # context_batch shape is batch_size, context_size
        if self.pi is not None:
            pi = self.pi(context_batch)
        else:
            pi = torch.ones(context_batch.shape[0], 1, device=self.device)
        mu = self.mu(context_batch)
        sigma = torch.exp(self.sigma(context_batch))
        if self.gaussian:
            nu = torch.zeros(context_batch.shape[0], 1, device=self.device)
            tau = torch.ones(context_batch.shape[0], 1, device=self.device)
        else:
            nu = self.nu(context_batch)
            tau = torch.exp(self.tau(context_batch))
        return pi, mu, sigma, nu, tau
    

ONEOVERSQRT2PI = 1.0 / np.sqrt(2 * np.pi)
def log_sinh_arcsinh_density(x, mu, sigma, nu, tau):
    def s(t, e, d):
        return torch.sinh(
            torch.arcsinh(t) * d - e
        )
    z = (x - mu)/sigma 
    s2 = s(z, nu, tau)**2
    
    base = torch.log(tau/sigma) + 0.5 * (torch.log(1 + s2) - torch.log(2 * np.pi * (1 + z**2)) - s2)
    base[torch.isinf(s2)] = -torch.inf
    return base

def sinh_arcsinh_mdn_nll_loss(pi, mu, sigma, nu, tau, target, weights=None):
    assert target.shape[-1] == 1
    density = (torch.exp(log_sinh_arcsinh_density(target, mu, sigma, nu, tau)) * pi).sum(dim=-1)
    if weights is None:
        weights = torch.ones(density.shape)
    assert density.shape == weights.shape
    return (-torch.log(density) * weights).sum()/weights.sum()


class ContextModel(torch.nn.Module):
    def __init__(
        self, 
        device: torch.device,
        num_channels: int, 
        final_grid_shape, 
        context_size: int, 
        kernel_size: int
    ):
        super().__init__()
        self.relu = torch.nn.ReLU()
        
        self.conv1 = torch.nn.Conv2d(num_channels, 8, kernel_size=kernel_size) # -4
        self.maxpool1 = torch.nn.MaxPool2d(4, stride=4) # /4
        self.bn1 = torch.nn.BatchNorm2d(8)
        # then relu
        
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=kernel_size) # - 4
        self.maxpool2 = torch.nn.MaxPool2d(4, stride=4) # /4
        self.bn2 = torch.nn.BatchNorm2d(16)
        # then relu
        
        # 100 -> 5**2 x 16 = 400
        kernel_trim = kernel_size - 1
        to_dense = int(((final_grid_shape[0] - kernel_trim)/4 - kernel_trim)/4) * int(((final_grid_shape[1] - kernel_trim)/4 - kernel_trim)/4) * 16
        self.fc1 = torch.nn.Linear(to_dense, context_size)
        self.bn3 = torch.nn.BatchNorm1d(context_size)
        # then relu
        
        self.device = device
        self.to(device)
        self.context_size = context_size
        
    def forward(self, features: torch.Tensor):
        b = features.shape[0]
        features = self.conv1(features)
        features = self.maxpool1(features)
        features = self.bn1(features)
        features = self.relu(features)
        
        features = self.conv2(features)
        features = self.maxpool2(features)
        features = self.bn2(features)
        features = self.relu(features)
        
        features = self.fc1(features.view(b, -1))
        features = self.bn3(features)
        return self.relu(features)
    
class BiggerContextModel(torch.nn.Module):
    def __init__(
        self, 
        device: torch.device,
        num_channels: int, 
        final_grid_shape, 
        context_size: int, 
        kernel_size: int
    ):
        super().__init__()
        
        max_pool_kernel_size = 2
        base_out_channels = 8
        num_conv_layers = 3
        
        self.cnn_modules = torch.nn.ModuleList()
        
        for i in range(num_conv_layers):
            self.cnn_modules.extend([
                torch.nn.Conv2d(
                    num_channels if i == 0 else base_out_channels * 2 ** (i - 1), 
                    base_out_channels * 2 ** i, 
                    kernel_size=kernel_size
                ),
                torch.nn.MaxPool2d(max_pool_kernel_size, stride=max_pool_kernel_size),
                torch.nn.BatchNorm2d(base_out_channels * 2 ** i),
                torch.nn.ReLU()
            ])
        
        # 100 -> 5**2 x 16 = 400
        out_channels = base_out_channels * 2 ** (num_conv_layers - 1)
        image_height = final_grid_shape[0]
        image_width = final_grid_shape[1]
        for _ in range(num_channels):
            image_height -= (kernel_size - 1)
            image_height /= max_pool_kernel_size
            
            image_width -= (kernel_size - 1)
            image_width /= max_pool_kernel_size
        
        to_dense = int(image_width) * int(image_height) * out_channels
        self.fc1 = torch.nn.Linear(to_dense, to_dense//2)
        self.bn1 = torch.nn.BatchNorm1d(to_dense//2)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(to_dense//2, context_size)
        
        self.device = device
        self.to(device)
        self.context_size = context_size
        
    def forward(self, features: torch.Tensor):
        b = features.shape[0]
        for module in self.cnn_modules:
            features = module(features)
            
        features = self.fc1(features.view(b, -1))
        features = self.bn1(features)
        features = self.relu1(features)
        return self.fc2(features)

class SplitCRNF(torch.nn.Module):
    def __init__(
        self, 
        device: torch.device,
        num_channels: int, 
        final_grid_shape, 
        context_size: int, 
        kernel_size: int,
        num_energy_mdn_components: int,
        num_angle_flows: int,
        gaussian: bool = False
    ) -> None:
        super().__init__()
        
        self.context_model = ContextModel(
            device,
            num_channels,
            final_grid_shape,
            context_size,
            kernel_size
        )

        # Energy ------------------------------------------------------
        # Sinh-Arcsinh MDN
        self.energy_mdn = torch.nn.Sequential(
            torch.nn.Linear(in_features=context_size, out_features=context_size//2),
            SinhArcsinhMDN1D(device, context_size//2, num_energy_mdn_components, gaussian)
        )
        self.num_energy_mdn_components = num_energy_mdn_components
        
        # End Energy -------------------------------------
        # Angle Flows
        if num_angle_flows > 0:
            latent_size = 2
            hidden_units = 128
            num_blocks = 2

            angle_flows = []
            for _ in range(num_angle_flows):
                angle_flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                            context_features=context_size, 
                                                            num_blocks=num_blocks)]
                angle_flows += [nf.flows.LULinearPermute(latent_size)]

            # Set base distribution
            q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
                
            # Construct flow model
            self.angle_nf = nf.ConditionalNormalizingFlow(q0, angle_flows)
        else:
            self.angle_nf = None
            
        self.device = device
        self.to(device)
        
class JointCRNF(torch.nn.Module):
    def __init__(
        self, 
        device: torch.device,
        context_model: Union[ContextModel, BiggerContextModel],
        num_flows: int,
        no_azimuth: bool = False
    ) -> None:
        super().__init__()
        
        self.context_model = context_model

        # MAF ------------------------------------------------------
        latent_size = 2 if no_azimuth else 3
        hidden_units = 128
        num_blocks = 2

        flows = []
        for _ in range(num_flows):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                        context_features=context_model.context_size, 
                                                        num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
            
        # Construct flow model
        self.nf = nf.ConditionalNormalizingFlow(q0, flows)
        
        self.device = device
        self.to(device)
        
class LF2IModelWrapper(torch.nn.Module):
    def __init__(self, model: Union[JointCRNF, SplitCRNF]) -> None:
        super().__init__()
        self.model = model
    
    # X IS SCALED!!!
    def sample(self, sample_shape, x: torch.Tensor, show_progress_bars=False):
        self.model.eval()
        with torch.no_grad():
            context = self.model.context_model.forward(x[None, :].to(self.model.device)).expand(sample_shape[0], -1)
            if type(self.model) is JointCRNF:
                samples, _ = self.model.nf.sample(sample_shape[0], context)
                return samples.cpu()
            elif type(self.model) is SplitCRNF:
                z = torch.randn(sample_shape[0], self.model.num_energy_mdn_components)
                pi, mu, sigma, nu, tau = self.model.energy_mdn(context)
                zx = sigma * torch.sinh((torch.arcsinh(z) + nu)/tau) + mu
                energy_samples = (zx * pi).sum(dim=-1)
                
                angle_samples, _ = self.model.angle_nf.sample(sample_shape[0], context)
                samples = torch.column_stack(energy_samples, angle_samples)
                raise Exception("Check this")
                return samples * (self.param_maxes - self.param_mins) + self.param_mins
            else:
                raise NotImplementedError
    
class WeightedNPSE(NPSE):
    def _loss(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
    ) -> Tensor:
        """Return loss from score estimator. Currently only single-round NPSE
         is implemented, i.e., no proposal correction is applied for later rounds.

        The loss is the negative log prob. Irrespective of the round or SNPE method
        (A, B, or C), it can be weighted with a calibration kernel.

        Returns:
            Calibration kernel-weighted negative log prob.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
        """
        if self._round == 0 or force_first_round_loss:
            # First round loss.
            loss = self._neural_net.loss(theta, x)
        else:
            raise NotImplementedError(
                "Multi-round NPSE with arbitrary proposals is not implemented"
            )

        return calibration_kernel(theta) * loss
    
from copy import deepcopy
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.inference.trainers.base import NeuralInference
from sbi.utils import (
    RestrictedPrior,
    handle_invalid_x,
    npe_msg_on_invalid_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
)
from sbi.utils import x_shape_from_simulation
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import time
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.utils.sbiutils import mask_sims_from_prior

class WeightedFMPE(FMPE):
    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
        calibration_kernel: Callable = lambda theta: 1,
    ) -> ConditionalDensityEstimator:
        """Train the flow matching estimator.

        Args:
            training_batch_size: Batch size for training. Defaults to 50.
            learning_rate: Learning rate for training. Defaults to 5e-4.
            validation_fraction: Fraction of the data to use for validation.
            stop_after_epochs: Number of epochs to train for. Defaults to 20.
            max_num_epochs: Maximum number of epochs to train for.
            clip_max_norm: Maximum norm for gradient clipping. Defaults to 5.0.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: Whether to allow training with
                simulations that have not been sampled from the prior, e.g., in a
                sequential inference setting. Note that can lead to biased inference
                results.
            show_train_summary: Whether to show the training summary. Defaults to False.
            dataloader_kwargs: Additional keyword arguments for the dataloader.

        Returns:
            DensityEstimator: Trained flow matching estimator.
        """

        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss or resume_training, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "FMPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        start_idx = 0  # as there is no multi-round FMPE yet

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training=resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        if self._neural_net is None:
            # Get theta, x to initialize NN
            theta, x, _ = self.get_simulations(starting_round=start_idx)

            # Use only training data for building the neural net (z-scoring transforms)
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x.to("cpu"))

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        # initialize optimizer and training parameters
        if not resume_training:
            self.optimizer = Adam(
                list(self._neural_net.net.parameters()), lr=learning_rate
            )
            self.epoch = 0
            # NOTE: in the FMPE context we use MSE loss, not log probs.
            self._val_loss = float("Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            self._neural_net._embedding_net.train()
            self._neural_net.net.train()
            train_loss_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # get batches on current device.
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )

                train_loss = self._neural_net.loss(theta_batch, x_batch)
                train_loss = (calibration_kernel(theta_batch) * train_loss).mean()
                train_loss_sum += train_loss.item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_loss_average = train_loss_sum / len(train_loader)  # type: ignore
            self._summary["training_loss"].append(train_loss_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    # Aggregate the validation losses.
                    val_losses = self._neural_net.loss(theta_batch, x_batch)
                    val_losses = (calibration_kernel(theta_batch) * val_losses)
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation loss for every epoch.
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def append_simulations(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        proposal: Optional[DirectPosterior] = None,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ) -> NeuralInference:
        if (
            proposal is None
            or proposal is self._prior
            or (
                isinstance(proposal, RestrictedPrior) and proposal._prior is self._prior
            )
        ):
            current_round = 0
        else:
            raise NotImplementedError(
                "Sequential FMPE with proposal different from prior is not implemented."
            )

        if exclude_invalid_x is None:
            exclude_invalid_x = current_round == 0

        if data_device is None:
            data_device = self._device

        # theta, x = validate_theta_and_x(
        #     theta, x, data_device=data_device, training_device=self._device
        # )

        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        # Check whether there are NaNs or Infs in the data and remove accordingly.
        npe_msg_on_invalid_x(
            num_nans=num_nans,
            num_infs=num_infs,
            exclude_invalid_x=exclude_invalid_x,
            algorithm="Single-round FMPE",
        )

        self._data_round_index.append(current_round)
        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        return self
    
class ScoreModel(torch.nn.Module):
    def __init__(
        self, 
        device: torch.device,
        num_channels: int, 
        final_grid_shape, 
        kernel_size: int,
        t_embedding_dim: int,
        param_dim: int
    ):
        super().__init__()
        self.relu = torch.nn.ReLU()
        
        self.conv1 = torch.nn.Conv2d(num_channels, 8, kernel_size=kernel_size) # -4
        self.maxpool1 = torch.nn.MaxPool2d(4, stride=4) # /4
        self.bn1 = torch.nn.BatchNorm2d(8)
        # then relu
        
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=kernel_size) # - 4
        self.maxpool2 = torch.nn.MaxPool2d(4, stride=4) # /4
        self.bn2 = torch.nn.BatchNorm2d(16)
        # then relu
        
        # 100 -> 5**2 x 16 = 400
        kernel_trim = kernel_size - 1
        to_dense = int(((final_grid_shape[0] - kernel_trim)/4 - kernel_trim)/4) * int(((final_grid_shape[1] - kernel_trim)/4 - kernel_trim)/4) * 16 + param_dim + t_embedding_dim
        self.fc1 = torch.nn.Linear(to_dense, 256)
        self.fc2 = torch.nn.Linear(256, param_dim)
        # then relu
        
        self.device = device
        self.to(device)
        
    def forward(self, params, features: torch.Tensor, t_embedding):
        b = features.shape[0]
        sample_dim = None
        if len(features.shape) == 5:
            sample_dim = features.shape[1]
            features = features.view(-1, *features.shape[2:])    

        features = self.conv1(features)
        features = self.maxpool1(features)
        features = self.bn1(features)
        features = self.relu(features)
        
        features = self.conv2(features)
        features = self.maxpool2(features)
        features = self.bn2(features)
        features = self.relu(features)
        
        if sample_dim is None:
            features = features.view(b, -1)
        else:
            features = features.view(b, sample_dim, -1)
        
        features = torch.cat((features, params, t_embedding), dim=-1)
        features = self.fc1(features)
        features = self.relu(features)
        out = self.fc2(features)
        
        if sample_dim is not None:
            out = out.view(b, sample_dim, -1)
        return out
    
class SbiPosteriorLF2IWrapper:
    def __init__(
        self, 
        base_posterior: ScorePosterior, 
        time_steps: int, 
        device: torch.device,
        is_npse: bool
    ) -> None:
        self.base_posterior = base_posterior
        self.ts = torch.linspace(1, 0.0005, time_steps, device=device)
        self.is_npse = is_npse
        
    def log_prob(self, *args, **kwargs):
        return self.base_posterior.log_prob(*args, **kwargs).cpu()
    
    def sample(self, sample_shape, x, **kwargs):
        if self.is_npse:
            return self.base_posterior.sample(sample_shape, x, ts=self.ts, **kwargs).cpu()
        else:
            return self.base_posterior.sample(sample_shape, x, **kwargs).cpu()
        
def img_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)
        
def get_context(model: Union[SplitCRNF, JointCRNF], features):
    return model.context_model(features.to(model.device))
    
def energy_grid_log_prob(model: SplitCRNF, features, energy_grid):
    model.eval()
    # features = single obs only
    # assert len(energy_grid.shape) == 2
    # assert features.shape[0] == 1
    context = get_context(model, features).expand(energy_grid.shape[0], -1)
    # pi, sigma, mu = model.energy_mdn(context)
    # return torch.log((pi * mdn.gaussian_probability(sigma, mu, torch.log(energy_grid))).sum(dim=-1))
    pi, mu, sigma, nu, tau = model.energy_mdn(context)
    # tau = torch.clip(tau, 0, 19)
    return torch.log((torch.exp(log_sinh_arcsinh_density(energy_grid, mu, sigma, nu, tau)) * pi).sum(dim=-1))

def forward_kld(model: Union[SplitCRNF, JointCRNF], features, params, weights=None):
    context = get_context(model, features.to(model.device))
    params = params.to(model.device)
    if type(model) is SplitCRNF:
        pi, mu, sigma, nu, tau = model.energy_mdn(context)
        energy_mdn_loss = sinh_arcsinh_mdn_nll_loss(pi, mu, sigma, nu, tau, params[:, 0, None])
        if model.angle_nf is not None:
            # angle_nf_loss = model.angle_nf.forward_kld(params[:, 1:], context)
            angle_nf_loss = _nf_weighted_forward_kld(model.angle_nf, params[:, 1:], context, weights)
        else:
            angle_nf_loss = None
        return energy_mdn_loss, angle_nf_loss
    elif type(model) is JointCRNF:
        # return model.nf.forward_kld(params, context), None
        return _nf_weighted_forward_kld(model.nf, params, context, weights), None
    else:
        raise NotImplementedError
    
def _nf_weighted_forward_kld(nf_model, x, context=None, weights=None):
    """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

    Args:
        x: Batch sampled from target distribution
        context: Batch of conditions/context

    Returns:
        Estimate of forward KL divergence averaged over batch
    """
    if weights is None:
        weights = torch.ones(len(x), device=x.device)
    log_q = torch.zeros(len(x), device=x.device)
    z = x
    for i in range(len(nf_model.flows) - 1, -1, -1):
        z, log_det = nf_model.flows[i].inverse(z, context=context)
        log_q += log_det
    log_q += nf_model.q0.log_prob(z, context=context)
    
    assert weights.shape == log_q.shape
    # return -(log_q * weights).sum()/weights.sum()
    return -(log_q * weights).sum()/weights.shape[0]
    
def log_probs(model: Union[SplitCRNF, JointCRNF], scaled_features: torch.Tensor, scaled_param_grid: torch.Tensor):
    model.eval()
    with torch.no_grad():
        context = get_context(model, scaled_features).expand(scaled_param_grid.shape[0], -1)
        if type(model) is SplitCRNF:
            pi, mu, sigma, nu, tau = model.energy_mdn(context)
            energy_log_probs = torch.log((torch.exp(log_sinh_arcsinh_density(scaled_param_grid[:, 0, None], mu, sigma, nu, tau)) * pi).sum(dim=-1))
            if model.angle_nf is not None:
                angle_log_probs = model.angle_nf.log_prob(scaled_param_grid[:, 1:], context)
            return energy_log_probs, angle_log_probs
        elif type(model) is JointCRNF:
            return model.nf.log_prob(scaled_param_grid, context), None
        else:
            raise NotImplementedError
        
def model_loss_acc_loader(
    model: Union[SplitCRNF, JointCRNF], 
    loader: torch.utils.data.DataLoader, 
    param_mins: List[float], 
    param_maxes: List[float],
    limit_batches: int = None
):
    total_loss1 = 0
    total_loss2 = 0
    total_size = 0
    
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(loader, desc="Loss Acc")):
            if limit_batches is not None and batch_id >= limit_batches:
                break
            te.scale_batch_params_inplace(batch, param_mins, param_maxes)
            loss1, loss2 = forward_kld(model, batch['features'].to(model.device), batch['params'].to(model.device), batch['weights'].to(model.device))
            total_loss1 += loss1.item() * batch['features'].shape[0]
            loss2 = 0 if loss2 is None else loss2.item()
            total_loss2 += loss2 * batch['features'].shape[0]
            total_size += batch['features'].shape[0]
    
    return total_loss1/total_size, total_loss2/total_size