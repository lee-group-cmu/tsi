from typing import Tuple, Optional, Dict
from tqdm import tqdm

import torch
import os
import torch.utils
from torch.utils.data import Dataset, DataLoader
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.core.utils import RuntimeLimits
from dingo.core.utils.trainutils import EarlyStopping


class SbiDataset(Dataset):
    """Dataset for SBI training with standardization. Just following the code from the paper FM for SBI paper."""
    
    def __init__(self, theta: torch.Tensor, x: torch.Tensor):
        super().__init__()
        self.standardization = {
            "x": {"mean": torch.mean(x, dim=0), "std": torch.std(x, dim=0)},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")

    def standardize(self, sample: torch.Tensor, label: str, inverse: bool = False):
        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]
        return (sample - mean) / std if not inverse else sample * std + mean

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx: int):
        return self.theta[idx], self.x[idx]


class DingoFMPE:
    
    def __init__(
        self, 
        device="cpu",
        activation_fn: str = 'gelu',
        batch_norm: bool = False,
        theta_with_glu: bool = False,
        x_with_glu: bool = False,
        dropout: float = 0.0,
        hidden_dims: Tuple[int] = (32, 64, 128, 256, 512, 1024, 1024, 1024, 512, 128, 64, 32),
        sigma_min: float = 0.0001,
        time_prior_exponent: int = 4
    ):
        self.device = device
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.theta_with_glu = theta_with_glu
        self.x_with_glu = x_with_glu
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        self.sigma_min = sigma_min
        self.time_prior_exponent = time_prior_exponent
        self.model = None

    def train(
        self, 
        theta: torch.Tensor, 
        x: torch.Tensor, 
        train_fraction: float = 0.95, 
        learning_rate: float = 0.0002,
        optimizer: str = "adam",
        scheduler_type: str = "reduce_on_plateau",
        scheduler_kwargs: Optional[Dict] = {"factor": 0.2, "patience": 1},
        early_stopping: bool = True,
        batch_size=64, 
        max_epochs=100, 
        num_dataloader_workers=0, 
        out_dir="./dingo_results"
    ):
        self.dataset = SbiDataset(theta, x)  # not really kosher to save the dataset internally, but for now it's okay
        train_size = int(train_fraction * len(self.dataset))
        validation_size = len(self.dataset) - train_size
        train_set, validation_set = torch.utils.data.random_split(self.dataset, [train_size, validation_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_dataloader_workers)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_dataloader_workers)

        # Auto-configure model settings
        model_kwargs = {
            "model": {
                "posterior_model_type": "flow_matching",
                "posterior_kwargs": {
                    "input_dim": self.dataset.theta.shape[1],
                    "context_dim": self.dataset.x.shape[1],
                    "activation": self.activation_fn,
                    "batch_norm": self.batch_norm,
                    "theta_with_glu": self.theta_with_glu,
                    "context_with_glu": self.x_with_glu,
                    "dropout": self.dropout,
                    "hidden_dims": self.hidden_dims,
                    "sigma_min": self.sigma_min,
                    "time_prior_exponent": self.time_prior_exponent
                },
                # "embedding_kwargs": {
                #     "added_context": False, 
                #     "input_dims": self.dataset.x.shape, 
                #     "output_dim": self.dataset.x.shape[1]  # same as context_dim above
                # }
            },
        }

        self.model = build_model_from_kwargs(settings={"train_settings": model_kwargs}, device=self.device)
        self.model.optimizer_kwargs = {"type": optimizer, "lr": learning_rate}
        self.model.scheduler_kwargs = {"type": scheduler_type, **scheduler_kwargs}
        self.model.initialize_optimizer_and_scheduler()

        runtime_limits = RuntimeLimits(epoch_start=0, max_epochs_total=max_epochs)
        self.model.train(
            train_loader,
            validation_loader, 
            train_dir=out_dir,
            runtime_limits=runtime_limits, 
            early_stopping=EarlyStopping(patience=10) if early_stopping else None
        )

        self.model = build_model_from_kwargs(filename=os.path.join(out_dir, "best_model.pt"), device=self.device)

    def sample(
        self, 
        sample_shape: Tuple[int], 
        x: torch.Tensor,
        batch_size: int = 500  # avoids huge memory usage from torch.odeint
    ) -> torch.Tensor:
        assert len(sample_shape) == 1, "Only the desired number of posterior samples has to be provided"
        x = self.dataset.standardize(x, label="x", inverse=False)

        num_batches = (sample_shape[0] + batch_size - 1) // batch_size
        posterior_samples = []
        sampled_so_far = 0
        for _ in tqdm(range(num_batches), desc="Sampling in batches for Dingo ..."):
            current_batch_size = min(batch_size, sample_shape[0] - sampled_so_far)
            print(current_batch_size, flush=True)
            batch_samples = self.model.sample(x, num_samples=current_batch_size)
            sampled_so_far += current_batch_size
            posterior_samples.append(batch_samples.reshape(current_batch_size, batch_samples.shape[-1]))
        posterior_samples = torch.cat(posterior_samples, dim=0)

        return self.dataset.standardize(posterior_samples, label='theta', inverse=True)
    
    def log_prob(
        self,
        theta: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.model.log_prob(theta, x)
