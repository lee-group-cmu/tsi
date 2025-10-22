import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import time
from lf2i.calibration.p_values import augment_calibration_set


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def clipped_relu(x):
    return torch.minimum(torch.maximum(torch.Tensor([0]), x), torch.Tensor([1]))


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = 0.5
    lam[:, -1] = 0.5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W**2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float()

    return cc_weights, steps


def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None):
    # Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    
    cc_weights, steps = cc_weights.to(x0), steps.to(x0)

    xT = x0 + nb_steps * step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        dzs = integrand(X_steps, h_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps + 1, -1)
        dzs = dzs * cc_weights.unsqueeze(0).expand(dzs.shape)
        z_est = dzs.sum(1)
        return z_est * (xT - x0) / 2
    else:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        x_tot = x_tot * (xT - x0) / 2
        x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1) * cc_weights.unsqueeze(0).expand(
            x_tot.shape[0], -1, x_tot.shape[1]
        )
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        x_tot_steps = x_tot_steps.contiguous().view(-1, x_tot.shape[1])

        g_param, g_h = computeIntegrand(X_steps, h_steps, integrand, x_tot_steps, nb_steps + 1)
        return g_param, g_h


def computeIntegrand(x, h, integrand, x_tot, nb_steps):
    h.requires_grad_(True)
    with torch.enable_grad():
        f = integrand.forward(x, h)
        g_param = _flatten(
            torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=False, retain_graph=True) # Testing without this step! #
        )
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    return g_param, g_h.view(int(x.shape[0] / nb_steps), nb_steps, -1).sum(1)


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.0


class ParallelNeuralIntegral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20):
        with torch.no_grad():
            x_tot = integrate(x0, nb_steps, (x - x0) / nb_steps, integrand, h, False)
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone(), h)
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, h = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        integrand_grad, h_grad = integrate(x0, nb_steps, x / nb_steps, integrand, h, True, grad_output)
        x_grad = integrand(x, h)
        x0_grad = integrand(x0, h)
        # Leibniz formula
        return -x0_grad * grad_output, x_grad * grad_output, None, integrand_grad, h_grad.view(h.shape), None


class MonotonicNN(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50, sigmoid=True, input_bounds=None):
        """
        Args:
            in_d: Input dimension
            hidden_layers: List of hidden layer sizes
            nb_steps: Number of steps for numerical integration
            sigmoid: Whether to apply sigmoid activation to output
            input_bounds: List of tuples [(min1, max1), (min2, max2), ...] defining 
                         bounds for each input dimension. If provided, inputs will be 
                         scaled from these bounds to [0, 3].
        """
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d - 1] + hidden_layers + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.nb_steps = nb_steps
        self.sigmoid = sigmoid
        
        # Store the bounds-based scaler as buffers
        if input_bounds is not None:
            if len(input_bounds) != in_d:
                raise ValueError(f"Number of bounds ({len(input_bounds)}) must match input dimension ({in_d})")
            
            # Convert bounds to tensors for scaling
            mins = torch.tensor([bound[0] for bound in input_bounds], dtype=torch.float32)
            maxs = torch.tensor([bound[1] for bound in input_bounds], dtype=torch.float32)
            ranges = maxs - mins
            
            # Check for zero ranges to avoid division by zero
            if torch.any(ranges == 0):
                raise ValueError("Input bounds cannot have zero range (min == max)")
            
            self.register_buffer('input_mins', mins)
            self.register_buffer('input_ranges', ranges)
            self.has_bounds = True
        else:
            self.has_bounds = False

    def _scale_input(self, x_input):
        """Scale input from given bounds to [0, 3]."""
        if self.has_bounds:
            # Scale from [min, max] to [0, 1], then to [0, 3]
            normalized = (x_input - self.input_mins) / self.input_ranges
            return normalized * 3.0
        return x_input

    def forward(self, x_input):
        x_input = x_input.float()
        
        # Apply bounds-based scaling if bounds are provided
        x_input_scaled = self._scale_input(x_input)
        
        x = x_input_scaled[:, 0][:, None]
        h = x_input_scaled[:, 1:]
        x0 = torch.zeros(x.shape).to(x_input)
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])
        if self.sigmoid:
            return torch.sigmoid(
                scaling
                * ParallelNeuralIntegral.apply(
                    x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps
                )
                + offset
            )
        else:
            return torch.squeeze(
                scaling
                * ParallelNeuralIntegral.apply(
                    x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps
                )
                + offset
            )

    def fit(self, X, y, config=None, logger=None):
        return 

    def predict(self, X):
        return self.forward(X).round()

    def predict_proba(self, X):
        proba = self.forward(X)
        return torch.hstack([1-proba, proba])


# Outputs object of type MonotonicNN
def train_monotonic_nn(T_prime, test_statistic, config, logger=None):
    """Train the Monotonic Neural Network with comprehensive debugging"""
    print("Training Monotonic Neural Network with debugging...")

    # === PRE-PROCESSING ===
    test_statistics_calib = test_statistic.evaluate(
            *T_prime, mode='critical_values'
        )
    augmented_inputs, rejection_indicators = augment_calibration_set(
        test_statistics=test_statistics_calib,
        poi=T_prime[0],
        num_augment=config['num_augment'],
        acceptance_region=test_statistic.acceptance_region
    )
    input_bounds = [(np.min(augmented_inputs[:, i]), np.max(augmented_inputs[:, i])) for i in range(augmented_inputs.shape[1])]

    # === DATA ANALYSIS ===
    print("\n" + "="*50)
    print("DATA ANALYSIS")
    print("="*50)
    print(f"Input shape: {augmented_inputs.shape}")
    print(f"Target shape: {rejection_indicators.shape}")
    
    # Check input statistics
    print("\nInput feature statistics:")
    for i in range(augmented_inputs.shape[1]):
        feat = augmented_inputs[:, i]
        print(f"Feature {i}: mean={np.mean(feat):.4f}, std={np.std(feat):.4f}, "
              f"min={np.min(feat):.4f}, max={np.max(feat):.4f}")
    
    # Check target distribution
    rejection_rate = np.mean(rejection_indicators)
    print(f"\nTarget distribution:")
    print(f"Rejection rate (1s): {rejection_rate:.4f}")
    print(f"Acceptance rate (0s): {1-rejection_rate:.4f}")
    
    if rejection_rate < 0.01 or rejection_rate > 0.99:
        print("⚠️  WARNING: Extreme class imbalance detected!")
    
    # === MODEL SETUP ===
    # Note: Model should output nonnegative values (sigmoid ensures [0,1])
    model = MonotonicNN(in_d=augmented_inputs.shape[-1], hidden_layers=config['hidden_layers'] or [256, 256, 256], sigmoid=True, input_bounds=input_bounds)
    model.to(config["DEVICE"])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {trainable_params:,} trainable, {total_params:,} total")
    
    # Create dataset with scaled inputs
    trainset = torch.utils.data.TensorDataset(
        torch.from_numpy(augmented_inputs).float(),
        torch.from_numpy(rejection_indicators[:, None]).float()
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"]
    )
    
    # === LOSS FUNCTION FOR NONNEGATIVE OUTPUTS ===
    # Since model uses sigmoid (outputs in [0,1]), use BCE loss directly on probabilities
    loss_fn = torch.nn.BCELoss(reduction="sum")
    print("Using BCE loss with sum reduction for nonnegative outputs (no class weighting)")
    
    # === OPTIMIZER WITH GRADIENT CLIPPING ===
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-7
    )
    
    train_size = len(trainset)
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    
    # === TRAINING LOOP WITH DEBUGGING ===
    print(f"\n{'='*50}")
    print("STARTING TRAINING")
    print(f"{'='*50}")
    
    for epoch in (pbar := tqdm(range(1, config["n_epochs"] + 1))):
        epoch_start_time = time.time()
        training_loss_batch = []
        model.train()
        
        # Gradient and output monitoring
        total_grad_norm = 0
        output_stats = []
        
        for batch_idx, (feature, target) in enumerate(train_dataloader):
            feature = feature.to(config["DEVICE"])
            target = target.to(config["DEVICE"])
            optimizer.zero_grad()
            
            # Forward pass (model outputs probabilities in [0,1] due to sigmoid)
            probs = model(feature)
            
            # # Verify outputs are nonnegative and in [0,1]
            # assert torch.all(probs >= 0) and torch.all(probs <= 1), \
            #     f"Batch {batch_idx}: model outputs outside [0,1] range!"

            # Check for NaNs or Infs in outputs
            num_nans = torch.isnan(probs).sum().item()
            num_infs = torch.isinf(probs).sum().item()
            total = probs.numel()
            if num_nans > 0 or num_infs > 0:
                print(f"Batch {batch_idx}: {num_nans}/{total} ({num_nans/total:.4%}) NaNs, "
                      f"{num_infs}/{total} ({num_infs/total:.4%}) Infs in outputs!")
            
            # Loss calculation using BCE on probabilities
            loss = loss_fn(probs.squeeze(), target.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Calculate gradient norm
            batch_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    batch_grad_norm += p.grad.data.norm(2).item() ** 2
            batch_grad_norm = batch_grad_norm ** 0.5
            total_grad_norm += batch_grad_norm
            
            optimizer.step()
            training_loss_batch.append(loss.item())
            
            # Monitor outputs
            output_stats.append({
                'mean_prob': probs.mean().item(),
                'std_prob': probs.std().item(),
                'min_prob': probs.min().item(),
                'max_prob': probs.max().item()
            })
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        train_loss_epoch = np.mean(training_loss_batch)
        avg_grad_norm = total_grad_norm / len(train_dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Output statistics
        avg_output_stats = {
            'mean_prob': np.mean([s['mean_prob'] for s in output_stats]),
            'std_prob': np.mean([s['std_prob'] for s in output_stats]),
            'min_prob': np.min([s['min_prob'] for s in output_stats]),
            'max_prob': np.max([s['max_prob'] for s in output_stats])
        }
        
        # Learning rate scheduling
        scheduler.step(train_loss_epoch)
        
        # Early stopping
        if train_loss_epoch < best_loss:
            best_loss = train_loss_epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'{config.get("assets_dir", ".")}/best_monotonic_nn.pt')
        else:
            patience_counter += 1
        
        # Log epoch data
        if logger:
            logger.log_epoch(
                epoch=epoch,
                epoch_loss=train_loss_epoch,
                batch_losses=training_loss_batch.copy(),
                epoch_time=epoch_time,
                lr=current_lr
            )
        
        # Detailed progress info
        epoch_len = len(str(config["n_epochs"]))
        msg = (f"[{epoch:>{epoch_len}}/{config['n_epochs']:>{epoch_len}}] | "
               f"loss: {train_loss_epoch:.5f} | "
               f"prob: {avg_output_stats['mean_prob']:.3f}±{avg_output_stats['std_prob']:.3f} | "
               f"range: [{avg_output_stats['min_prob']:.3f}, {avg_output_stats['max_prob']:.3f}] | "
               f"grad: {avg_grad_norm:.2e} | "
               f"lr: {current_lr:.2e}")
        pbar.set_description(msg)
        
        # Print detailed stats every 10 epochs or if outputs are problematic
        if epoch % 10 == 0 or avg_output_stats['max_prob'] < 0.01 or avg_output_stats['min_prob'] > 0.99:
            print(f"\nEpoch {epoch} detailed stats:")
            print(f"  Loss: {train_loss_epoch:.6f}")
            print(f"  Output probabilities: {avg_output_stats['mean_prob']:.4f} ± {avg_output_stats['std_prob']:.4f}")
            print(f"  Output range: [{avg_output_stats['min_prob']:.4f}, {avg_output_stats['max_prob']:.4f}]")
            print(f"  Gradient norm: {avg_grad_norm:.2e}")
            print(f"  Learning rate: {current_lr:.2e}")
            
            if avg_output_stats['max_prob'] < 0.01:
                print("  ⚠️  WARNING: All outputs near zero!")
            elif avg_output_stats['min_prob'] > 0.99:
                print("  ⚠️  WARNING: All outputs near one!")
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f'{config.get("assets_dir", ".")}/best_monotonic_nn.pt', weights_only=True))
    
    # === FINAL MODEL EVALUATION ===
    print(f"\n{'='*50}")
    print("FINAL MODEL EVALUATION")
    print(f"{'='*50}")
    
    model.eval()
    with torch.no_grad():
        # Test on a sample of data
        sample_size = min(1000, len(trainset))
        sample_indices = np.random.choice(len(trainset), sample_size, replace=False)
        
        sample_features = torch.stack([trainset[i][0] for i in sample_indices])
        sample_targets = torch.stack([trainset[i][1] for i in sample_indices])
        
        sample_features = sample_features.to(config["DEVICE"])
        sample_probs = model(sample_features)
        
        # Verify final outputs are nonnegative
        assert torch.all(sample_probs >= 0), "Final model outputs contain negative values!"
        print("✓ All final outputs are nonnegative")
        
        print(f"Final model output statistics (n={sample_size}):")
        print(f"  Mean probability: {sample_probs.mean().item():.4f}")
        print(f"  Std probability: {sample_probs.std().item():.4f}")
        print(f"  Min probability: {sample_probs.min().item():.4f}")
        print(f"  Max probability: {sample_probs.max().item():.4f}")
        
        # Check correlation with targets
        correlation = np.corrcoef(
            sample_probs.cpu().numpy().flatten(), 
            sample_targets.cpu().numpy().flatten()
        )[0, 1]
        print(f"  Correlation with targets: {correlation:.4f}")
        
        # Classification metrics
        predictions = (sample_probs > 0.5).float()
        # Ensure both tensors are on the same device
        sample_targets = sample_targets.to(predictions.device)
        accuracy = (predictions.squeeze() == sample_targets.squeeze()).float().mean()
        print(f"  Accuracy (threshold=0.5): {accuracy.item():.4f}")

    print("✓ Monotonic Neural Network training completed")
    return model
