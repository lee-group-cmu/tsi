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
    def __init__(self, in_d, hidden_layers, nb_steps=50, dropout_rate=0.0, sigmoid=True, input_bounds=None):
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
        for idx, (h0, h1) in enumerate(zip(hs, hs[1:])):
            if idx < len(hs) - 2 and dropout_rate > 0: # Not output layer
                self.net.extend(
                    [
                        nn.Linear(h0, h1, device=None, dtype=None),
                        nn.Dropout(dropout_rate),
                        nn.ReLU(),
                    ]
                )
            else:
                self.net.extend(
                    [
                        nn.Linear(h0, h1, device=None, dtype=None),
                        nn.ReLU(),
                    ]
                )
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.nb_steps = nb_steps
        self.sigmoid = sigmoid
        self.dropout_rate = dropout_rate
        
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

    def forward_batch(self, x_input):
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

    def forward(self, x_input, batch_size=4096, show_progress=True):
        """
        Memory-safe batched forward pass for large inputs.
        
        Args:
            x_input: Input tensor or numpy array of shape [N, input_dim]
            batch_size: Number of samples to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Output tensor of shape [N, 1] (or [N] if not sigmoid)
        """
        # Convert to tensor if needed
        if isinstance(x_input, np.ndarray):
            x_input = torch.from_numpy(x_input)
        
        x_input = x_input.float()
        device = next(self.parameters()).device
        
        # If input is small enough, use regular forward
        if x_input.shape[0] <= batch_size:
            return self.forward_batch(x_input.to(device))
        
        # Process in batches
        n_samples = x_input.shape[0]
        outputs = []
        
        # Setup progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n_samples, desc="Forward pass")
            except ImportError:
                show_progress = False
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                # Get batch
                batch = x_input[i:i+batch_size].to(device)
                
                # Regular forward pass on batch
                batch_output = self.forward_batch(batch)
                
                # Move to CPU immediately to free GPU memory
                outputs.append(batch_output.cpu())
                
                # Update progress
                if show_progress:
                    pbar.update(batch.shape[0])
                
                # Clear GPU cache periodically
                if torch.cuda.is_available() and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        if show_progress:
            pbar.close()
        
        # Concatenate all batch outputs
        return torch.cat(outputs, dim=0)

    def fit(self, X, y, config=None, logger=None):
        return 

    def predict(self, X):
        return self.forward(X).round()

    def predict_proba(self, X):
        proba = self.forward(X)
        return torch.hstack([1-proba, proba])


class CompositeLoss(nn.Module):
    def __init__(self, lambda_gp=0.0, lambda_bce=1.0):
        """
        Composite loss combining BCE and gradient penalty.
        
        Args:
            lambda_gp: Weight for gradient penalty term
            lambda_bce: Weight for BCE term (usually 1.0)
        """
        super().__init__()
        self.bce = nn.BCELoss(reduction="sum")
        self.lambda_gp = lambda_gp
        self.lambda_bce = lambda_bce
    
    def gradient_penalty(self, model, x_input, predictions):
        """
        Compute gradient penalty w.r.t. test statistic (first input dimension).
        
        Args:
            model: The MonotonicNN model
            x_input: Input tensor with requires_grad=True
            predictions: Already computed predictions (to avoid recomputation)
        
        Returns:
            Gradient penalty scalar
        """
        if self.lambda_gp == 0.0:
            return torch.tensor(0.0, device=x_input.device)
        
        # Compute gradients w.r.t. input
        gradients = torch.autograd.grad(
            outputs=predictions,
            inputs=x_input,
            grad_outputs=torch.ones_like(predictions),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Only penalize gradient w.r.t. first dimension (test statistic)
        grad_test_stat = gradients[:, 0]
        
        # Penalize deviation from unit gradient (smoother extrapolation)
        # Alternative: just penalize large gradients with grad_test_stat.pow(2).mean()
        gp = ((grad_test_stat.abs() - 1).clamp(min=0) ** 2).mean()
        
        return gp
    
    def forward(self, model, x_input, predictions, targets):
        """
        Compute composite loss.
        
        Args:
            model: The MonotonicNN model (needed for gradient penalty)
            x_input: Input tensor (should have requires_grad=True for GP)
            predictions: Model predictions
            targets: Ground truth labels
        
        Returns:
            total_loss, loss_dict with individual components
        """
        # BCE loss
        bce_loss = self.bce(predictions, targets)
        
        # Gradient penalty
        gp_loss = self.gradient_penalty(model, x_input, predictions)
        
        # Total loss
        total_loss = self.lambda_bce * bce_loss + self.lambda_gp * gp_loss
        
        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'bce': bce_loss.item(),
            'gp': gp_loss.item() if self.lambda_gp > 0 else 0.0
        }
        
        return total_loss, loss_dict


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
    model = MonotonicNN(
        in_d=augmented_inputs.shape[-1],
        hidden_layers=config['hidden_layers'] or [256, 256, 256],
        dropout_rate=config['dropout_rate'] or 0.0,
        sigmoid=True,
        input_bounds=input_bounds
    )
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
        num_workers=0
    )
    
    # === LOSS FUNCTION FOR NONNEGATIVE OUTPUTS ===
    # Since model uses sigmoid (outputs in [0,1]), use BCE loss directly on probabilities
    loss_fn = CompositeLoss(
        lambda_gp=config.get("lambda_gp", 0.0),  # Default to 0 (no GP)
        lambda_bce=1.0
    )
    print(f"Using composite loss: BCE (weight={loss_fn.lambda_bce}) + "
          f"Gradient Penalty (weight={loss_fn.lambda_gp})")

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
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {train_size:,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient penalty: λ_GP={loss_fn.lambda_gp}")
    print(f"Dropout rate: {model.dropout_rate if hasattr(model, 'dropout_rate') else 0.0}")
    print(f"{'='*50}\n")
    
    for epoch in (pbar := tqdm(range(1, config["n_epochs"] + 1))):
        epoch_start_time = time.time()
        training_loss_batch = []
        bce_losses = []
        gp_losses = []
        model.train()
        
        # Gradient and output monitoring
        total_grad_norm = 0
        output_stats = []
        
        for batch_idx, (feature, target) in enumerate(train_dataloader):
            feature = feature.to(config["DEVICE"])
            target = target.to(config["DEVICE"])
            
            # Enable gradient tracking for inputs if using gradient penalty
            if loss_fn.lambda_gp > 0:
                feature.requires_grad_(True)
            
            optimizer.zero_grad()
            
            # Forward pass
            probs = model(feature)
            
            # Check for NaNs or Infs
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                num_nans = torch.isnan(probs).sum().item()
                num_infs = torch.isinf(probs).sum().item()
                total = probs.numel()
                print(f"\n⚠️  Epoch {epoch}, Batch {batch_idx}: "
                    f"{num_nans}/{total} NaNs, {num_infs}/{total} Infs")
            
            # Loss calculation
            loss, loss_dict = loss_fn(model, feature, probs.squeeze(), target.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Calculate gradient norm
            batch_grad_norm = sum(
                p.grad.data.norm(2).item() ** 2 
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            total_grad_norm += batch_grad_norm
            
            optimizer.step()
            
            # Store losses
            training_loss_batch.append(loss_dict['total'])
            bce_losses.append(loss_dict['bce'])
            gp_losses.append(loss_dict['gp'])
            
            # Monitor outputs
            output_stats.append({
                'mean': probs.mean().item(),
                'std': probs.std().item(),
                'min': probs.min().item(),
                'max': probs.max().item()
            })
        
        # === EPOCH SUMMARY ===
        epoch_time = time.time() - epoch_start_time
        train_loss_epoch = np.mean(training_loss_batch)
        avg_bce = np.mean(bce_losses)
        avg_gp = np.mean(gp_losses)
        avg_grad_norm = total_grad_norm / len(train_dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Output statistics
        avg_prob_mean = np.mean([s['mean'] for s in output_stats])
        avg_prob_std = np.mean([s['std'] for s in output_stats])
        prob_min = np.min([s['min'] for s in output_stats])
        prob_max = np.max([s['max'] for s in output_stats])
        
        # Update progress bar with key metrics
        pbar.set_postfix({
            'Loss': f'{train_loss_epoch:.4f}',
            'BCE': f'{avg_bce:.4f}',
            'GP': f'{avg_gp:.4e}',
            'LR': f'{current_lr:.2e}',
            'GradNorm': f'{avg_grad_norm:.3f}'
        })
        
        # Detailed logging every N epochs or at end
        if epoch % 10 == 0 or epoch == config["n_epochs"]:
            print(f"\n{'─'*70}")
            print(f"EPOCH {epoch}/{config['n_epochs']} SUMMARY ({epoch_time:.1f}s)")
            print(f"{'─'*70}")
            print(f"Loss:      Total={train_loss_epoch:.6f}  BCE={avg_bce:.6f}  GP={avg_gp:.6e}")
            print(f"Outputs:   μ={avg_prob_mean:.4f}  σ={avg_prob_std:.4f}  "
                f"range=[{prob_min:.4f}, {prob_max:.4f}]")
            print(f"Training:  LR={current_lr:.2e}  GradNorm={avg_grad_norm:.4f}")
            print(f"Progress:  Best={best_loss:.6f}  Patience={patience_counter}/{early_stop_patience}")
            print(f"{'─'*70}\n")
        
        # Learning rate scheduling
        scheduler.step(train_loss_epoch)
        
        # Early stopping
        if train_loss_epoch < best_loss:
            improvement = best_loss - train_loss_epoch
            best_loss = train_loss_epoch
            patience_counter = 0
            
            # Save best model
            save_path = f'{config.get("assets_dir", ".")}/best_monotonic_nn.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, save_path)
            
            if epoch % 10 == 0:
                print(f"✓ New best model saved (improved by {improvement:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n{'='*70}")
                print(f"EARLY STOPPING at epoch {epoch}")
                print(f"Best loss: {best_loss:.6f} (epoch {epoch - patience_counter})")
                print(f"{'='*70}\n")
                break
        
        # Log epoch data
        if logger:
            logger.log_epoch(
                epoch=epoch,
                epoch_loss=train_loss_epoch,
                bce_loss=avg_bce,
                gp_loss=avg_gp,
                epoch_time=epoch_time,
                lr=current_lr,
            )
    
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
    return model, input_bounds
