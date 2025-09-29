import torch
import time
import numpy as np

from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel
from gpytorch.priors import LogNormalPrior, GammaPrior
import gpytorch

from print_utils import pretty_vec, RED, GREEN, YELLOW, EOC

def gradient_descent_on_gp(x_init, gp, beta, lr, max_steps, tol, bounds_lower, bounds_upper, upper_boundary_tol):
    """
    Minimize: -posterior_mean + beta * posterior_std
    Returns: (x_final, final_loss, steps, hit_upper_boundary)
    """
    x = x_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    prev_loss = float('inf')
    for step in range(max_steps):
        optimizer.zero_grad()

        # Get posterior at current x
        with torch.enable_grad():
            posterior = gp.posterior(x.unsqueeze(0))
            mean = posterior.mean.squeeze()
            std = posterior.variance.sqrt().squeeze()

            # Objective: minimize -mean + beta*std
            loss = -mean + beta * std

        loss.backward()
        optimizer.step()

        # Project back to bounds
        with torch.no_grad():
            x.clamp_(bounds_lower, bounds_upper)

            # Check if we hit a UPPER boundary
            if ((x - bounds_upper).abs() < upper_boundary_tol).any().item():
                return x.detach(), loss.item(), step + 1, True  # Hit upper boundary!

        # Check convergence
        if abs(loss.item() - prev_loss) < tol:
            break
        prev_loss = loss.item()

    return x.detach(), loss.item(), step + 1, False


LENGTHSCALE_LOGNORMAL_PRIOR_LOC = np.log(0.1)
LENGTHSCALE_LOGNORMAL_PRIOR_SCALE = 0.5
OUTPUT_GAMMA_PRIOR_CONC = 2.0
OUTPUT_GAMMA_PRIOR_RATE = 0.15
NOISE_GAMMA_PRIOR_CONC = 2.0
NOISE_GAMMA_PRIOR_RATE = 4.0

def noisy_blackbox_optimization(objective_function, bounds,
                                x_init = None,
                                num_initial_points = 30,
                                num_iterations = 1000,
                                num_samples_per_iteration = 1,
                                resample_best_interval = 5,
                                log_folder = "tmp",
                                num_sobol_samples = 512,
                                num_restarts = 40,
                                raw_samples = 1024,
                                ):

    D = bounds.shape[1]
    all_cost_components = []

    # Draw random initial points within bounds, if they were not provided
    if x_init is None:
        train_X = torch.empty(num_initial_points, D, dtype=torch.float64)
        for i in range(D):
            train_X[:, i] = torch.rand(num_initial_points, dtype=torch.float64) * (bounds[1, i] - bounds[0, i]) + bounds[0, i]
    else:
        train_X = x_init.clone().to(dtype=torch.float64)
        num_initial_points = train_X.shape[0]

    # Compute initial values
    train_Y = torch.empty(num_initial_points, dtype=torch.float64)
    for i in range(train_X.shape[0]):
        print(f"Initial point {i + 1}/{num_initial_points}: x={pretty_vec(train_X[i, :])} ", end="", flush=True)
        train_Y[i], cost_component = objective_function(train_X[i, :])
        all_cost_components.append(cost_component)
        print("=" * 80)

    start_time = time.time()
    best_location_list = []
    current_best_location = train_X[train_Y.argmin()].tolist()
    consecutive_best = 0
    G = torch.rand(2000, D, dtype=train_X.dtype) * (bounds[1] - bounds[0]) + bounds[0] # This is for IVR computation

    for iteration in range(num_iterations):

        print(f"Iteration {iteration + 1}:")

        # 1. Fit a GP with explicit ScaleKernel : we negate Y because we do minimization and BoTorch does maximization.
        # GP expects a tensor of shape (n, 1)
        # For the kernel, we use a Matern kernel with nu=2.5 and ARD lengthscales.
        # This kernel is less smooth than RBF and often better for modeling noisy functions.
        # The ARD (Automatic Relevance Determination) allows different lengthscales for each input dimension.
        # The lengthscale prior is LogNormal to encourage positive values.
        # With loc = np.log(0.1) it encodes the fact that we expect lengthscales around 0.1
        # The outputscale prior is GammaPrior which is weakly informative, because we don't have strong prior knowledge about the function variance.
        # The likelihood noise prior is also GammaPrior to keep noise positive and avoid overfitting.
        # GammaPrior(2.0, 4.0) informs the noise variance about 0.5.

        gp = SingleTaskGP(
            train_X,
            -train_Y.unsqueeze(-1),
            covar_module=ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=D, lengthscale_prior=LogNormalPrior(LENGTHSCALE_LOGNORMAL_PRIOR_LOC, LENGTHSCALE_LOGNORMAL_PRIOR_SCALE)),
                outputscale_prior=GammaPrior(OUTPUT_GAMMA_PRIOR_CONC, OUTPUT_GAMMA_PRIOR_RATE)),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=GammaPrior(NOISE_GAMMA_PRIOR_CONC, NOISE_GAMMA_PRIOR_RATE),
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
            ),
            input_transform=Normalize(d=D),  # still useful even on [0,1]^D
            outcome_transform=Standardize(m=1),  # stabilizes MLL and acq
        )


        # 2. Optimize the model hyperparameters
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        gp.likelihood.eval()

        # Print learned hyperparameters
        with torch.no_grad():
            length_scales = gp.covar_module.base_kernel.lengthscale.squeeze()
            output_scale = gp.covar_module.outputscale.item()  # Now this exists!
            noise = gp.likelihood.noise.item()
            print(f"GP-HP : ℓ={pretty_vec(length_scales)}, σ²={output_scale:.3f}, σ²_n={noise:.3f}")
        # Print integrated variance (should decrease over time)
        ivr_latent = gp.posterior(G, observation_noise=False).variance.mean().item()
        ivr_pred = gp.posterior(G, observation_noise=True).variance.mean().item()
        print(f"{YELLOW}Integrated variance: latent={ivr_latent:.3f}  predictive={ivr_pred:.3f}{EOC}")

        # 3. Define the acquisition function (we need as sampler for NEI)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_sobol_samples]))
        acq_function = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_X, sampler=sampler)

        # 4. Optimize the acquisition function
        candidates, value = optimize_acqf(acq_function=acq_function, bounds=bounds, q=num_samples_per_iteration,
                                          num_restarts=num_restarts, raw_samples=raw_samples, sequential=(num_samples_per_iteration>1))
        first_candidate = candidates[0, :]
        print(f"Candidate {pretty_vec(first_candidate)} Acq value={pretty_vec(value)}")

        # 4b. Add a resample point at the best posterior mean among observed points
        if iteration % resample_best_interval == 0:
            posterior = gp.posterior(train_X)
            best_observed_idx = posterior.mean.argmax()
            resampled = train_X[best_observed_idx, :]
            candidates = torch.cat([candidates, resampled.unsqueeze(0)], dim=0)
            print(f"{RED}Resample at best location {pretty_vec(resampled)}{EOC}")

        # 5. Evaluate the objective function at the new candidate points and the resample point
        new_ys = torch.empty(candidates.shape[0], dtype=torch.float64)
        for i in range(candidates.shape[0]):
            new_ys[i], cost_component = objective_function(candidates[i, :])
            all_cost_components.append(cost_component)
        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, new_ys], dim=0)

        # X_u, Y_mean, Y_var = consolidate_replicates(train_X, train_Y)
        # print(f"consolidate_replicates results in {X_u.size(0)} unique points Y_mean= {Y_mean}, Y_var= {Y_var}")

        # 6. Report best values
        best_observed_value = train_Y.min().item()
        best_observed_location = train_X[train_Y.argmin()].tolist()

        best_posterior_mean_among_observed = - gp.posterior(train_X).mean.max().item()  # Reverse because we modeled -Y
        best_posterior_mean_location = train_X[gp.posterior(train_X).mean.argmax()].tolist()

        # Gradient descent from the best posterior mean location
        gd_start = torch.tensor(best_posterior_mean_location, dtype=torch.float64)
        gd_location, gd_loss, gd_steps, hit_boundary = gradient_descent_on_gp(gd_start, gp, 2.0, 0.01, 1000, 1e-6,
                                                                              bounds[0, :], bounds[1, :], 1e-5)
        gd_posterior = gp.posterior(gd_location.unsqueeze(0))

        if best_posterior_mean_location != current_best_location:
            print(f"Old best posterior mean location: {pretty_vec(current_best_location)}")
            print(f"{RED}New best posterior mean location: {pretty_vec(best_posterior_mean_location)}{EOC}")
            current_best_location = best_posterior_mean_location
            best_location_list.append(current_best_location)
            consecutive_best = 0
        consecutive_best += 1

        # Save the GP to the log folder
        if (iteration + 1) % 50 == 0 or iteration == num_iterations - 1:
            torch.save({
                'train_X': train_X,
                'train_Y': train_Y,
                'all_cost_components': all_cost_components,
                'gp_state_dict': gp.state_dict(),
                'mll_state_dict': mll.state_dict(),
                'best_location_list': best_location_list,
            }, f'{log_folder}/gp_iteration_{iteration + 1}.pt')

        elapsed = time.time() - start_time
        expected_completion_time = time.time() + (elapsed / (iteration + 1)) * (num_iterations - iteration - 1)

        print(f"Best observed value: {best_observed_value:.2f} at {pretty_vec(best_observed_location)}")
        print(
            f"Best posterior mean among observed points: {best_posterior_mean_among_observed:.2f} at {pretty_vec(best_posterior_mean_location)} Consecutive best: {consecutive_best}")
        print(
            f"GD from best: {-gd_posterior.mean.item():.2f}, σ={np.sqrt(gd_posterior.variance.item()):.2f} at {pretty_vec(gd_location.tolist())} Hit boundary: {hit_boundary}")
        print(
            f"Elapsed time: {elapsed / 60:.2f} min, Expected completion time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_completion_time))}")
        print("=" * 80)
        print()

    return {"X": train_X, "Y": train_Y, "best_x": train_X[train_Y.argmin()], "best_y": train_Y.min(),
            "log": best_location_list}


def load_gp(checkpoint):

    train_X = checkpoint['train_X']
    train_Y = checkpoint['train_Y']
    D = train_X.shape[1]

    gp = SingleTaskGP(
        train_X,
        -train_Y.unsqueeze(-1),
        covar_module=ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=D, lengthscale_prior=LogNormalPrior(LENGTHSCALE_LOGNORMAL_PRIOR_LOC,
                                                                                  LENGTHSCALE_LOGNORMAL_PRIOR_SCALE)),
                        outputscale_prior=GammaPrior(OUTPUT_GAMMA_PRIOR_CONC, OUTPUT_GAMMA_PRIOR_RATE)),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=GammaPrior(NOISE_GAMMA_PRIOR_CONC, NOISE_GAMMA_PRIOR_RATE),
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
            ),
        input_transform=Normalize(d=D),
        outcome_transform=Standardize(m=1),
    )

    gp.load_state_dict(checkpoint['gp_state_dict'])

    return gp, train_X, train_Y
