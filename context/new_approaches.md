# VarShare: KL/Prior Experiments Pack (Detailed Spec)

This document specifies a set of separate experiments to evaluate different strategies for controlling the KL regularizer and/or the prior variance in VarShare. Each item below should be implemented as its own experiment configuration (or method variant), so runs are directly comparable and results can be attributed to the intended change.

General requirements (apply to all experiments)
- Keep the RL backbone and training pipeline identical across experiments (PPO, same rollout/update schedule, same task sampling, same evaluation protocol).
- Keep network architecture identical across experiments (unless explicitly capacity-matched baselines are part of the broader project).
- Use the same random seeds across experiment variants to improve paired comparability.
- Treat each strategy as a togglable option controlled by config, not by code edits scattered throughout the repo.
- Log all key quantities needed to interpret behavior: reward/success curves, KL, posterior mean magnitude, posterior variance statistics, and any adaptive-controller variables used.
- Clearly separate “rollout-time behavior” and “training-time behavior” with explicit flags, especially regarding whether weights are sampled or means are used for acting.

Shared notation
- Tasks: m ∈ {1,…,M}
- Shared parameters: θ
- Task residual distribution: q_m(φ_m) = Normal(μ_m, diag(σ_m^2))
- Prior: p_m(φ_m) = Normal(0, σ_prior,m^2 I)
- Effective weights: W_m = θ + φ_m (or θ + g(φ_m) if low-rank adapters; the same logic applies)
- KL per task: KL_m = KL(q_m || p_m)
- Total objective (conceptual): PPO_loss + β(t) * KL_total + optional hyperprior penalty terms
- “Update step” means one PPO parameter update iteration (not an environment step)

Metrics to log for all variants
A. Performance metrics
- Training: mean return per task (or success rate for Meta-World) aggregated across tasks, plus per-task curves
- Evaluation: same metrics in deterministic evaluation mode (typically using posterior means, unless explicitly sampling)

B. KL and posterior diagnostics
- KL_total (sum or mean across tasks present in batch)
- KL per task (at least for MT10; possibly for a subset of tasks if logging volume is high)
- Mean magnitude of μ: e.g., median over parameters of |μ| per task and/or layer
- Posterior scale: median/mean of σ per task and/or layer; also log mean log σ
- Optional: ratio statistic R = median(σ) / (median(|θ|)+ε), globally and per layer (useful for sampling noise analysis)

C. Controller state (when applicable)
- β(t) trajectory over time
- Controller trigger events count
- Any EMA variables used by the controller
- σ_prior,m values over time (for adaptive prior variants)
- Hyperprior penalty value (if used)

Common evaluation mode convention
- Evaluation should use posterior mean (μ) rather than sampled φ, unless a specific experiment requires stochastic evaluation.
- If training uses sampling, explicitly log the “acting mode”: mean vs sampled weights during rollout collection.

--------------------------------------------------------------------------------
EXPERIMENT 1: Fixed sigma_prior = 0.01 (change nothing else)
--------------------------------------------------------------------------------

Goal
- Test whether a much tighter prior scale prevents posterior σ inflation and destructive sampling noise, without introducing additional complexity (no annealing, no adaptive updates).

Definition
- Set σ_prior,m = 0.01 for all tasks m and all layers (global fixed prior scale).
- Keep β fixed at the current default (no annealing).
- Keep all other VarShare settings unchanged.

Rationale
- Under a unit normal prior, KL is minimized at σ=1, which can make sampling extremely noisy in weight units.
- A small σ_prior shifts the KL optimum to a small variance scale, strongly discouraging both large μ and large σ.
- This experiment isolates the effect of correcting prior scale alone.

Expected outcomes / things to watch
- σ_m should be pulled toward ~σ_prior (very small), reducing sampling noise.
- μ_m may also be pulled toward 0 more strongly; if performance collapses, prior may be too tight.
- KL values may become larger initially; verify numerical stability.

Implementation details to enforce
- Ensure σ_prior is actually used in the KL formula (both the log term and denominator).
- Confirm no other code path overrides σ_prior.

Logging emphasis
- Track posterior σ statistics carefully; confirm they do not drift to ~1.
- Track μ magnitude; ensure specialization is not entirely suppressed.

--------------------------------------------------------------------------------
EXPERIMENT 2: Normal KL beta annealing (time-based)
--------------------------------------------------------------------------------

Goal
- Improve training stability and/or specialization emergence by gradually introducing KL pressure, while keeping the method simple and paper-defensible.

Chosen schedule
- Use a three-phase schedule: warm-up → hold → late ramp-up.
- This is chosen because it targets both early “over-sharing collapse” and late “variance/noise issues”.

Definition
Let total training length be T updates (or a known proxy such as total env steps; choose one and use consistently).

Phase A (Warm-up)
- For t in [0, T_warm]:
  β(t) increases from 0 to β1.
  β(t) = β1 * (t / T_warm)

Phase B (Hold)
- For t in (T_warm, T_ramp_start]:
  β(t) = β1

Phase C (Late ramp-up / compression)
- For t in (T_ramp_start, T]:
  β(t) increases from β1 to β2 > β1:
  β(t) = β1 + (β2 - β1) * (t - T_ramp_start) / (T - T_ramp_start)

Suggested default fractions (interpretable, stable)
- T_warm = 0.10 * T
- T_ramp_start = 0.80 * T
- β2 = 2 * β1
- β1 is the “baseline β” you would otherwise have used without annealing

Rationale
- Early: let PPO discover useful specialization signals before strong sharing pressure kicks in.
- Late: increase regularization to reduce posterior variance and compress unnecessary specialization.

Expected outcomes / things to watch
- Compare learning curves to baseline: annealing can reduce early instability or accelerate specialization.
- If late ramp harms final performance, β2 may be too high or ramp too aggressive.

Logging emphasis
- Log β(t) every update.
- Track σ statistics; late ramp should reduce σ inflation if that was present.

--------------------------------------------------------------------------------
EXPERIMENT 3: Event-triggered KL beta annealing (noise-to-weight ratio controller)
--------------------------------------------------------------------------------

Goal
- Control KL strength adaptively using an interpretable signal that relates directly to sampling noise, without specifying a KL target.

Core idea
- Maintain the posterior sampling noise at a reasonable fraction of typical weight magnitude.
- Adjust β multiplicatively to keep the ratio within a band.

Signal definition
Let
- σ_all be the collection of posterior std parameters currently used for sampling (across tasks, layers, parameters).
- θ_all be the collection of shared backbone weights (optionally per layer).

Define a robust noise-to-weight ratio:
- numerator: median over (m, layer, param) of σ_{m,layer,param}
- denominator: median over (layer, param) of |θ_{layer,param}| + ε

R_raw = median(σ_all) / (median(|θ_all|) + ε)

Use EMA smoothing:
R_ema ← (1 - α) * R_ema + α * R_raw

Controller band and update rule
Choose a band [R_min, R_max] with interpretable meaning:
- R_min = 0.05
- R_max = 0.20
Interpretation: typical posterior noise should be about 5–20% of typical weight magnitude.

Update every K_controller PPO updates (not every minibatch):
- if R_ema > R_max: β ← min(β * γ_up, β_max)
- else if R_ema < R_min: β ← max(β / γ_down, β_min)
- else: β unchanged

Suggested defaults
- EMA α = 0.05
- K_controller = 20 PPO updates
- γ_up = γ_down = 1.10
- β_min and β_max: set wide but finite bounds to avoid runaway (example: β_min = 1e-4, β_max = 100, but pick based on your current β scale)

Rationale
- Directly addresses the concern: if posterior σ grows too large relative to weights, sampled rollouts become unstable.
- Avoids choosing a KL target that is hard to interpret.

Expected outcomes / things to watch
- β should rise when σ inflates, reducing noise; and fall if σ becomes too small, allowing specialization.
- Potential risk: if θ magnitudes change significantly over training, ratio becomes more stable than absolute σ control (that is intended).
- Ensure controller updates are infrequent to avoid oscillations.

Logging emphasis
- Log R_raw and R_ema over time.
- Log β changes and the trigger direction.
- Track σ distributions per layer to verify controller acts on the real failure mode.

--------------------------------------------------------------------------------
EXPERIMENT 4: sigma_prior adaptive updating (recommended approach: loss-gap signal, bounded mapping)
--------------------------------------------------------------------------------

Goal
- Make σ_prior task-adaptive based on an interpretable “need for specialization” signal, without learning σ_prior via gradients.
- This reduces manual tuning and ties prior loosening to measurable benefit of specialization.

Key design choice
- Use a critic loss-gap metric: how much better the adapted model fits critic targets than the shared-only model.
- Update σ_prior slowly (meta-parameter), with hard bounds.

Diagnostic batches
- Maintain or sample diagnostic data per task m.
- For each task m, obtain a batch B_m periodically (every K_prior updates).

Loss definitions (critic-focused for stability)
Compute two critic losses on B_m, using posterior mean (no sampling):
- Shared-only: L_shared,m = mean over B_m of (V_θ(s) - V_target(s))^2
- Adapted: L_adapt,m = mean over B_m of (V_{θ+μ_m}(s) - V_target(s))^2

Define gap:
Δ_m = L_shared,m - L_adapt,m

Smoothing
Maintain EMA:
Δ̄_m ← (1 - α) * Δ̄_m + α * Δ_m

Normalize across tasks
Compute robust center and scale:
c = median_m(Δ̄_m)
s = MAD_m(Δ̄_m) + ε
z_m = (Δ̄_m - c) / s

Mapping to σ_prior,m (bounded monotone)
σ_prior,m^2 = σ_min^2 + (σ_max^2 - σ_min^2) * sigmoid(κ * z_m)

Recommended defaults (conceptual starting points)
- Update every K_prior = 20 PPO updates
- EMA α = 0.05
- Bounds: σ_min = 0.01 (or 0.03), σ_max = 0.3 (or 1.0 depending on weight scale)
- κ = 1.0 (moderate sensitivity)

Operational constraints
- Do not backpropagate through σ_prior,m updates.
- σ_prior,m is held constant between meta-updates.
- Prefer using critic loss rather than policy loss to reduce noise.

Rationale
- Tasks that benefit from specialization get looser prior; tasks that don’t remain tightly regularized.
- This aligns with VarShare’s philosophy and is interpretable.

Expected outcomes / things to watch
- σ_prior,m should differ across tasks; outlier tasks should get larger σ_prior,m.
- σ_prior,m should not drift upward for all tasks simultaneously (if it does, bounds are too loose or signal biased).
- Sampling noise should reduce for tasks with small σ_prior,m.

Logging emphasis
- Log Δ_m, Δ̄_m, z_m and σ_prior,m per task over time.
- Correlate σ_prior,m with μ magnitude and task performance.

--------------------------------------------------------------------------------
EXPERIMENT 5: sigma_prior adaptive via Empirical Bayes gradient learning (your approach, with stabilizers)
--------------------------------------------------------------------------------

Goal
- Learn σ_prior (layer-wise) by gradient descent as part of the overall objective (Type-II ML / Empirical Bayes), rather than using a rule-based controller.

Core idea
- Treat log σ_prior,layer as a learnable parameter and include it in the gradient optimization of the objective.
- Add a weak hyperprior to prevent collapse/explosion.

Granularity
- Layer-wise σ_prior: one scalar parameter per layer (or per logical block of parameters that share a prior).
- The residual prior for that layer uses σ_prior,layer in the KL for all tasks m for residual dimensions in that layer.

Parameterization
- Introduce learnable parameters:
  log_sigma_prior,layer (trainable scalar)
- Convert for use:
  σ_prior,layer = exp(log_sigma_prior,layer)

Initialization
- Initialize all log_sigma_prior,layer to log(0.1) (uniform, no randomness).

KL usage
- In the KL(q||p) computation for residual dimensions in a given layer, replace fixed σ_prior with σ_prior,layer.

Important stabilizers (recommended for RL)
A. Time-scale separation
- Use a lower effective learning rate for log_sigma_prior parameters (e.g. by scaling their gradient or assigning them a separate optimizer group).
- Rationale: RL gradients are noisy; σ_prior should not chase noise.

B. Hyperprior penalty (“spring”)
Add to total objective:
Hyperprior_penalty = λ_hyper * sum_over_layers (log_sigma_prior,layer - log_sigma_target)^2
with:
- log_sigma_target = log(0.1)
- λ_hyper small (example order: 1e-3 to 1e-2; treat as tunable but small)

C. Optional but recommended: restrict what drives σ_prior
To improve interpretability and avoid σ_prior explaining posterior noise, consider learning σ_prior using gradients that depend primarily on μ rather than σ:
- Use detached σ in the KL gradient path for σ_prior, or otherwise reduce σ contribution to σ_prior updates.
This keeps σ_prior tied to systematic specialization rather than stochastic uncertainty.

Rationale
- Empirical Bayes equilibrium encourages σ_prior^2 ≈ average second moment of residuals; this yields automatic plasticity learning.
- Layer-wise learning captures that some layers should remain shared while others allow specialization.

Expected outcomes / things to watch
- σ_prior,layer should become small for layers where μ stays small (rigid sharing).
- σ_prior,layer may grow for layers where specialization is consistently beneficial.
- Watch for degeneracy:
  - Collapse: σ_prior,layer → 0 and stays there (over-sharing, no plasticity)
  - Explosion: σ_prior,layer → large, effectively removing KL

Logging emphasis
- Log σ_prior,layer over time (both raw and log).
- Log hyperprior penalty magnitude.
- Correlate σ_prior,layer with μ magnitudes per layer and with task performance.

--------------------------------------------------------------------------------
Notes on experimental comparison and interpretation
--------------------------------------------------------------------------------

1) Keep the baseline VarShare variant (no annealing, fixed σ_prior default) as a control for all comparisons.

2) Separate the conceptual purposes:
- Experiment 1 (σ_prior=0.01) tests “prior scale correction” alone.
- Experiment 2 (normal β annealing) tests “time-based KL scheduling”.
- Experiment 3 (triggered β) tests “feedback control to prevent destructive sampling noise”.
- Experiment 4 (rule-based adaptive σ_prior) tests “task-specific plasticity via performance-aligned diagnostic signal”.
- Experiment 5 (Empirical Bayes σ_prior) tests “learned layer-wise plasticity via ELBO optimization”.

3) Sampling policy must be explicit
Because posterior σ affects sampled weights, ensure the experiment configs specify:
- whether rollouts use sampled weights or posterior mean,
- whether sampling is restricted to adapters only or the full residual parameterization,
- and whether evaluation uses means (recommended).

4) If σ inflation to ~1 remains present
If posterior σ still drifts to ~1 in any variant and rollouts sample weights:
- expect performance degradation due to noisy action distributions.
This is not necessarily a bug; it may indicate mis-scaled priors or insufficient KL strength. The logs above should diagnose it.

End of spec.
