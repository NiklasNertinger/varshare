# VarShare Project — Comprehensive Context & Design Summary

> This document is a **conceptual, strategic, and organizational summary** of the VarShare project, reflecting the cumulative discussions, decisions, constraints, and goals agreed upon so far.  
>  
> It is **not** an implementation guide, does **not** contain code-level instructions, and is **not** an experiment report.  
>  
> Its purpose is to:
> - crystallize *what problem we are solving*,
> - clarify *what claims we aim to make*,
> - document *which baselines and paradigms we engage with*,
> - explain *why certain choices were made*,
> - and provide a stable mental model for anyone (human or AI) working on the project.

---

## 1. High-level goal of the project

The goal of this project is to develop, implement, and rigorously evaluate **VarShare**, a new method for **Multi-Task Reinforcement Learning (MTRL)**.

At a high level, the project aims to answer the following research question:

> *How can we enable principled task specialization in multi-task RL while preserving efficient parameter sharing, under realistic model capacity constraints?*

The intended output is a **research paper** supported by a clean, well-organized experimental codebase that:
- implements VarShare faithfully,
- evaluates it against strong and representative baselines,
- follows accepted community norms in RL experimentation,
- and yields results that are interpretable, defensible, and reviewer-robust.

This is **not** a benchmark-hacking effort, nor a brute-force scaling exercise. The focus is on *inductive bias*, *regularization*, and *principled architectural design*.

---

## 2. Conceptual position of VarShare in the MTRL landscape

### 2.1 The MTRL challenge

Multi-Task Reinforcement Learning faces a fundamental tension:
- **Sharing** improves data efficiency and generalization.
- **Specialization** is necessary when tasks conflict.

Classic failure modes in MTRL include:
- negative transfer between tasks,
- representation collapse toward a compromise solution,
- over-specialization that destroys transfer,
- heavy reliance on unconstrained task embeddings,
- or brute-force architectural scaling that hides interference rather than resolving it.

### 2.2 What VarShare is (conceptually)

VarShare introduces **task-specific residual adapters in weight space**, not activation space.

Key conceptual pillars:
- A **shared base network** captures common structure across tasks.
- Each task has a **task-specific residual parameterization** added to the shared weights.
- These residuals are **probabilistic**, not deterministic.
- A **KL-based regularizer** (derived from a variational objective) enforces *default sharing*.
- Specialization emerges **only when justified by the reward signal**.

Crucially, VarShare does *not* rely on:
- explicit routing,
- hard task partitions,
- large expert pools,
- or unconstrained task embeddings.

Instead, it frames task specialization as an **information-theoretic cost-benefit tradeoff**.

---

## 3. Philosophy of evaluation

### 3.1 What the paper does *not* aim to do

The project explicitly avoids:
- competing in the “largest model wins” regime,
- claiming universal dominance over *all* possible MTRL methods,
- or rebranding scaling tricks as algorithmic contributions.

The goal is **not** to beat massive architectures with small ones by brute force.

### 3.2 What the paper *does* aim to show

The evaluation is designed to support claims of the form:

> *VarShare outperforms representative methods across the dominant paradigms of modern MTRL, under realistic capacity and compute constraints.*

The emphasis is on:
- fairness,
- representativeness,
- and conceptual coverage rather than raw benchmark saturation.

---

## 4. Baseline taxonomy and selection rationale

Instead of comparing to *every* published MTRL method, the project adopts a **paradigm-based baseline strategy**.

Modern MTRL approaches cluster into a small number of dominant paradigms. The baseline list is chosen to cover each one.

### 4.1 Baseline list

The current agreed-upon baselines are:

1. **Single-task oracle**
   - One PPO agent trained per task.
   - Serves as an upper bound and calibration reference.

2. **Shared MTL with task embedding**
   - A single shared policy conditioned on task identity.
   - Represents the standard “vanilla” MTRL approach.

3. **PaCo (Parameter-Compositional MTRL)**
   - Represents *parameter subspace composition* approaches.
   - Task-specific coefficients combine shared basis parameters.

4. **Soft Modularization**
   - Represents *routing-based modular architectures*.
   - Tasks specialize via learned module routing.

5. **PCGrad**
   - Represents *gradient-level interference mitigation*.
   - Applied on top of the shared MTL baseline.

6. **VarShare**
   - Represents *variational weight-space regularization*.

Each baseline is included **not** because it is “new” or “trendy”, but because it represents a **distinct inductive bias** for addressing task interference.

### 4.2 What is deliberately excluded

The project intentionally excludes:
- extreme scaling approaches (e.g. massive task-conditioned critics),
- large mixture-of-experts systems requiring extensive tuning,
- purely meta-learning formulations (e.g. RL², MAML),
- obsolete or superseded methods.

These exclusions are deliberate, justified, and aligned with the scope of the claims.

---

## 5. Choice of RL backbone (PPO)

A central design decision is to use **PPO as the base algorithm for all methods**.

### Rationale:
- PPO is stable and well-understood.
- It supports both continuous and discrete control tasks.
- It avoids confounding effects from algorithm-specific tricks.
- All selected baselines can, in principle, be implemented on top of PPO.

This ensures that:
- algorithmic differences arise from *MTRL mechanisms*, not from differing RL backbones,
- fairness is easier to reason about,
- comparisons are easier to interpret.

---

## 6. Environments and benchmarks

### 6.1 Toy environments

Toy environments such as **CartPole** and **LunarLander** are used for:
- early sanity checks,
- debugging,
- validating correctness of the VarShare objective,
- developing intuition for metrics and diagnostics.

They are **not** used to support final claims.

### 6.2 Meta-World

**Meta-World** is the main benchmark suite.

Key points:
- MT10 is the primary evaluation suite.
- MT3 (or MT2) is used as a **smoke test**, not a predictor of final performance.
- Evaluation follows common Meta-World conventions (success rate, task-wise reporting).

Meta-World is chosen because it is:
- widely accepted in MTRL,
- task-diverse,
- challenging without being prohibitively expensive,
- and well-suited to studying both sharing and interference.

---

## 7. Evaluation philosophy and metrics

### 7.1 Primary axes

- **Environment steps** are the primary training budget axis.
- PPO update schedules are kept consistent across methods.
- Wall-clock time and gradient-step counts are logged as diagnostics.

### 7.2 Metrics

Metrics include:
- average performance across tasks,
- per-task performance distributions,
- robust aggregate measures (e.g. IQM),
- stability and variance across seeds.

For VarShare specifically, additional **diagnostic signals** are important:
- magnitude of task-specific residuals,
- KL regularization behavior,
- degree of effective specialization vs sharing.

These metrics are used primarily for *analysis and interpretation*, not leaderboard ranking.

---

## 8. Capacity and fairness considerations

A key concern throughout the project is **capacity fairness**.

VarShare introduces task-specific parameters, which raises valid concerns about:
- parameter count,
- memory footprint,
- and compute fairness.

The adopted principle is:
- comparisons should be matched by **active parameter count**,
- while also reporting total stored parameters for transparency.

This reflects a realistic notion of compute cost:
- only parameters involved in forward/backward passes matter for runtime,
- while stored but inactive parameters reflect memory rather than compute.

---

## 9. Experimental progression and risk management

The project follows a **stage-wise experimental plan** designed to minimize wasted effort:

1. Single-task PPO sanity checks
2. VarShare-only toy experiments
3. All baselines on toy MTL
4. Meta-World MT3 smoke tests
5. Full MT10 evaluations
6. LOO experiments
7. L2Soft vs VarShare ablations
8. Optional extension to additional benchmarks

This progression prioritizes:
- early detection of bugs,
- isolation of conceptual vs implementation errors,
- controlled complexity growth.

---

## 10. Role of the AI coding agent

The AI coding agent is treated as:
- an implementation partner,
- a productivity multiplier,
- but **not** an unquestioned authority.

Key expectations:
- the agent must explain its choices,
- must ask when decisions affect results,
- must challenge questionable assumptions,
- must prioritize clarity and cleanliness over speed.

The agent is instructed to:
- work within a structured repo,
- rely on provided context documents (especially `.tex` sources),
- and treat this as a serious research codebase, not a prototype dump.

---

## 11. Context documents and knowledge sources

The `context/` folder contains:
- the VarShare proposal (authoritative for the method),
- LaTeX source code for baseline papers (authoritative for competitors),
- legacy notes labeled as non-authoritative reference material.

These documents serve to:
- prevent reinvention,
- avoid misrepresenting baselines,
- and ensure faithful implementation and discussion.

---

## 12. Overall guiding principles

The project is guided by the following principles:

- **Conceptual clarity over brute force**
- **Representative coverage over exhaustive comparison**
- **Fairness and transparency over gaming metrics**
- **Clean engineering as a scientific enabler**
- **Interpretability as a first-class concern**

If these principles are upheld, the resulting work should:
- stand up to reviewer scrutiny,
- produce meaningful insights about MTRL,
- and form a solid foundation for future extensions.

---

## 13. Final note

This document should remain **stable** as the conceptual backbone of the project.

Implementation details, hyperparameters, scripts, and logs will evolve —  
but the **intent**, **scope**, and **standards** described here should not drift without explicit discussion.

Any proposed deviation from these principles should be treated as a **research decision**, not a coding detail.
