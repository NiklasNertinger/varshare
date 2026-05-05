# PPO Baselines for Meta-World MT10

## Executive conclusion

The literature is much sparser than it first appears if the requirement is **Meta-World MT10 plus PPO**. The best-known MT10 papers that reviewers often name-check—**Soft Modularization** and **CARE**—are in fact **SAC-based** on Meta-World, not PPO-based, and **MOORE** also uses **SAC for Meta-World** while using PPO only on MiniGrid. That means there are **not** three clean, recent, top-tier MT10 papers that all expose a fully specified PPO baseline with exact actor/critic architecture, optimiser settings, rollout parameters, and total steps. The strongest high-confidence PPO source I found is **GSL (ICML 2022)**; the original **Meta-World** benchmark is benchmark-defining but under-specifies PPO; **OPAL (ICLR 2021)** validates PPO on MT10/MT50 but also leaves important PPO details unstated. citeturn9view1turn8view1turn15view1turn23view0turn37view0

The practical consequence for your paper is straightforward: if you want a **reviewer-proof PPO baseline for MT10**, the safest choice is to **copy GSL’s reported PPO setup exactly where it is explicit**, and to be transparent about which fields are genuinely recoverable from the literature and which are not. I would **not** copy Soft Modularization or CARE hyperparameters for a PPO baseline, because doing so would mix **SAC-era engineering decisions** into an on-policy baseline. That is exactly the kind of mismatch reviewers notice. citeturn23view0turn9view1turn8view1

## What the MT10 literature actually supports

Soft Modularization states plainly that it trains with **Soft Actor-Critic**, compares against **MT-SAC** and **MT-MH-SAC**, and reports MT10/MT50 results in that SAC setting. CARE likewise states that for end-to-end RL it uses **SAC**, and its supplement gives SAC hyperparameters rather than PPO ones. MOORE is useful as a recent MT10 paper, but for Meta-World it explicitly switches to **SAC** and reserves PPO for MiniGrid. So if your goal is a defensible **PPO** baseline, these papers are valuable as MT10 context, but **not** as direct PPO architecture/hyperparameter sources. citeturn9view1turn8view1turn15view1

A second complication is that later Meta-World papers are not always directly comparable because of **versioning and reward-function drift**. The 2025 Meta-World+ benchmark paper explicitly argues that many reported results are clouded by undocumented benchmark changes and that V1 and V2 reward-function results are not directly comparable. For a reviewer-facing baseline section, it is therefore worth stating exactly **which paper and which configuration lineage** you copied, rather than saying only “PPO on MT10.” citeturn13view0

## Best fully specified PPO source

### Improving Policy Optimization with Generalist-Specialist Learning

This is the **single best source** I found if your objective is “copy a PPO MT10 baseline exactly enough that a reviewer cannot say the baseline was under-specified.” The paper evaluates PPO on **MT-10 and MT-50 from Meta-World**, reports train performance, and its appendix provides an unusually concrete hyperparameter table for Meta-World PPO. The official repository also states that the Meta-World experiments adapt **PPO from garage**, which strengthens the implementation traceability. citeturn23view0turn41search0

**Neural network architecture.** The appendix states that for both MT-10 and MT-50, the generalist and specialists are trained using PPO with a **policy network of two hidden layers, each of size 32**. The paper also specifies **ReLU** as the nonlinearity. What it does **not** explicitly disclose is a separate critic architecture; the appendix names only the policy network. So the exact actor architecture is recoverable as **[32, 32] with ReLU**, but the critic architecture is **not explicitly stated** in the paper. citeturn23view0

**PPO hyperparameters.** The appendix gives a single PPO learning rate of **2.5 × 10⁻⁴**, **γ = 0.99**, **GAE λ = 0.95**, **clip range = 0.2**, and entropy coefficient **0.005 for MT-10** and **0.05 for MT-50**. It also reports Gaussian policy standard-deviation bounds of **min std 0.5** and **max std 1.5**. The paper does **not** report separate actor and critic learning rates, so the exact defensible statement is that it reports a **single PPO learning rate** rather than distinct actor/critic rates. citeturn23view0

**Rollout and optimisation specifications.** The appendix reports **10 threads** for MT-10, **10⁵ samples per PPO epoch**, and **32 samples per minibatch**. Those are the exact rollout/minibatch figures it discloses. However, I did **not** find an explicit **update-epochs / k_epochs** value in the paper appendix. Since the repository says the implementation is adapted from **garage PPO**, there may be an implicit default in the underlying framework, but the paper itself does not pin that down explicitly enough for me to call it exact. citeturn23view0turn41search0

**Total environment steps.** The appendix explicitly states **2 × 10⁷ total simulation steps for MT-10** and **4 × 10⁷ for MT-50**. For your use case, this is the clearest top-tier PPO MT10 step budget in the literature I found. citeturn23view0

**What I would copy into your paper.** If you want a reviewer-resistant MT10 PPO baseline, I would copy GSL as: actor MLP **[32, 32]**, **ReLU**, PPO learning rate **2.5e-4**, **γ 0.99**, **λ 0.95**, **clip 0.2**, **entropy coefficient 0.005**, **10 threads**, **100k samples per PPO epoch**, **minibatch 32**, **20M total steps**, and I would add one sentence that **the critic architecture and PPO update-epochs count are not explicitly reported in the paper appendix**. That sentence protects you from overclaiming. citeturn23view0

## Benchmark-defining PPO source

### Meta-World benchmark paper

The original Meta-World benchmark paper remains authoritative because it **defines MT10/MT50** and directly discusses **multi-task PPO** as one of the canonical baselines. It reports that MT10 evaluates learning one policy over **10 tasks**, notes that single-task PPO can solve a large majority of individual tasks, and that **multi-task PPO** performs substantially worse than MT-SAC on MT10 and MT50. So from a benchmarking standpoint, this is the canonical source that justifies including **MT-PPO** at all. citeturn16search1turn19search1turn19search2

The problem is reproducibility detail. The paper itself, in the material I recovered, does **not** expose a full PPO recipe with exact learning rate, entropy coefficient, rollout size, minibatch size, or update-epochs count for the MT10 baseline. What does exist is the **official garage MT-PPO example**, which implements an MT10 example with actor hidden sizes **(64, 64)**, value-function hidden sizes **(32, 32)**, **tanh** activations, **discount 0.99**, **GAE λ 0.95**, **clip range 0.2**, **500 epochs**, and **batch size 1024**. citeturn44view0turn19search5

That example is extremely useful, but I would be careful with wording: it is an **official reference implementation example**, not an explicit claim that these are the exact hyperparameters used to produce the benchmark paper’s published MT10 numbers. I therefore would not cite it as “the paper baseline” without qualification. The honest phrasing is that it is the **canonical public implementation lineage** for MT-PPO in the Meta-World ecosystem. citeturn44view0turn19search5

For your extraction request, the defensible fields are therefore:

**Architecture.** Actor **[64, 64]**; value function **[32, 32]**. citeturn44view0

**Activation.** **Tanh** for both policy and value function. citeturn44view0

**PPO hyperparameters explicitly shown.** **γ = 0.99**, **GAE λ = 0.95**, **clip = 0.2**. **Actor LR, critic LR, ent_coef** are **not disclosed** in the example excerpt I recovered, and I did not recover a paper appendix specifying them. citeturn44view0turn16search1

**Rollout specs.** **500 epochs**, **batch size 1024** in the official example. **Update epochs / k_epochs** are **not explicitly shown** in the recovered documentation. citeturn44view0

**Total environment steps.** The recovered **example** implies **500 × 1024 = 512,000** environment steps if run unchanged, but that should be treated as the example schedule, **not** as a guaranteed reconstruction of the benchmark paper’s published MT10 baseline. The paper baseline’s exact step budget was **not explicitly recoverable** from the sources I gathered. citeturn44view0turn16search1

## Another top-tier PPO MT10 source

### OPAL

OPAL is important because it is an **ICLR 2021** paper that explicitly evaluates **PPO** and **PPO+OPAL** on **Meta-World MT10 and MT50**, and shows a large gain from temporal abstraction: **PPO 15.2 ± 4.8 on MT10** versus **PPO+OPAL 70.1 ± 4.3**. So this paper is a legitimate top-tier source showing that PPO was used on MT10 in the early Meta-World literature. citeturn37view0

For architecture, the appendix is better than the main paper. It states that the **task policy** used across environments is a fully connected network with **three hidden layers of size 256** and **ReLU** activation. For the Meta-World transfer setup it also states **c = 5** and **latent dimension Z = 8**, and that the task policy takes in the **task id** for multi-task transfer. Those are exact, paper-backed architectural details for the PPO+OPAL setup. citeturn37view0turn38view0turn38view1

However, OPAL is **not** a fully specified source for a vanilla PPO baseline. The appendix I recovered does **not** disclose the baseline PPO optimiser hyperparameters for MT10: I did not find exact **learning rate**, **clip coefficient**, **entropy coefficient**, **GAE λ**, **rollout length**, **minibatch size**, or **update-epochs count** for the MT10 PPO baseline itself. In other words, OPAL is authoritative for “PPO was evaluated on MT10 here” and for the **task-policy architecture**, but **not** for a perfect standalone vanilla-PPO reimplementation recipe. citeturn37view0turn38view0turn38view3

For your extraction table, the exact defensible entries are therefore:

**Architecture.** Task policy MLP **[256, 256, 256]**. citeturn38view0

**Activation.** **ReLU**. citeturn38view0

**PPO hyperparameters.** **Not explicitly disclosed** for the MT10 baseline in the recovered appendix. citeturn37view0turn38view3

**Rollout specs.** **Not explicitly disclosed** for the MT10 baseline in the recovered appendix. citeturn37view0turn38view3

**Total environment steps.** **Not explicitly disclosed** in the recovered source excerpt for MT10 PPO. citeturn37view0

## Recommended baseline choice for your paper

If your goal is to make the baseline section as hard as possible for reviewers to attack, I would write it this way:

Use **GSL (ICML 2022)** as the **primary PPO baseline specification** because it is the only top-tier MT10+PPO paper I found that reports almost all of the important knobs explicitly: actor architecture, nonlinearity, learning rate, clip, gamma, GAE lambda, entropy coefficient, rollout scale, minibatch size, and total MT10 step budget. Then add a short note that the paper does **not explicitly disclose the critic architecture or PPO update-epochs count**, and that those two fields are therefore left unchanged from the authors’ public implementation lineage rather than guessed. citeturn23view0turn41search0

If you want a second PPO reference point, use the **official garage MT10 example** as an **implementation-lineage cross-check**, not as a claimed reproduction of the Meta-World paper baseline. Its main value is that it gives you a canonical MT-PPO actor/value architecture and the standard Meta-World-facing PPO settings of **tanh, gamma 0.99, lambda 0.95, clip 0.2** within that codebase. citeturn44view0

I would explicitly **exclude Soft Modularization, CARE, and MOORE** from any “copied PPO hyperparameters” claim, because those papers are **SAC on Meta-World**. You can and should still compare your method against their **results** if you match their environment version and reward setting, but you should not present their optimisation settings as PPO baselines. citeturn9view1turn8view1turn15view1turn13view0

## Open questions and limitations

The main limitation is not lack of effort but lack of disclosure in the literature. I did **not** find three top-tier MT10 papers from 2020–2025 that all provide a fully explicit, exact vanilla-PPO recipe including actor and critic architecture, separate learning rates, clip, entropy, gamma, GAE lambda, rollout length, minibatch size, update epochs, and total step count. The literature is dominated by **SAC-based** MT10 work, while the PPO-on-MT10 papers are fewer and less completely specified. citeturn9view1turn8view1turn15view1turn23view0turn37view0

The two exact unresolved fields that matter most for a perfect reproduction are **critic architecture** and **PPO update-epochs / k_epochs** for GSL, plus nearly the whole PPO optimiser/rollout recipe for the Meta-World paper and OPAL baseline. If you want to be maximally conservative in your own manuscript, the cleanest wording is: “We adopt the exact published GSL MT10 PPO hyperparameters where explicitly reported, and we do not infer unpublished settings from unrelated SAC papers.” That is both true and reviewer-safe. citeturn23view0turn41search0