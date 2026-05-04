# Core Agent Rules

These rules apply universally to any coding or implementation task you undertake in this workspace.

## 1. The "Zero-Prompt" Testing Mandate
You must **never** consider an implementation task complete without independently testing it first. 
I do not read code, so you are entirely responsible for the structural and mathematical soundness of your implementations.

- **Autonomy in Testing**: You must autonomously decide the appropriate level of testing based on the complexity of your changes. 
- **Methodological Sanity Checks**: For RL algorithms, do not just check if the code runs without crashing. You must design and execute conceptual sanity checks. Examples include (but are not limited to):
  - Verifying gradients are flowing to the correct layers and are not `None`.
  - Checking that the loss is actually decreasing on a 1-step dummy environment.
  - Ensuring rewards, advantages, and policy ratios remain within expected bounds without producing `NaN`s.
- **Reporting Failures**: If a methodological sanity check fails (e.g., the algorithm mathematically does not behave as expected), do not silently hack around it. Stop, report the failure to me, explain the conceptual issue, and propose solutions.

## 2. The Architectural Map
Since I rely on you for implementation, I need to know *where* the core logic lives without digging through the codebase.

- Whenever you build a new feature or modify an existing one, your final response must include an **Architectural Map**.
- The map must briefly explain the data/information flow and high-level architecture.
- It must explicitly list the **core files and exact line numbers** where the central mathematical or algorithmic logic resides, so I can review those specific segments if I choose to.
