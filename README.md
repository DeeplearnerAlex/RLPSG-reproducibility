# RLPSG implementation and pretrained policies

This repository provides method implementations and three pretrained, target-specific RLPSG actors for an 8 by 8 bin grid. The top-level example generates a new synthetic upper-level allocation run using the selected frozen actor and current-HD terminal switching.

**This repository does not distribute the manuscript raw trial data, baseline comparisons, or audit material.** The example generates its own inputs and outputs; it is not a reproduction of the manuscript benchmark records.

## Repository contents

- policy.py loads an actor-only model and its frozen observation normalization.
- demo_allocation.py demonstrates bin-level allocation with a synthetic concentrated initial population.
- models/ring.npz, models/letter_A.npz, and models/separated.npz contain the three target-specific actors.
- models/models.json describes the exported parameters and model interface.
- src/ contains production implementation modules, including terminal allocation and lower-level motion planning and safety filtering, for integration.
- environment-reference.txt records the original author-side numerical environment. It is not the dependency list for the NumPy demonstration.

The exported NPZ models contain actor parameters, the target distribution, edge ordering, and observation-normalization statistics. They do not contain a PPO critic, optimizer state, or training-resumption state. A different model is used for each target; these models are not a single target-conditioned policy.

## Installation

Python 3.10 is recommended. Create and activate a virtual environment, then install the CPU dependencies. NumPy and SciPy support the allocation example; OSQP supports lower-level safety-filter integration:

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
~~~

Activate the environment using the normal command for your operating system before the second command. The default example does not require JAX, CUDA, Flax, a GPU, or compiled motion kernels.

The author-side full environment is recorded separately in environment-reference.txt. Those reference versions include packages used by training or lower-level experiments, not just packages required by the top-level demonstration.

## Generate a new allocation example

Run from the repository root:

~~~bash
python demo_allocation.py --help
python demo_allocation.py --target ring --robots 2000 --initial-bin 0 --steps 30 --seed 0 --output demo_outputs/ring
~~~

Use letter_A or separated instead of ring to select another target-specific model:

~~~bash
python demo_allocation.py --target letter_A --steps 30 --seed 0 --output demo_outputs/letter_A
python demo_allocation.py --target separated --steps 30 --seed 0 --output demo_outputs/separated
~~~

An output directory must be new. Without --output, the demonstration prints its summary without saving a run. The optional output includes a JSON summary and a new NPZ history containing bin counts, HD, inter-bin changes, and mode selections. These are newly generated example outputs, not supplied manuscript data. Use --models-dir to select a different directory containing the three exported model files.

The initial population is concentrated in the selected bin. The example evolves bin assignments; it does not simulate robot trajectories, obstacles, sensing errors, communication disturbances, or the lower-level safety filter.

## Load a frozen actor in Python

~~~python
from pathlib import Path
import numpy as np
from policy import MeanActionPolicy

policy = MeanActionPolicy(Path("models/ring.npz"))
density = np.zeros(64, dtype=np.float64)
density[0] = 1.0

action = policy.mean_action(density)
transition = policy.transition_matrix(density)

print(action.shape)                 # (274,)
print(transition.shape)             # (64, 64)
print(transition.sum(axis=0))       # column sums
~~~

The input is a nonnegative, normalized 64-bin density vector. The 65-component actor observation contains that density followed by the training-compatible smoothed HD feature, not an L1-error feature. The policy applies the frozen observation normalization and evaluates the actor mean. The architecture uses three 512-unit hidden layers with LayerNorm and GELU. Current-HD branch selection separately uses unsmoothed float64 HD.

The transition matrix is column stochastic: column j gives destination probabilities for robots currently in bin j. The stored target is available as policy.target; policy.target32 retains the float32 target used for numerical gating.

NumPy CPU inference is not guaranteed to be bitwise identical to the archived JAX/CUDA implementation. Floating-point and BLAS differences can change near-zero directed weights and subsequent sampled paths; the default example is not a replay of the original manuscript trials.

## Allocation rule

Ordinary allocation uses the target-specific PPO actor and the source and receiving restrictions. Receiving into positive-target bins is permitted while they are deficient, and zero-target bins can serve as transit bins.

At each allocation endpoint, current HD selects the branch:

- HD at or above 0.05 uses ordinary probabilistic allocation.
- HD below 0.05 uses zero-target cleanup together with PPO redistribution between neighboring positive-target bins.
- If HD rises back to the threshold, ordinary allocation resumes; the rule is not a permanent first-entry latch.

When population changes are introduced in another application, apply the event and recompute the current density before selecting the branch. Release obsolete cleanup state on exit and initialize it from the current state on re-entry.

The command-line allocation demo operates on the bin graph. Relative-position adjustment and lower-level physical execution require additional robot-state inputs and are not exercised by that demo.

## Position-aware target-bin sampling

The upper-level method in src/relative_position.py provides sample_goals(points, matrix, seed, allocation_step, quadrature_side=48). It fits the spatial-average balancing coefficients K and samples goals from current relative positions. The following optional example creates its own positions and does not load experimental records:

~~~python
from pathlib import Path
import sys
import numpy as np
from policy import MeanActionPolicy

sys.path.insert(0, str(Path("src").resolve()))
from relative_position import sample_goals

policy = MeanActionPolicy("models/ring.npz")
points = np.array([[20.0, 30.0], [40.0, 50.0]])  # both inside bin 0
density = np.zeros(64)
density[0] = 1.0
matrix = policy.transition_matrix(density)
goals, fit_residual = sample_goals(points, matrix, seed=0,
                                 allocation_step=0, quadrature_side=48)
~~~

Positions use the 800 by 800 metre domain with 100 metre bins. The returned values are assigned bin indices and the largest balancing-fit residual, not physical trajectories or an experimental validation result.

## Lower-level implementation modules

The src directory includes lower-level numerical implementations for applications that supply robot positions, velocities, assignments, workspace geometry, and model parameters. It is available for integration, not a turnkey full physical closed-loop demonstration.

Lower-level safety-filter calls use OSQP 1.1.3, included in requirements.txt.

Some accelerated paths require the companion C++ kernels to be compiled explicitly. They are not required for demo_allocation.py. The fixed build helper supports five targets:

| Source | Build target |
| --- | --- |
| accelerated_arc_batch_kernel.cpp | accelerated_arc_batch_kernel |
| command_projection.cpp | command_projection_kernel |
| warm_command_projection.cpp | warm_command_projection_kernel |
| native_voronoi_centroids.cpp | native_voronoi_centroids_kernel |
| nonideal_prediction_native.cpp | nonideal_prediction_native_kernel_20261105 |

List the source/output pairs or compile one target with a C++17 compiler:

~~~bash
python build_native.py --list
python build_native.py --target warm_command_projection_kernel
~~~

Running build_native.py without --target builds all five. The helper selects .so on Linux and .dll on Windows and disables FMA contraction. Do not enable fast-math. Missing native libraries do not imply a complete automatic Python fallback for every lower-level path. Numerical behavior and timing on a different compiler, platform, or configuration should be checked for that integration.

## Scope

This is a code and pretrained-model distribution. It provides a runnable upper-level synthetic demonstration and lower-level method implementations for integration. It does not supply the manuscript benchmark dataset or claim that running the example reconstructs the manuscript figures, comparisons, or full physical validation.

