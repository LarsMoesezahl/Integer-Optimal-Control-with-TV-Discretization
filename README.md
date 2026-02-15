# TV-Regularized Integer Optimal Control

### Sequential Linear Integer Programming with Lazy Constraint Callback

This repository contains the implementation accompanying the master thesis:

> *Integrating Total Variation Discretization into Integer Optimal Control*  
> Lars Moesezahl, TU Dortmund, 2025

The code implements a **Sequential Linear Integer Programming (SLIP)** trust-region framework for solving integer-valued PDE-constrained optimization problems regularized by total variation (TV).

This repository focuses exclusively on the **callback-based outer-approximation enhancement** of the TV term.  
The patch-based domain decomposition approach from the thesis is intentionally **not included** here.

## Problem Setting

We consider integer optimal control problems of the form

$$
\min_{v \in L^2(\Omega)}
J(v) = F(v) + \omega \, TV(v)
$$

subject to

$$
v(x) \in \mathcal{E} \subset \mathbb{Z}
\quad \text{a.e. in } \Omega,
$$

where:

- $F(v)$ is typically a tracking-type functional composed with a PDE solution operator,
- $TV(v)$ enforces compactness and prevents oscillatory chattering controls,
- $\mathcal{E}$ is a finite integer set.

Total variation regularization restores existence of minimizers in function space and penalizes perimeter of level sets.

The theoretical foundation (existence, BV compactness, L-stationarity, and $\Gamma$-convergence) follows the framework developed in the thesis.

## Algorithmic Framework

### Sequential Linear Integer Programming (SLIP)

The optimization is solved with a **trust-region method in function space**:

1. Linearize $F$ at the current iterate $\bar v$.
2. Keep the TV term exact.
3. Solve the trust-region subproblem:

$$
\min_v \, (\nabla F(\bar v), v - \bar v) + \omega TV(v)
$$
$$
\text{s.t. } |v - \bar v|_{L^1} \le \Delta,
\quad v(x) \in \mathcal{E}
$$

The discretized subproblem yields a **Mixed-Integer Linear Program (MILP)**.

## Callback-Based Outer Approximation (Core Contribution)

A key computational bottleneck is the discretized total variation term, represented through a Raviart-Thomas dual formulation and associated constraints.

Instead of adding all TV constraints upfront, this implementation uses:

### Lazy Constraint Callback (Gurobi)

- Start with a relaxed MILP.
- Detect violated TV constraints at incumbent solutions.
- Add them dynamically via Gurobi lazy constraints.
- Repeat until no relevant violations remain.

This outer-approximation strategy:

- reduces initial model size,
- avoids unnecessary constraints,
- preserves exactness of the discretized TV model,
- improves runtime in practice.

This corresponds to the on-demand lazy-constraint enhancement described in Chapter 3 of the thesis.

## Discretization Strategy

The implementation follows the dual-mesh construction from the thesis:

- Fine mesh: integer-valued piecewise-constant controls.
- Coarse mesh: discretized TV via lowest-order Raviart-Thomas elements.

Mesh sizes are superlinearly coupled to support:

- $\Gamma$-convergence of the discretized TV,
- recovery of the correct total variation in the limit,
- preservation of integer feasibility.

This construction mitigates checkerboard null-space effects of naive discretizations.

## Implementation Overview

| Module | Purpose |
| --- | --- |
| `trust_region_solver.py` | Outer SLIP trust-region loop |
| `trust_region_subproblem.py` | Trust-region subproblem coordinator |
| `mip_subproblem.py` | MILP trust-region subproblem |
| `rt0_dual_solver.py` | Discretized TV dual formulation |
| `opt_tv_base.py` | TV operators and mesh coupling |
| `opt_blur.py` | Objective and adjoint gradient |
| `bvp_blur.py` | FEM state and adjoint solves (FEniCSx) |
| `synthetic_problem.py` | Benchmark problem construction |
| `run_experiment.py` | Main entry point |

## Numerical Experiments

The thesis evaluates:

- exact-solution validation,
- parameter sensitivity,
- runtime and objective behavior under mesh refinement.

The callback-based outer approximation shows:

- reduced runtime,
- maintained objective quality,
- preserved integer feasibility,
- better scalability than full upfront TV-constraint assembly.

### Covered Numerical Experiment (Solved Problem)

For our first numerical experiment, we consider problem $(P)$ with the
construction discussed in Section 3.1 of the thesis, with
$F : L^2(\Omega)\to\mathbb{R}$ defined by
$F(v)=\frac{1}{2}\lVert S(v+f)-y_d\rVert_{L^2(\Omega)}^2$,
where $f,y_d\in L^2(\Omega)$.

The solution operator $S : L^2(\Omega)\to H_0^1(\Omega)$ is defined by:
find $y=S(w)\in H_0^1(\Omega)$ such that

$$
-\beta\,\Delta y + y = w
\quad\text{in }\Omega,
\qquad
y = 0
\quad\text{on }\partial\Omega,
$$

with fixed $\beta>0$ and
$p_1=\frac{1}{32}$, $p_2=\frac{31}{32}$.

Therefore, we solve on $\Omega=(0,2)^2$:

$$
\min_{v\in L^2(\Omega)}
\frac{1}{2}\lVert y-y_d\rVert_{L^2(\Omega)}^2 + \alpha\,TV(v)
$$
$$
\text{s.t.}\quad
\begin{cases}
-\,\beta\,\Delta y + y = v + f & \text{in }\Omega,\\
y = 0 & \text{on }\partial\Omega,\\
v(x)\in \Lambda=\{0,1\} & \text{for a.e. }x\in\Omega,
\end{cases}
$$

with $\alpha>0$ and known optimal solution
$D=D_{0.5}((1,1)^\top)\subset\Omega$:

$$
\bar v(x)=
\begin{cases}
1, & x\in D,\\
0, & x\in \Omega\setminus D.
\end{cases}
$$

## Installation

### Option 1 - pip

```bash
pip install -r requirements.txt
```

### Option 2 - conda

```bash
conda env create -f environment.yml
conda activate tv-pde-inverse-problem
```

Requirements:

- Python 3.10+
- FEniCSx / dolfinx
- PETSc / petsc4py
- Gurobi (with valid license)

## Reproduction

```bash
./reproduce.sh
```

or

```bash
make reproduce
```

## Limitations

- Gurobi license required.
- FEniCSx + MPI setup can be platform-sensitive.
- Fine meshes lead to large MILPs.
- Full experiments are computationally expensive.

## Citation

If you use this repository, please cite:

- the associated master thesis,
- this software repository (see `CITATION.cff`).

## License

MIT License.
