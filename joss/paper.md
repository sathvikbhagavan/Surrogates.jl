---
title: 'Surrogates.jl: Surrogate Modeling and Surrogate-Based Optimization in Julia'
tags:
  - Julia
  - Surrogate Modeling
authors:
  - name: Sathvik Bhagavan
    orcid: 0000-0003-0785-3586
    corresponding: true
    affiliation: 1
  - name: Ludovico Bessi
    affiliation: 4
  - name: Vikramaditya Narayan
    affiliation: 4
  - name: Christopher Rackauckas
    orcid: 0000-0001-5850-0663
    affiliation: "2, 3"
affiliations:
  - name: EPFL
    index: 1
  - name: Massachusetts Institute of Technology, USA
    index: 2
  - name: JuliaHub, USA
    index: 3
  - name: Google Summer of Code Contributor
    index: 4
date: 21 September 2026
bibliography: paper.bib
---

# Summary

Many scientific and engineering workflows depend on a function $f$ that is expensive to evaluate: a PDE solve, a finite-element simulation, or a physical experiment. *Surrogate modeling* replaces $f$ by a cheap approximation $g$ fitted to a small design of experiments, and then optimizes, explores, or differentiates $g$ instead.

`Surrogates.jl` gives a single Julia interface to a broad collection of surrogate families: radial basis functions, compactly supported Wendland kernels [@wendland1995], Kriging [@jones1998efficient] together with its gradient-enhanced (`GEK` [@han2013gek], `GEKPLS` [@bouhlel2019gekpls]) and partial-least-squares variants (`KPLS`, `KPLSK` [@bouhlel2016kpls]), multivariate adaptive regression splines [@friedman1991multivariate], Lobachevsky splines [@allasia2013lobachevsky], inverse-distance (Shepard) interpolation [@shepard1968], linear and second-order polynomial models, and multi-fidelity correction surrogates [@forrester2007multifidelity]. Package extensions add mixtures of experts [@jordan1994hierarchical], neural and gradient-enhanced neural networks [@bouhlel2020gann] via `Flux.jl` [@Fluxjl], support vector machines via `LIBSVM` [@chang2011libsvm], gradient-boosted trees via `XGBoost` [@chen2016xgboost], polynomial chaos expansions via `PolyChaos.jl` [@muehlpfordt2020polychaos], and general Gaussian processes via `AbstractGPs.jl` [@AbstractGPsjl].

Every surrogate is an ordinary Julia callable: `s(x)` evaluates the approximation, `update!(s, x, y)` incorporates new observations without a full refit where the algorithm allows, and models with a predictive distribution (the Kriging family and Gaussian processes) implement `std_error_at_point`. The initial design is drawn with any sampler from `QuasiMonteCarlo.jl` [@QuasiMonteCarlojl] — Sobol, Latin hypercube, Halton, and others — plus a `SectionSample` strategy that holds chosen coordinates fixed.

Beyond fitting, `Surrogates.jl` implements surrogate-based optimization through `surrogate_optimize!`: stochastic RBF search [@regis2007stochastic], DYCORS [@regis2013combining], lower confidence bound [@cox1992statistical; @srinivas2010gaussian] and expected improvement [@jones1998efficient] acquisition, the SOP algorithm for parallel search centers [@krityakierne2016sop], and two multi-objective loops — a surrogate-screened candidate selection (SMB) and the rolling tide evolutionary algorithm [@fieldsend2015rtea] for noisy objectives. Where $f$ can be evaluated at several points at once, constant-liar and Kriging-believer strategies [@ginsbourger2010kriging] propose a batch per surrogate update. Because fitted surrogates are plain Julia functions, they compose directly with `ForwardDiff.jl` [@revels2016forward] and `Zygote.jl` [@innes2018zygote], so a surrogate can act as a cheap, differentiable component inside a larger program with no glue code.

# Statement of need

Surrogate modeling and surrogate-based optimization are used across aerospace design, hyperparameter tuning, and simulation-based engineering, and mature toolkits exist outside Julia [@bouhlel2019python; @saves2024smt; @eriksson2019pysot]. Julia lacked a single package spanning this range under one interface, so users comparing, say, Kriging against radial basis functions against a neural network on the same problem had to learn several APIs or hand-assemble packages such as `GaussianProcesses.jl` [@GaussianProcessesjl] and `Flux.jl` [@Fluxjl] around their own optimization loop.

`Surrogates.jl` fills that gap for the Julia [@julia] and SciML [@rackauckas2017differentialequations] ecosystems, and reuses SciML infrastructure rather than reimplementing it: where a surrogate fits hyperparameters numerically, the fit is posed as an `OptimizationProblem` and solved through `Optimization.jl` [@optimizationjl]; sampling comes from `QuasiMonteCarlo.jl`; and the common interface (`parameters`, `hyperparameters`, `update!`) implements `SurrogatesBase.jl` [@SurrogatesBasejl], the abstract interface shared with the wider SciML surrogate ecosystem. Surrogates therefore slot into existing Julia optimization and machine learning pipelines instead of forming a closed toolbox.

# Functionality

A typical workflow has three stages:

1. **Sampling.** `sample(n, lb, ub, SobolSample())` builds the design of experiments with any `QuasiMonteCarlo.jl` sampler.
2. **Fitting.** `Kriging(x, y, lb, ub)` or `RadialBasis(x, y, lb, ub)` — all built-in surrogates share this convention regardless of input dimension (scalar or vector) or response type (scalar- or vector-valued), and accept `update!` for new observations.
3. **Optimization.** `surrogate_optimize!(f, alg, lb, ub, surrogate, sampler)` drives the surrogate/expensive-function loop, optionally proposing a batch per iteration.

Several surrogates exploit extra problem structure to reach a given accuracy in fewer evaluations of $f$: `GEK` and `GEKPLS` fold in derivative information, often available for free from adjoint-based PDE solvers, to raise accuracy per sample; `KPLS` and `KPLSK` fit correlation length scales in a PLS-reduced subspace to keep Kriging tractable in high input dimension; and `VariableFidelitySurrogate` corrects a cheap low-fidelity model with a small set of expensive high-fidelity samples, the common multi-fidelity setting in engineering design [@forrester2007multifidelity].

# Comparison with existing software

\autoref{tab:comparison} places `Surrogates.jl` among comparable open-source toolkits. In Python, SMT covers the Kriging family with derivatives and mixed-variable support and provides EGO with EI, SBO and LCB infill criteria and qEI batching [@bouhlel2019python; @saves2024smt]; pySOT pairs RBF, GP, MARS and polynomial surrogates with the SRBF, DYCORS, SOP, EI and LCB strategies and asynchronous parallel evaluation [@eriksson2019pysot]; and BoTorch, with Ax, builds Monte-Carlo acquisition functions on PyTorch, including multi-objective batching [@balandat2020botorch; @daulton2020ehvi]. In R, DiceKriging and DiceOptim cover Kriging metamodeling and EGO with noisy, constrained and parallel variants [@roustant2012dicekriging; @DiceOptimCRAN], while mlrMBO makes the surrogate and infill criterion swappable over any `mlr` learner [@bischl2017mlrmbo]. Dakota embeds surrogate-based optimization in a large simulation framework alongside uncertainty quantification and sensitivity analysis [@adams2020dakota]. Within Julia, `BayesianOptimization.jl` fits GP surrogates with expected improvement, probability of improvement, upper confidence bound and Thompson sampling [@BayesianOptimizationjl], and `GaussianProcesses.jl` [@GaussianProcessesjl] and `AbstractGPs.jl` [@AbstractGPsjl] supply the GP models themselves without an optimization loop.

Table: Comparison of open-source surrogate modeling toolkits. \label{tab:comparison}

| Package | Language | Gradient-enhanced Kriging | Surrogate-based optimization | Multi-objective | AD through the surrogate |
|---|---|---|---|---|---|
| `Surrogates.jl` (this work) | Julia | GEK, GEKPLS | SRBF, DYCORS, SOP, EI, LCBS | SMB, RTEA | ForwardDiff, Zygote |
| SMT [@bouhlel2019python; @saves2024smt] | Python | yes | EGO (EI, SBO, LCB), qEI batch | no | analytic derivatives |
| pySOT [@eriksson2019pysot] | Python | no | SRBF, DYCORS, SOP, EI, LCB | no | no |
| BoTorch / Ax [@balandat2020botorch] | Python | no | MC acquisition, batch | yes | PyTorch |
| DiceOptim [@roustant2012dicekriging; @DiceOptimCRAN] | R | no | EGO, noisy and parallel variants | no | no |
| `BayesianOptimization.jl` [@BayesianOptimizationjl] | Julia | no | EI, PI, UCB, Thompson | no | no |

The contribution of `Surrogates.jl` is coverage rather than a new algorithm. It is the only Julia package spanning classical interpolation, the Kriging family with gradient enhancement and PLS dimension reduction, splines, and machine-learning surrogates behind one API, and it brings together in a single package the RBF-based global optimizers associated with pySOT, Kriging-family acquisition functions, and multi-objective surrogate loops.

# Acknowledgements

`Surrogates.jl` is developed as part of the SciML organization. We thank everyone who has contributed code, documentation, and bug reports through GitHub.

# References
