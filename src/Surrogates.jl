module Surrogates

using LinearAlgebra
using Distributions
using Sobol
using LatinHypercubeSampling

abstract type AbstractSurrogate <: Function end

include("Radials.jl")
include("Kriging.jl")
include("Sampling.jl")
include("Optimization.jl")

export Kriging, RadialBasis, add_point!, current_estimate, std_error_at_point
export sample, GridSample, UniformSample, SobolSample, LatinHypercubeSample, LowDiscrepancySample
export SRBF,LCBS,EI,DYCORS,surrogate_optimize

end
