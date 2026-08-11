"""
    RMTCSurrogate(x, y, lb, ub; num_elements = 4, energy_weight = 1.0e-4,
        regularization_weight = 1.0e-10)

Regularized minimal-energy tensor-product cubic Hermite spline (RMTC) surrogate.

`RMTCSurrogate` fits a tensor-product grid of cubic Hermite splines to scattered
data by regularized least squares: the nodal function values are chosen to
minimize the fit residual plus a curvature-energy penalty, so the surrogate
approximates (rather than interpolates) the training data and stays smooth
between sparsely sampled regions. Nodal slopes are not free parameters; each
one is the finite-difference estimate from its neighboring nodal values, so
the whole fit reduces to a single (regularized) linear least-squares solve.
The fitted surrogate is callable as `rmtc(x_new)` and can be updated with
`update!(rmtc, x_new, y_new)`.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `grids`: per-dimension tuple of uniform grid node coordinates.
  - `derivs`: per-dimension tuple of finite-difference operators mapping nodal
    values to nodal slopes.
  - `coeff`: fitted nodal function values, stored as an array shaped like the
    tensor-product grid.
  - `num_elements`: number of grid intervals per dimension.
  - `energy_weight`: curvature-energy regularization weight.
  - `regularization_weight`: ridge term guarding the normal-equations solve.

# Arguments

  - `x`: sample locations. Use scalars for one-dimensional inputs or tuples for
    multidimensional inputs.
  - `y`: observed values at `x`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `num_elements`: number of grid intervals per dimension. Either an integer
    (used for every dimension) or a per-dimension collection of integers. The
    number of grid degrees of freedom grows as `prod(num_elements .+ 1)`, so
    keep this modest in higher dimensions.
  - `energy_weight`: weight applied to the curvature-energy regularization
    term. Larger values produce smoother, less locally reactive surfaces.
  - `regularization_weight`: small ridge term added to the normal equations to
    guard against a singular solve.

# Returns

An `RMTCSurrogate` that satisfies the generic surrogate interface:
`surrogate(x)` evaluates the approximation and `update!(surrogate, x, y)`
refits after adding a sample.
"""
mutable struct RMTCSurrogate{X, Y, L, U, G, D, C, N, EW, RW} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    grids::G
    derivs::D
    coeff::C
    num_elements::N
    energy_weight::EW
    regularization_weight::RW
end

_rmtc_point(val::Number) = (val,)
_rmtc_point(val) = Tuple(val)

function _rmtc_grid(lb::Number, ub::Number, num_elements::Integer)
    return collect(range(lb, ub, length = num_elements + 1))
end

# Finite-difference operator mapping nodal values to nodal slopes: central
# differences in the interior, one-sided differences at the two ends. This is
# what keeps the whole RMTC fit linear in the nodal values alone (no separate
# slope degrees of freedom to solve for).
function _rmtc_deriv_operator(t::AbstractVector)
    n = length(t)
    D = zeros(eltype(t), n, n)
    n == 1 && return D
    D[1, 1] = -1 / (t[2] - t[1])
    D[1, 2] = 1 / (t[2] - t[1])
    D[n, n - 1] = -1 / (t[n] - t[n - 1])
    D[n, n] = 1 / (t[n] - t[n - 1])
    for i in 2:(n - 1)
        D[i, i - 1] = -1 / (t[i + 1] - t[i - 1])
        D[i, i + 1] = 1 / (t[i + 1] - t[i - 1])
    end
    return D
end

# Used only while fitting (building the design/energy matrices below), where
# the nodal values are still unknowns being solved for, so each training
# point needs its contribution as a linear functional of those unknowns.
# DataInterpolations.CubicHermiteSpline can't serve that role: it evaluates a
# spline from already-known (u, du) data and doesn't expose the H00/H10/H01/H11
# blending weights as a reusable primitive. Evaluation of a *fitted* surrogate
# (`_rmtc_eval` below) has no such need and is delegated to it entirely.
function _hermite_weights(s)
    H00 = 2s^3 - 3s^2 + 1
    H10 = s^3 - 2s^2 + s
    H01 = -2s^3 + 3s^2
    H11 = s^3 - s^2
    return H00, H10, H01, H11
end

function _rmtc_find_cell(t::AbstractVector, x)
    n = length(t)
    x <= t[1] && return 1
    x >= t[n] && return n - 1
    i = searchsortedlast(t, x)
    return clamp(i, 1, n - 1)
end

# Basis-weight row r such that r' * w gives the tensor-product-Hermite value at
# x for a dimension whose nodal values are w (nodal slopes are D * w).
function _hermite_row(t::AbstractVector, D::AbstractMatrix, x)
    n = length(t)
    i = _rmtc_find_cell(t, x)
    h = t[i + 1] - t[i]
    s = (x - t[i]) / h
    H00, H10, H01, H11 = _hermite_weights(s)
    r = zeros(promote_type(eltype(t), typeof(x)), n)
    r[i] += H00
    r[i + 1] += H01
    r .+= (H10 * h) .* @view(D[i, :])
    r .+= (H11 * h) .* @view(D[i + 1, :])
    return r
end

function _rmtc_row(grids::NTuple{d, TG}, Ds::NTuple{d, TD}, x::NTuple{d, TX}) where {d, TG, TD, TX}
    rows = ntuple(k -> _hermite_row(grids[k], Ds[k], x[k]), d)
    shaped = ntuple(k -> reshape(rows[k], ntuple(j -> j == k ? length(rows[k]) : 1, d)), d)
    outer = reduce(.*, shaped)
    return vec(outer)
end

function _rmtc_design_matrix(x, grids::NTuple{d, TG}, Ds::NTuple{d, TD}) where {d, TG, TD}
    n = length(x)
    N = prod(length.(grids))
    X = zeros(eltype(grids[1]), n, N)
    for a in 1:n
        X[a, :] .= _rmtc_row(grids, Ds, _rmtc_point(x[a]))
    end
    return X
end

# Euler-Bernoulli beam-element stiffness matrix (EI = 1): closed-form
# integral of (d^2/dx^2)^2 over a cell for local DOFs [u_i, du_i, u_{i+1}, du_{i+1}].
function _hermite_energy_local(h)
    return (1 / h^3) .* [
        12.0 6h -12.0 6h
        6h 4h^2 -6h 2h^2
        -12.0 -6h 12.0 -6h
        6h 2h^2 -6h 4h^2
    ]
end

function _energy_matrix_1d(t::AbstractVector, D::AbstractMatrix)
    n = length(t)
    H = zeros(eltype(t), n, n)
    for i in 1:(n - 1)
        h = t[i + 1] - t[i]
        Kloc = _hermite_energy_local(h)
        P = zeros(eltype(t), 4, n)
        P[1, i] = 1
        P[2, :] .= @view(D[i, :])
        P[3, i + 1] = 1
        P[4, :] .= @view(D[i + 1, :])
        H .+= P' * Kloc * P
    end
    return H
end

# Sum of the one-directional curvature energies, each broadcast across the
# tensor grid via a Kronecker product against identities in the other
# dimensions (dimension 1 is the fastest-varying / rightmost factor, matching
# Julia's column-major `vec` of an ND array).
function _rmtc_energy_matrix(grids::NTuple{d, TG}, Ds::NTuple{d, TD}) where {d, TG, TD}
    ns = length.(grids)
    N = prod(ns)
    T = eltype(grids[1])
    H = zeros(T, N, N)
    for k in 1:d
        mats = ntuple(
            j -> j == k ? _energy_matrix_1d(grids[k], Ds[k]) : Matrix{T}(I, ns[j], ns[j]), d
        )
        H .+= reduce(kron, reverse(mats))
    end
    return H
end

function _rmtc_fit(x, y, grids::NTuple{d, TG}, Ds::NTuple{d, TD}, energy_weight, regularization_weight) where {d, TG, TD}
    X = _rmtc_design_matrix(x, grids, Ds)
    H = _rmtc_energy_matrix(grids, Ds)
    N = size(X, 2)
    A = Symmetric(X' * X .+ energy_weight .* H .+ regularization_weight .* I(N))
    b = X' * collect(y)
    wvec = A \ b
    return reshape(wvec, length.(grids))
end

function _rmtc_num_elements(num_elements::Integer, d)
    return ntuple(_ -> num_elements, d)
end
function _rmtc_num_elements(num_elements, d)
    return ntuple(k -> num_elements[k], d)
end

# Shared by both outer constructors below: turn per-dimension bounds and
# `num_elements` into grids/derivative-operators, then fit.
function _rmtc_construct(
        x, y, lbs::NTuple{d, TL}, ubs::NTuple{d, TU}, num_elements_tup::NTuple{d, TN},
        energy_weight, regularization_weight
    ) where {d, TL, TU, TN}
    grids = ntuple(k -> _rmtc_grid(lbs[k], ubs[k], num_elements_tup[k]), d)
    Ds = ntuple(k -> _rmtc_deriv_operator(grids[k]), d)
    coeff = _rmtc_fit(x, y, grids, Ds, energy_weight, regularization_weight)
    return grids, Ds, coeff
end

function RMTCSurrogate(
        x, y, lb::Number, ub::Number; num_elements::Integer = 4,
        energy_weight::Number = 1.0e-4, regularization_weight::Number = 1.0e-10
    )
    grids, Ds, coeff = _rmtc_construct(
        x, y, (lb,), (ub,), (num_elements,), energy_weight, regularization_weight
    )
    return RMTCSurrogate(
        x, y, lb, ub, grids, Ds, coeff, num_elements, energy_weight, regularization_weight
    )
end

function RMTCSurrogate(
        x, y, lb, ub; num_elements = 4,
        energy_weight::Number = 1.0e-4, regularization_weight::Number = 1.0e-10
    )
    d = length(lb)
    num_elements_tup = _rmtc_num_elements(num_elements, d)
    grids, Ds, coeff = _rmtc_construct(
        x, y, Tuple(lb), Tuple(ub), num_elements_tup, energy_weight, regularization_weight
    )
    return RMTCSurrogate(
        x, y, lb, ub, grids, Ds, coeff, num_elements_tup, energy_weight, regularization_weight
    )
end

# Evaluate the fitted tensor-product Hermite surface at x by collapsing one
# grid dimension at a time: axis 1 of the current array A is interpolated
# (via a cubic Hermite spline through its nodal values and finite-difference
# slopes) and dropped, leaving an array over the remaining axes. After d
# collapses only a scalar remains. Once the nodal values are fixed numbers,
# each collapse step is exactly a cubic Hermite spline evaluation through the
# Matrix-data form of DataInterpolations.CubicHermiteSpline, which returns a
# length-`rest` vector that reshapes back into the remaining axes (an empty
# shape, i.e. a 0-d array, once the last axis is collapsed).
function _rmtc_eval(grids::NTuple{d, TG}, Ds::NTuple{d, TD}, W::AbstractArray, x::NTuple{d, TX}) where {d, TG, TD, TX}
    A = W
    for k in 1:d
        n_k = size(A, 1)
        rest = length(A) ÷ n_k
        Amat = reshape(A, n_k, rest)
        DUmat = Ds[k] * Amat
        spline = CubicHermiteSpline(permutedims(DUmat), permutedims(Amat), grids[k])
        A = reshape(spline(x[k]), size(A)[2:end])
    end
    return A[]
end

function (rmtc::RMTCSurrogate)(val)
    _check_dimension(rmtc, val)
    return _rmtc_eval(rmtc.grids, rmtc.derivs, rmtc.coeff, _rmtc_point(val))
end

function SurrogatesBase.update!(rmtc::RMTCSurrogate, x_new, y_new)
    rmtc.x = vcat(rmtc.x, x_new)
    rmtc.y = vcat(rmtc.y, y_new)
    rmtc.coeff = _rmtc_fit(
        rmtc.x, rmtc.y, rmtc.grids, rmtc.derivs, rmtc.energy_weight, rmtc.regularization_weight
    )
    return nothing
end
