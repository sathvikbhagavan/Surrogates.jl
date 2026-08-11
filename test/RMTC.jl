using Surrogates

#1D
lb = 0.0
ub = 5.0
n = 25
x = sample(n, lb, ub, SobolSample())
f = x -> 2 * x + x^2
y = f.(x)
my_rmtc1d = RMTCSurrogate(x, y, lb, ub, num_elements = 8, energy_weight = 1.0e-6)
val = my_rmtc1d(3.0)
@test isapprox(val, f(3.0), atol = 0.5)
update!(my_rmtc1d, 4.5, f(4.5))
@test isapprox(my_rmtc1d(4.5), f(4.5), atol = 0.5)

# Test that input dimension is properly checked for 1D RMTC surrogates
@test_throws ArgumentError my_rmtc1d(Float64[])
@test_throws ArgumentError my_rmtc1d((2.0, 3.0, 4.0))

#ND
lb = [0.0, 0.0]
ub = [10.0, 10.0]
n = 60
x = sample(n, lb, ub, SobolSample())
f = x -> x[1]^2 + x[2]^2 + x[1] * x[2]
y = f.(x)
my_rmtcnd = RMTCSurrogate(x, y, lb, ub, num_elements = 5, energy_weight = 1.0e-6)
val = my_rmtcnd((3.0, 4.0))
@test isapprox(val, f((3.0, 4.0)), atol = 5.0)
update!(my_rmtcnd, (5.0, 5.0), f((5.0, 5.0)))
@test isapprox(my_rmtcnd((5.0, 5.0)), f((5.0, 5.0)), atol = 5.0)

# Test that input dimension is properly checked for ND RMTC surrogates
@test_throws ArgumentError my_rmtcnd(Float64[])
@test_throws ArgumentError my_rmtcnd(2.0)
@test_throws ArgumentError my_rmtcnd((2.0, 3.0, 4.0))

# Per-dimension num_elements
my_rmtcnd2 = RMTCSurrogate(x, y, lb, ub, num_elements = (4, 6), energy_weight = 1.0e-6)
@test isapprox(my_rmtcnd2((3.0, 4.0)), f((3.0, 4.0)), atol = 5.0)

# The CubicHermiteSpline-collapse call overload and the analytic
# tensor-product basis-row used to build the fit must agree exactly, both in
# 1D and in the general ND case.
row = Surrogates._rmtc_row(my_rmtc1d.grids, my_rmtc1d.derivs, (3.0,))
@test isapprox(row' * vec(my_rmtc1d.coeff), my_rmtc1d(3.0), atol = 1.0e-8)

row_nd = Surrogates._rmtc_row(my_rmtcnd.grids, my_rmtcnd.derivs, (3.0, 4.0))
@test isapprox(row_nd' * vec(my_rmtcnd.coeff), my_rmtcnd((3.0, 4.0)), atol = 1.0e-8)
