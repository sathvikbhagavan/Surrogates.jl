using SafeTestsets

@safetestset "SurrogatesFlux" begin
    using Surrogates
    using Flux
    using SurrogatesFlux
    using LinearAlgebra
    using Zygote

    @testset "1D" begin
        a = 0.0
        b = 10.0
        obj_1D = x -> 2 * x + 3
        x = reduce(hcat, sample(10, 0.0, 10.0, SobolSample())[:, :])
        y = obj_1D.(x)
        my_model = Chain(Dense(1, 1))
        my_neural_kwargs = NeuralSurrogate(x, y, a, b, model = my_model)
        my_neural = NeuralSurrogate(x, y, a, b)
        update!(my_neural, reduce(hcat, [[8.5]]), reduce(hcat, [[20.0]]))
        update!(my_neural, reduce(hcat, [3.2, 3.5]), reduce(hcat, [7.4, 8.0]))
        val = my_neural(5.0)
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        x = reduce(hcat, collect.(sample(5, lb, ub, SobolSample())))
        obj_ND_neural(x) = x[1] * x[2]
        y = reduce(hcat, obj_ND_neural.(eachcol(x)))
        my_model = Chain(Dense(2, 1))
        my_loss(x, y) = Flux.mse(my_model(x), y)
        my_opt = Descent(0.01)
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = my_loss,
            opt = my_opt, n_epochs = 1)
        my_neural_kwargs = NeuralSurrogate(x, y, lb, ub, model = my_model)
        my_neural((3.5, 1.49))
        my_neural([3.4, 1.4])
        add_point!(my_neural, (3.5, 1.4), 4.9)
        add_point!(my_neural, [(3.5, 1.4), (1.5, 1.4), (1.3, 1.2)], [1.3, 1.4, 1.5])
    end

    # Multi-output #98
    f = x -> [x^2, x]
    lb = 1.0
    ub = 10.0
    x = sample(5, lb, ub, SobolSample())
    push!(x, 2.0)
    y = f.(x)
    my_model = Chain(Dense(1, 2))
    my_loss(x, y) = Flux.mse(my_model(x), y)
    surrogate = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = my_loss,
        opt = my_opt, n_echos = 1)
    surr_kwargs = NeuralSurrogate(x, y, lb, ub)

    f = x -> [x[1], x[2]^2]
    lb = [1.0, 2.0]
    ub = [10.0, 8.5]
    x = sample(20, lb, ub, SobolSample())
    push!(x, (1.0, 2.0))
    y = f.(x)
    my_model = Chain(Dense(2, 2))
    my_loss(x, y) = Flux.mse(my_model(x), y)
    surrogate = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = my_loss,
        opt = my_opt, n_echos = 1)
    surrogate_kwargs = NeuralSurrogate(x, y, lb, ub)
    surrogate((1.0, 2.0))
    x_new = (2.0, 2.0)
    y_new = f(x_new)
    add_point!(surrogate, x_new, y_new)

    #Optimization 
    lb = [1.0, 1.0]
    ub = [6.0, 6.0]
    x = sample(5, lb, ub, SobolSample())
    objective_function_ND = z -> 3 * norm(z) + 1
    y = objective_function_ND.(x)
    model = Chain(Dense(2, 1), first)
    loss(x, y) = Flux.mse(model(x), y)
    opt = Descent(0.01)
    n_echos = 1
    my_neural_ND_neural = NeuralSurrogate(x, y, lb, ub)
    surrogate_optimize(objective_function_ND, SRBF(), lb, ub, my_neural_ND_neural,
        SobolSample(), maxiters = 15)

    # AD Compatibility
    lb = 0.0
    ub = 3.0
    n = 10
    x = sample(n, lb, ub, SobolSample())
    f = x -> x^2
    y = f.(x)
    #NN
    @testset "NN" begin
        my_model = Chain(Dense(1, 1), first)
        my_loss(x, y) = Flux.mse(my_model(x), y)
        my_opt = Descent(0.01)
        n_echos = 1
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = my_loss,
            opt = my_opt, n_echos = 1)
        g = x -> my_neural'(x)
        g(3.4)
    end

    lb = [0.0, 0.0]
    ub = [10.0, 10.0]
    n = 5
    x = sample(n, lb, ub, SobolSample())
    f = x -> x[1] * x[2]
    y = f.(x)

    #NN
    @testset "NN ND" begin
        my_model = Chain(Dense(2, 1), first)
        my_loss(x, y) = Flux.mse(my_model(x), y)
        my_opt = Descent(0.01)
        n_echos = 1
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = my_loss,
            opt = my_opt, n_echos = 1)
        g = x -> Zygote.gradient(my_neural, x)
        g((2.0, 5.0))
    end

    # ###### ND -> ND ######

    lb = [0.0, 0.0]
    ub = [10.0, 2.0]
    n = 5
    x = sample(n, lb, ub, SobolSample())
    f = x -> [x[1]^2, x[2]]
    y = f.(x)

    #NN
    @testset "NN ND -> ND" begin
        my_model = Chain(Dense(2, 2))
        my_loss(x, y) = Flux.mse(my_model(x), y)
        my_opt = Descent(0.01)
        n_echos = 1
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = my_loss,
            opt = my_opt, n_echos = 1)
        Zygote.gradient(x -> sum(my_neural(x)), (2.0, 5.0))

        my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
        Zygote.gradient(x -> sum(my_rad(x)), (2.0, 5.0))

        my_p = 1.4
        my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
        my_inverse((2.0, 5.0))
        Zygote.gradient(x -> sum(my_inverse(x)), (2.0, 5.0))

        my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        Zygote.gradient(x -> sum(my_second(x)), (2.0, 5.0))
    end
end
