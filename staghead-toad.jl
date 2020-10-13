# Simple Flux.jl models for Microprediction
#
# Allows a flexible set of periods and lags to build
# a neural network that outputs the parameters to a Normal
# distribution.
#
# Author: Rusty Conover <rusty@conover.me>
# 
using Flux
using Microprediction
using MicropredictionHistory
using TimeSeries
using DistributionsAD
using StatsBase
using Printf
using NamedTupleTools
using Hyperopt

MICROPREDICTION_HISTORY_BASE = "/Users/rusty/Development/pluto/data"
WRITE_KEY = "46ca0a3e0a2dc58c06643531abde9946"

"""
    createFluxModelNormal(stream_name, periods, lags)

Create a Flux model using the specified parameters and train it
for the specified epochs.

# Arguments
- `stream_name`: The name of the stream to load_live_data
- `periods`: add Fourier terms of the specified periods as features
- `lags`: add the specified lags as features
- `architecture`: the architecture of the neural network
- `learning_rate`: the learning rate of the neural network
- `batchsize`: the batch size of the neural network
- `epochs`: the number of epochs to train the neural network for

"""
function createFluxModelNormal(
    stream_name::AbstractString, 
    periods::AbstractArray{<:Integer}, 
    lags::AbstractArray{<:Integer};
    architecture::AbstractArray{<:Integer} = [128, 64, 32],
    learning_rate::AbstractFloat=0.001,
    batchsize::Integer=32,
    epochs::Integer=10,
    )
    
    l, cleaned_stats = stream_data_loader(stream_name, periods, lags; batchsize=batchsize)

    # Determine the number of inputs to the nn
    # by looking at the width of the first input
    input_count = length(l.data[1][1])

    paired_architecture_sizes = view.(Ref(architecture), 
        (:).(1:length(architecture)-1,2:length(architecture)))

    # Train two networks one that predicts the mean 
    # the other that predicts the standard deviation
    mu_model = Chain(
        Dense(input_count, architecture[1], leakyrelu),
        # Pair up the layers such that the input and sizes match
        map(pair -> Dense(pair[1], pair[2]), paired_architecture_sizes)...,
        Dense(architecture[end], 1)
    )

    stddev_model = Chain(
        Dense(input_count, architecture[1], leakyrelu),
        map(pair -> Dense(pair[1], pair[2]), paired_architecture_sizes)...,
        Dense(architecture[end], 1, softplus)
    )
        

    # The loss function of this network will be the 
    # sum of the negative log likelihood for Normal
    # distributions of the training data.
    function loss(ŷ, y)
        # Take the time to make the batch into a matrix from
        # such the computation time is reduced.
        inputs = reduce(hcat, ŷ)
        mu_out = mu_model(inputs)
        stddev_out = exp.(stddev_model(inputs))

        return -reduce((c, (index, y)) ->
           c + DistributionsAD.logpdf(
                DistributionsAD.Normal.(mu_out[index], stddev_out[index]), y),
            enumerate(y),
            init=0
        )
    end        


    optimiser = ADAM(learning_rate)
    p = Flux.params(mu_model, stddev_model)

    try 
        for i in 1:epochs
            Flux.train!(loss, p, l, optimiser)
            @printf "Epoch %d Loss: %.0f\n" i loss(l.data[1], l.data[2])
        end
    catch

        # If the fit failed return a large final loss
        # so that in a hyperparameter optimization this
        # model isn't chosen.
        return Dict(:mean => cleaned_stats[1],
                    :stream_name => stream_name,
                    :std => cleaned_stats[2],
                    :mu_model => mu_model,
                    :periods => periods,
                    :final_loss => 999999999,
                    :lags => lags,
                    :stddev_model => stddev_model)
    end

    # Make sure that the residuals have a mean zero near

    return Dict(:mean => cleaned_stats[1],
                :stream_name => stream_name,
                :std => cleaned_stats[2],
                :mu_model => mu_model,
                :periods => periods,
                :final_loss => loss(l.data[1], l.data[2]),
                :lags => lags,
                :stddev_model => stddev_model)
end

"""
    Replace missing values in the TimeSeries with randomly selected
    values from the time series.
"""
function randomly_select_missing_values(stream::MicropredictionHistory.LoadedStream)::AbstractArray{<:Number}
    # We should impute all of the missing values from random samples
    missing_times = findwhen(stream.data[:A] .=== missing)

    # Build the pool of samples that the missing values will be drawn
    # from.
    sampleable_values = filter(x -> x !== missing, values(stream.data))

    # Build a list of values which include the random samples if the value
    # is missing.
    cleaned_values = map(
        x -> x === missing ? sample(sampleable_values) : x,
        values(stream.data))

    return cleaned_values
end

"""
    stream_data_loader(stream_name, periods, lags; [batchsize])

Load a stream's history of values and prepare it such that it can 
be learned by a model by returning a Flux DataLoader.

# Arguments
- `stream_name`: The name of the stream to load_live_data
- `periods`: add Fourier terms of the specified periods as features
- `lags`: add the specified lags as features

"""
function stream_data_loader(
    stream_name::AbstractString, 
    periods::AbstractArray{<:Integer}, 
    lags::AbstractArray{<:Integer}; 
    batchsize::Number=32)
    stream = MicropredictionHistory.loadStream(MICROPREDICTION_HISTORY_BASE, stream_name; load_live_data=true)

    cleaned_values = convert(Array{Float32}, randomly_select_missing_values(stream))
    cleaned_stats = mean_and_std(cleaned_values)
    cleaned_z = zscore(cleaned_values)

    d = Dict(
        :value => cleaned_values,
        :datetime => timestamp(stream.data),
        :value_zscore => cleaned_z,
        )

    period_symbols = []
    # Now create the fourier terms for the specified periods.
    for period in periods 
        for name in ["sin", "cos"] 
            f = getfield(Main, Symbol(name))
            full_name = Symbol("$(name)_$(period)_values")
            d[full_name] = map(x -> f(2 * pi * x / period), 1:length(cleaned_values))
            push!(period_symbols, full_name)
        end
    end

    
    cleaned = TimeArray(namedtuple(d), timestamp=:datetime)

    lag_symbols = [];

    local lagged = cleaned

    # Build up all of the various lags.
    for l in lags
        symbs = map(x -> Symbol(x), ["value_lag_$(l)", "value_lag_zscore_$(l)"])
        
        push!(lag_symbols, symbs...)
        lagged = merge(lagged, 
            rename(
                lag(cleaned[:value, :value_zscore], l),
                symbs
            )
        )
    end

    lag_inputs = map(x -> Symbol("value_lag_zscore_$(x)"), lags)

    x_data = map(x -> [x...], 
    zip(
        map(x -> convert(Array{Float32}, values(lagged[x])), 
            [lag_inputs..., period_symbols...])...
    ))

    y_data = convert(Array{Float32}, values(lagged[:value]))

    return (Flux.Data.DataLoader((x_data, y_data), 
        batchsize=batchsize, 
        shuffle=true), cleaned_stats)
end

```
    project_stream_forward(model, forward_interval_count)
 
Project a models regressor values forward so that a forecast
can be returned.

```
function project_stream_forward(m::Dict{Symbol,Any}, forward_interval_count::Number) 
    stream = MicropredictionHistory.loadStream("/Users/rusty/Development/pluto/data", m[:stream_name]; load_live_data=true)
    cleaned_values = randomly_select_missing_values(stream)

    d::Dict{Symbol,Any} = Dict()

    period_symbols = []
    # Now create the fourier terms for the specified periods.
    period_offset = (length(cleaned_values) + forward_interval_count)

    for period in m[:periods] 
        for name in ["sin", "cos"] 
            f = getfield(Main, Symbol(name))
            full_name = Symbol("$(name)_$(period)_values")
            d[full_name] = f(2 * pi * period_offset / period)
            push!(period_symbols, full_name)
        end
    end

    # Now put in the lagged value.
    lag_inputs = []
    for l in m[:lags]
        v = (cleaned_values[end - l + forward_interval_count] - m[:mean]) / m[:std]
        name = Symbol("value_lag_zscore_$(l)")
        d[name] = v
        push!(lag_inputs, name)
    end

    # Now build the inputs to the model.
    model_input = map(x -> d[x], [lag_inputs..., period_symbols...])

    predicted_mu = m[:mu_model](model_input)[1]
    predicted_stddev = exp.(m[:stddev_model](model_input))[1]

    return DistributionsAD.Normal(predicted_mu, predicted_stddev)
end

```
    forecast_stream(model)

Send a forecast to Microprediction that is produced by a model.
```
function forecast_stream(model)
    write_config = Microprediction.Config(WRITE_KEY);

    distribution = project_stream_forward(model, 1)
    samples = collect(rand(distribution, write_config.numPredictions))
    samples = round.(samples)

    println("$(model[:stream_name]) $(distribution)")
    @async Microprediction.submit(write_config, model[:stream_name], convert(Array{Float64}, samples), 70);
end

```
    forecast_loop(models)

Loop over all models and send their forecasts then wait a minute
and do it all over again.
```
function forecast_loop(models::AbstractArray) 
    while true
        for m in models
            forecast_stream(m)
        end
        sleep(60)
    end
end

"""
    optimize(stream_name; search_count, epochs)

Perform a hyperparameter random search for the optimal model architecture

"""
function optimize(stream_name;
     search_count=3, epochs=3)
    ho = @phyperopt for i=search_count, 
        lags = [1:10, 1:30, 1:60, [1:15..., 60:-5:15...], [1:60..., 1440:-5:1440-60...]],
        architecture = [
            [128, 64, 32],
            [256, 64, 64, 32],
            [128, 64, 32, 16],
        ]
        m = createFluxModelNormal(stream_name,
            [1440, 1440*7, 720], 
            lags, 
            architecture = architecture,
            epochs=epochs)
        println("")
        println("architecture:", "\t", architecture, "\t", "lags:\t", lags, "\t", "final loss:", "\t", m[:final_loss])
        m[:final_loss]
    end
    return ho
end