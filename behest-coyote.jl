using Microprediction
using Distributions
using GaussianMixtures

# behest-coyote.jl
# 
# Try to predict the z1 streams using a Gaussian Mixture Model with 2 Gaussians.
#
# Result: Didn't go bankrupt but didn't do great either.
#
# Made 275.49.
#
write_config = Microprediction.Config("92f5994ec4266a96b7d6d9378aeafc59")

function make_sample(stream_name)
    values = Microprediction.get_lagged_values(write_config, stream_name)

    distribution = GMM(2, convert(Vector{Float64}, values))

    # Grab random samples
    samples = rand(distribution, write_config.numPredictions)
    for delay in write_config.delays
        Microprediction.submit(write_config, stream_name, convert(Array{Float64}, samples), delay)
    end
    println("Sending samples")
    println(samples)
end

z1_streams = filter((v) -> startswith(v, "z1~"), keys(Microprediction.get_sponsors(write_config)))

z1_streams = filter((v) -> !startswith(v, "z1~pandemic"), z1_streams)

while true
    for stream_name in z1_streams
        println("Doing stream $(stream_name)")
        make_sample(stream_name)
    end
    sleep(60)
end
