using Microprediction
using Distributions
using GaussianMixtures

# Calaboose Dog

# Attempt to predict the z1 streams using a Gaussian Mixture Model with
# three Gaussians.
#
# Didn't do so great -860.
#
write_config = Microprediction.Config("a3709a2591f66173f968d82fcc0bbc72")

function make_sample(stream_name)
    values = Microprediction.get_lagged_values(write_config, stream_name)

    distribution = GMM(3, convert(Vector{Float64}, values))

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
        try
            make_sample(stream_name)
        catch
        end

    end
    sleep(60)
end