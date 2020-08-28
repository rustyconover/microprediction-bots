using Microprediction
using Distributions

# Slaggable Dog
#
# Attempt to fit the z1 streams with a Cauchy distribution
#
# Failed.  -9509.
write_config = Microprediction.Config("8a9067bcdd6fae5eb07a09245a93ce3e")

function make_sample(stream_name)
    values = Microprediction.get_lagged_values(write_config, stream_name)

    # Since emoji's are counts and there are no partial usages fit a poission distribution.
    distribution = fit(Cauchy, convert(Array{Float64}, values))

    # Grab random samples
    samples = rand(distribution, write_config.numPredictions)
    for delay in write_config.delays
        Microprediction.submit(write_config, stream_name, convert(Array{Float64}, samples), delay)
    end
    println("Sending samples")
    println(samples)
end

z1_streams = filter((v) -> startswith(v, "z1~"), keys(Microprediction.get_sponsors(write_config)))

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