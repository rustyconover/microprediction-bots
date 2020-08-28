using Microprediction
using Distributions

# Sealable Goat
#
# Predict the z1 streams using a Normal distribution fit over the history of values.
#
# Worked pretty well +542
#
write_config = Microprediction.Config("69827d96043c0c2689792a0523140dd8")

function make_sample(stream_name)
    values = Microprediction.get_lagged_values(write_config, stream_name)

    # Since emoji's are counts and there are no partial usages fit a poission distribution.
    distribution = fit(Normal, convert(Array{Float64}, values))

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
        make_sample(stream_name)
    end
    sleep(60)
end