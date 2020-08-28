using Microprediction
using Distributions

# Agalaxy Moose
#
# Predict the number of tweets by fitting a Poission distribution to the
# latest 30 datapoints of every emoji tracker stream.
#
write_config = Microprediction.Config("68a789b6d49cc76854f33482eb32649a")

function make_sample(stream_name)
    values = Microprediction.get_lagged_values(write_config, stream_name)
    # Grab the latest 50 values, which is kind of strange they are listed first.
    good_values = values[1:30]
    println(good_values)

    # Since emoji's are counts and there are no partial usages fit a poission distribution.
    distribution = fit(Poisson, convert(Array{Int64}, good_values))

    println(distribution)
    # Grab random samples
    for delay in write_config.delays
        samples = convert(Array{Float64}, rand(distribution, write_config.numPredictions))
        Microprediction.submit(write_config, stream_name, samples, delay)
        println(samples)
    end
end

emoji_streams = filter((v) -> startswith(v, "emojitracker-"), keys(Microprediction.get_sponsors(write_config)))

while true
    for stream_name in emoji_streams
        println("Doing stream $(stream_name)")
        make_sample(stream_name)
    end
    sleep(60)
end