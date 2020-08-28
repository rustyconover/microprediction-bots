using Microprediction
using Distributions
using KernelDensityEstimate

# Azoxazole Fox
#
# Attempt to fit every stream but emjoi trackers with a Kernel Density Estimate.

write_config = Microprediction.Config("cbcdebcfee86ba6f7f1438a75968f5cf")

function make_sample(stream_name)
    values = Microprediction.get_lagged_values(write_config, stream_name)

    distribution = kde!(convert(Array{Float64}, values))

    # Grab random samples
    samples = rand(distribution, write_config.numPredictions)
    for delay in write_config.delays
        Microprediction.submit(write_config, stream_name, convert(Array{Float64}, samples), delay)
    end
    println("Sending samples")
    println(samples)
end

streams = keys(Microprediction.get_sponsors(write_config))

# This bot was losing too much money to emojitracker streams.

streams = filter((v) -> !startswith(v, "emojitracker"), streams)

while true
    for stream_name in streams
        println("Doing stream $(stream_name)")
        try
            make_sample(stream_name)
        catch
        end
    end
    sleep(60)
end
