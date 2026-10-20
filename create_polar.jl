include("wind.jl")
include("read_write_data.jl")
include("routing.jl")
include("discritise_domain.jl")
include("plots.jl")
include("polars.jl")
include("great_circle.jl")

using CSV, DataFrames

function create_polar(power)
    println("Creating polar for $(power) kW...")
    
    twas = collect(0.0:2.0:180.0)  # 0 to 180 degrees, step 2
    twss = collect(0.0:0.2:20.0)   # 0 to 20 m/s, step 0.2
    
    df = DataFrame()
    df[!,:"TWA/TWS"] = twas  # Match the expected column name!
    
    for tws in twss
        speeds = zeros(length(twas))
        for (j, twa) in enumerate(twas)
            ret, result = getPerformanceFromPower(twa, tws, 0.0, 0.0, power)
            if ret == 0
                # Convert knots to m/s
                speeds[j] = round(result.speed_over_ground_kn * 0.514444, digits=2)
            else
                speeds[j] = 0.0  # Invalid combination
            end
        end
        df[!,string(tws)] = speeds
    end
    
    filename = "Polars/RisePol$(power)kW.csv"
    CSV.write(filename, df)
    println("✅ Created $filename")
end

# Create all required power levels
# Note: 0 kW = sail only, other values = motor-sailing with that power
power_levels = [0.0, 400.0, 1200.0, 2000.0]

# Add 2600kW if you need it (your original hardcoded value)
if !(2600.0 in power_levels)
    push!(power_levels, 2600.0)
end

for P_kw in power_levels
    create_polar(P_kw)
end

println("All polars created successfully!")