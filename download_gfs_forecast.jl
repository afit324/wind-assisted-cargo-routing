using Dates

# ============================================================
# SETTINGS
# ============================================================

# Number of forecast days to download per window-start date
const NUM_DAYS = 20

const CYCLE = "00"

# Full forecast horizon kept per daily cycle (hours). 336 = 14 days
const MAX_HOUR = 336

const BASH_SCRIPT = "./download_gfs_rolling_series.sh"

# Target directory for GFS downloads
const GFS_DIR = replace(joinpath("inputs", "gfs"), '\\' => '/')  # forward slashes for bash

const OUTBASE_PREFIX = "gfs_top5"

const PARALLEL_JOBS = 12

# ============================================================
# CHOOSE YOUR DATES HERE
# ============================================================
# Add or remove dates from this list. 
# The format must be Date(YYYY, MM, DD).

const top_dates = [
    Date(2022, 10, 03),
    Date(2022, 12, 15),
    Date(2023, 11, 06),
    Date(2024, 06, 16),
    Date(2024, 08, 22),
]

# ============================================================
# BASH RESOLUTION
# ============================================================

function resolve_bash()
    if !Sys.iswindows()
        return "bash"
    end
    if haskey(ENV, "BASH_EXE") && isfile(ENV["BASH_EXE"])
        return ENV["BASH_EXE"]
    end
    candidates = [
        "C:\\Program Files\\Git\\bin\\bash.exe",
        "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
        joinpath(get(ENV, "LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
    ]
    for c in candidates
        isfile(c) && return c
    end
    error("Could not find Git Bash's bash.exe. Set the BASH_EXE environment " *
          "variable to its full path, e.g. \"C:\\\\Program Files\\\\Git\\\\bin\\\\bash.exe\".")
end

const BASH_EXE = resolve_bash()

# ============================================================
# CHECKS
# ============================================================

if !isfile(BASH_SCRIPT)
    error("Could not find Bash script: $BASH_SCRIPT")
end

chmod(BASH_SCRIPT, 0o755)

# Ensure inputs/gfs exists
mkpath(GFS_DIR)

# ============================================================
# DOWNLOAD EACH DATE
# ============================================================

println("$(length(top_dates)) dates selected for download:")
for d in top_dates
    println("  ", d)
end
println()

for d in top_dates
    date_string = Dates.format(d, "yyyymmdd")
    
    # This builds: inputs/gfs/gfs_top5_20221003
    outbase = replace(joinpath(GFS_DIR, "$(OUTBASE_PREFIX)_$(date_string)"), '\\' => '/')

    println()
    println("============================================================")
    println(" GFS ROLLING FORECAST DOWNLOAD — window start $d")
    println("============================================================")
    println("Start date:       $d")
    println("Number of days:   $NUM_DAYS")
    println("Cycle:            $(CYCLE)Z")
    println("Forecast horizon: f000..f$(MAX_HOUR) per day")
    println("Parallel jobs:    $PARALLEL_JOBS (flattened across ALL days)")
    println("Output:           $outbase")
    println("Using bash:       $BASH_EXE")
    println("============================================================")
    println()

    try
        run(`$BASH_EXE $BASH_SCRIPT $date_string $NUM_DAYS $CYCLE $MAX_HOUR $outbase $PARALLEL_JOBS`)
        println()
        println("✓ Completed rolling download for $d: $NUM_DAYS day(s) from $date_string $(CYCLE)Z")
    catch err
        println()
        println("✗ FAILED rolling download for $d")
        println("  Error: $err")
        rethrow(err)
    end
end

println()
println("============================================================")
println(" ALL $(length(top_dates)) WINDOWS DOWNLOADED")
println("============================================================")
