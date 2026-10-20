# using Dates, CSV, DataFrames, Random

# # ============================================================
# # SETTINGS
# # ============================================================

# # Number of forecast days to download per window-start date, matching
# # NDAYS in run_top5_windows.jl -- these two numbers must stay in sync.
# const NUM_DAYS = 20

# const CYCLE = "00"

# # Full forecast horizon kept per daily cycle (hours). 336 = 14 days --
# # matches MAX_HOUR in run_top5_windows.jl.
# const MAX_HOUR = 336

# const BASH_SCRIPT = "./download_gfs_rolling_series.sh"

# # One output prefix PER DATE, so each window gets its own directory tree
# # (matches OUTBASE_PREFIX in run_top5_windows.jl).
# const OUTBASE_PREFIX = "gfs_top5"

# const PARALLEL_JOBS = 12

# # Random date selection
# const N_WINDOWS     = 15
# const MIN_YEAR_DATE = Date(2022, 1, 1)
# const WINDOW_LEN    = 18                         # length of scored windows (from the CSV name)
# const MIN_GAP_DAYS  = max(WINDOW_LEN, NUM_DAYS)  # no overlap in windows OR downloaded days
# const RNG_SEED      = 42                         # set to nothing for a different draw each run

# # On Windows, plain `bash` on PATH resolves to C:\Windows\System32\bash.exe,
# # which is the WSL launcher stub -- not a real shell. If WSL has no distro
# # installed/initialized, that stub fails immediately with an execvpe error.
# # Git for Windows ships an actual bash build that works standalone, so on
# # Windows we resolve its bash.exe explicitly instead of trusting PATH.
# # Override with the BASH_EXE env var if Git Bash lives somewhere else.
# function resolve_bash()
#     if !Sys.iswindows()
#         return "bash"
#     end
#     if haskey(ENV, "BASH_EXE") && isfile(ENV["BASH_EXE"])
#         return ENV["BASH_EXE"]
#     end
#     candidates = [
#         "C:\\Program Files\\Git\\bin\\bash.exe",
#         "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
#         joinpath(get(ENV, "LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
#     ]
#     for c in candidates
#         isfile(c) && return c
#     end
#     error("Could not find Git Bash's bash.exe. Set the BASH_EXE environment " *
#           "variable to its full path, e.g. \"C:\\\\Program Files\\\\Git\\\\bin\\\\bash.exe\".")
# end

# const BASH_EXE = resolve_bash()

# # ============================================================
# # CHECKS
# # ============================================================

# if !isfile(BASH_SCRIPT)
#     error("Could not find Bash script: $BASH_SCRIPT")
# end

# chmod(BASH_SCRIPT, 0o755)   # no-op/harmless on Windows

# # ============================================================
# # RANDOM NON-OVERLAPPING DATES
# # ============================================================

# window_scores = CSV.read("climatology_window_scores_18d.csv", DataFrame)

# # Restrict to windows starting 2021-01-01 or later
# window_scores = window_scores[window_scores.window_start .>= MIN_YEAR_DATE, :]

# # Shuffle the candidate start dates, then greedily keep each one that is at
# # least `min_gap_days` away from everything already picked.
# function pick_random_windows(window_scores::DataFrame, k::Int;
#                              min_gap_days::Int = 20, seed = nothing)
#     rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
#     candidates = shuffle(rng, collect(window_scores.window_start))
#     picked = Date[]
#     for d in candidates
#         if all(abs(Dates.value(d - p)) >= min_gap_days for p in picked)
#             push!(picked, d)
#             length(picked) >= k && break
#         end
#     end
#     length(picked) < k && @warn "Only found $(length(picked)) non-overlapping windows (wanted $k)"
#     return sort(picked)
# end

# top_dates = pick_random_windows(window_scores, N_WINDOWS;
#                                 min_gap_days = MIN_GAP_DAYS, seed = RNG_SEED)

# println("$(length(top_dates)) random non-overlapping window-start dates (2021+) to download:")
# for d in top_dates
#     println("  ", d)
# end

# # ============================================================
# # DOWNLOAD EACH DATE
# # ============================================================

# for d in top_dates
#     date_string = Dates.format(d, "yyyymmdd")
#     outbase = "$(OUTBASE_PREFIX)_$(date_string)"

#     println()
#     println("============================================================")
#     println(" GFS ROLLING FORECAST DOWNLOAD — window start $d")
#     println("============================================================")
#     println("Start date:       $d")
#     println("Number of days:   $NUM_DAYS")
#     println("Cycle:            $(CYCLE)Z")
#     println("Forecast horizon: f000..f$(MAX_HOUR) per day")
#     println("Parallel jobs:    $PARALLEL_JOBS (flattened across ALL days)")
#     println("Output:           $outbase")
#     println("Using bash:       $BASH_EXE")
#     println("============================================================")
#     println()

#     try
#         run(`$BASH_EXE $BASH_SCRIPT $date_string $NUM_DAYS $CYCLE $MAX_HOUR $outbase $PARALLEL_JOBS`)
#         println()
#         println("✓ Completed rolling download for $d: $NUM_DAYS day(s) from $date_string $(CYCLE)Z")
#     catch err
#         println()
#         println("✗ FAILED rolling download for $d")
#         println("  Error: $err")
#         rethrow(err)
#     end
# end

# println()
# println("============================================================")
# println(" ALL $(length(top_dates)) WINDOWS DOWNLOADED")
# println("============================================================")

# using Dates, CSV, DataFrames, Random

# # ---- Same settings as your download script -------------------------------
# const N_WINDOWS     = 15
# const MIN_YEAR_DATE = Date(2022, 1, 1)
# const WINDOW_LEN    = 18
# const NUM_DAYS      = 20
# const MIN_GAP_DAYS  = max(WINDOW_LEN, NUM_DAYS)   # = 20
# const RNG_SEED      = 42

# # ---- Re-run the exact picker with the same seed --------------------------
# window_scores = CSV.read("climatology_window_scores_18d.csv", DataFrame)
# window_scores = window_scores[window_scores.window_start .>= MIN_YEAR_DATE, :]

# function pick_random_windows(window_scores::DataFrame, k::Int;
#                              min_gap_days::Int = 20, seed = nothing)
#     rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
#     candidates = shuffle(rng, collect(window_scores.window_start))
#     picked = Date[]
#     for d in candidates
#         if all(abs(Dates.value(d - p)) >= min_gap_days for p in picked)
#             push!(picked, d)
#             length(picked) >= k && break
#         end
#     end
#     return sort(picked)
# end

# random_dates = pick_random_windows(window_scores, N_WINDOWS;
#                                    min_gap_days = MIN_GAP_DAYS, seed = RNG_SEED)

# random_strings = Set(Dates.format.(random_dates, "yyyymmdd"))

# println("Windows from RNG_SEED = $RNG_SEED ($(length(random_strings))):")
# for d in random_dates; println("  ", d); end

# # ---- What's actually on disk --------------------------------------------
# on_disk = sort(collect(keys(result.counts)))   # result from check_gfs_downloads

# from_random = String[]
# from_manual = String[]

# for ws in on_disk
#     if ws in random_strings
#         push!(from_random, ws)
#     else
#         push!(from_manual, ws)
#     end
# end

# println("\n" * "="^70)
# println("FROM RANDOM SEED ($(length(from_random))):")
# for ws in from_random; println("  ", ws); end

# println("\nFROM MANUAL / OTHER DOWNLOADS ($(length(from_manual))):")
# for ws in from_manual; println("  ", ws); end

# # ---- Sanity check --------------------------------------------------------
# println("\n" * "="^70)
# println("RANDOM DATES NOT ON DISK ($(length(setdiff(random_strings, Set(on_disk))))):")
# for ws in sort(collect(setdiff(random_strings, Set(on_disk))))
#     println("  ", ws, "   <- you still need to download this one")
# end


# cleanup_runs.jl -- standalone (no includes). Run from your project folder:
#     julia cleanup_runs.jl
#
# DRY_RUN = true  -> only LISTS what would be deleted (and how much space it frees).
# DRY_RUN = false -> actually deletes. Check the dry-run list first; deletion is permanent.
#
# DELETES:
#   1. run outputs: hindsight/committed/rolling .jld2, blend caches, legs_*.csv,
#      run_status.csv, failed_windows.csv, selected_windows.csv
#   2. every gfs_top5_* forecast folder whose date is NOT in KEEP_TAGS
# KEEPS:
#   - the gfs_top5_* forecast folders for the dates in KEEP_TAGS
#   - everything else: conv_*, routes_*, ablation_*, convergence_*, ver_* outputs,
#     the old Nov-2024 files, gfs_rolling_*, ClimatologyOceanData, OceanData, ...
#
# To delete the kept forecasts as well, set KEEP_TAGS = String[].
