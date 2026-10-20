#!/usr/bin/env bash
#
# download_gfs_rolling_series.sh
#
# Downloads one GFS forecast cycle PER CALENDAR DAY over a date range.
#
# Each daily cycle is downloaded out to a full 12-day forecast horizon.
# This is intended for a rolling-replan experiment where the routing
# algorithm gets a fresh GFS forecast every replan interval (24h by default)
# and re-solves the remaining route using the newest forecast.
#
# For example, with a 14-day backtest:
#
#   20241101 00Z -> f000..f288
#   20241102 00Z -> f000..f288
#   20241103 00Z -> f000..f288
#   ...
#
# Although this means downloading substantially more data than a short
# 24-48h rolling horizon, it ensures that every daily replan has access
# to the full remaining forecast horizon.
#
# PERFORMANCE NOTES (this version):
#   1. All (date, forecast-hour) jobs across the ENTIRE date range are
#      flattened into a single work queue and run under one `xargs -P`.
#      The previous per-day loop meant that near the end of each day's
#      hour list, fewer than JOBS workers were active (long tail),
#      then the whole pipeline stalled at the day boundary while the
#      next day's idx fetches started cold. Flattening keeps the worker
#      pool saturated across the whole backtest window.
#   2. UGRD/VGRD @ 10 m are (almost always) adjacent records in the GFS
#      idx file. Previously each variable triggered its own curl range
#      request (2 requests/hour). Adjacent index entries are now merged
#      into a single byte range, halving the number of range requests
#      in the common case.
#
# Source: same NOAA public AWS archive as download_gfs_14day_wind.sh
# (noaa-gfs-bdp-pds), which retains GFS data from 2021-present -- so this
# works fine for a historical backtest window, not just "live" downloads.
#
# Usage:
#   ./download_gfs_rolling_series.sh START_DATE NUM_DAYS CYCLE [MAX_HOUR] [OUTBASE] [PARALLEL_JOBS]
#
# Example (14-day backtest window, 00Z cycle each day, 48h lead time kept
# per cycle):
#   ./download_gfs_rolling_series.sh 20241101 14 00 48 gfs_rolling 6
#
# Output: one directory per day, OUTBASE_YYYYMMDD_CCz/ , each containing
# gfs.tCCz.pgrb2.0p25.fFFF.uv10m.grib2 files for f000..fMAX_HOUR -- same
# per-file naming/format as download_gfs_14day_wind.sh (just a shorter
# forecast-hour range), so set_gfs_forecast_source! needs no changes to
# read these.
#
set -euo pipefail

START_DATE="${1:?Usage: $0 START_DATE(YYYYMMDD) NUM_DAYS CYCLE [MAX_HOUR] [OUTBASE] [PARALLEL_JOBS]}"
NUM_DAYS="${2:?Usage: $0 START_DATE(YYYYMMDD) NUM_DAYS CYCLE [MAX_HOUR] [OUTBASE] [PARALLEL_JOBS]}"
CYCLE="${3:?Usage: $0 START_DATE(YYYYMMDD) NUM_DAYS CYCLE [MAX_HOUR] [OUTBASE] [PARALLEL_JOBS]}"
MAX_HOUR="${4:-336}"
OUTBASE="${5:-gfs_rolling}"
JOBS="${6:-12}"

# Forecast-hour list, capped at MAX_HOUR: hourly 0-120 (or up to MAX_HOUR if
# smaller), then 3-hourly 123-336 if MAX_HOUR reaches that far. Mirrors
# download_gfs_14day_wind.sh's native-resolution schedule, just truncated.
build_hours() {
    local max_h="$1"
    local hrs=()
    local hourly_cap=$(( max_h < 120 ? max_h : 120 ))
    for h in $(seq 0 "$hourly_cap"); do hrs+=("$h"); done
    if [ "$max_h" -gt 120 ]; then
        for h in $(seq 123 3 "$max_h"); do hrs+=("$h"); done
    fi
    printf '%s\n' "${hrs[@]}"
}

# Convert a start date + day offset into YYYYMMDD, portably (GNU or BSD date).
offset_date() {
    local start="$1" offset="$2"
    date -u -d "${start} +${offset} days" +%Y%m%d 2>/dev/null || \
        date -u -j -f "%Y%m%d" -v+"${offset}"d "${start}" +%Y%m%d
}

fetch_hour() {
    local h="$1" cycle="$2" base_url="$3" outdir="$4"
    local FHR FILE URL IDX_URL OUT IDX ranges range

    FHR=$(printf "%03d" "$h")
    FILE="gfs.t${cycle}z.pgrb2.0p25.f${FHR}"
    URL="${base_url}/${FILE}"
    IDX_URL="${URL}.idx"
    OUT="${outdir}/${FILE}.uv10m.grib2"

    if [ -s "$OUT" ]; then
        echo "    [${outdir}][f${FHR}] already exists, skipping"
        return 0
    fi

    IDX=$(curl -sS -f --connect-timeout 15 --retry 5 --retry-delay 2 \
                --retry-connrefused "$IDX_URL") || {
        echo "    [${outdir}][f${FHR}] ERROR: could not fetch .idx (run may not exist at this hour)"
        return 1
    }

    # idx format: msgnum:byteoffset:d=YYYYMMDDHH:VAR:LEVEL:FCST:...
    # Merge consecutive matching index entries (UGRD/VGRD @ 10 m are almost
    # always adjacent records) into a single byte range, so we typically
    # issue ONE range request per forecast hour instead of two.
    ranges=$(awk -F: '
        { off[NR]=$2; var[NR]=$4; lvl[NR]=$5; n=NR }
        END {
            m = 0
            for (i = 1; i <= n; i++) {
                if ((var[i] == "UGRD" || var[i] == "VGRD") && lvl[i] == "10 m above ground") {
                    matches[++m] = i
                }
            }
            i = 1
            while (i <= m) {
                start = matches[i]
                j = i
                while (j < m && matches[j+1] == matches[j] + 1) j++
                end = matches[j]
                if (end < n) print off[start]"-"(off[end+1]-1)
                else         print off[start]"-"
                i = j + 1
            }
        }' <<< "$IDX")

    if [ -z "$ranges" ]; then
        echo "    [${outdir}][f${FHR}] ERROR: UGRD/VGRD @ 10 m not found in idx"
        return 1
    fi

    : > "${OUT}.tmp"
    while IFS= read -r range; do
        if ! curl -sS -f --connect-timeout 15 --retry 5 --retry-delay 2 \
                  --retry-connrefused --range "$range" "$URL" >> "${OUT}.tmp"; then
            echo "    [${outdir}][f${FHR}] ERROR: range fetch failed (${range})"
            rm -f "${OUT}.tmp"
            return 1
        fi
    done <<< "$ranges"

    mv "${OUT}.tmp" "$OUT"
    local size
    size=$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")
    echo "    [${outdir}][f${FHR}] OK (${size} bytes)"
}
export -f fetch_hour

mapfile -t HOURS < <(build_hours "$MAX_HOUR")

echo "=============================================="
echo "GFS rolling-series wind download"
echo "=============================================="
echo "Start date:      ${START_DATE}"
echo "Num days:        ${NUM_DAYS}"
echo "Cycle:           ${CYCLE}Z (every day)"
echo "Max hour:        f${MAX_HOUR} per cycle"
echo "Hours per cycle: ${#HOURS[@]}  (f000..f${MAX_HOUR})"
echo "Output base:     ${OUTBASE}_YYYYMMDD_${CYCLE}z/"
echo "Parallel:        ${JOBS} workers (flattened across ALL days)"
echo "=============================================="
echo ""

# Build the full (date, hour) job list across every day up front, then
# create each day's output directory once, and run a single flattened
# xargs pool over everything. This keeps all $JOBS workers busy for the
# entire run instead of draining to zero at each day boundary.
JOBLIST=$(mktemp)
trap 'rm -f "$JOBLIST"' EXIT

for ((d = 0; d < NUM_DAYS; d++)); do
    THIS_DATE=$(offset_date "$START_DATE" "$d")
    OUTDIR="${OUTBASE}_${THIS_DATE}_${CYCLE}z"
    BASE_URL="https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.${THIS_DATE}/${CYCLE}/atmos"
    mkdir -p "$OUTDIR"
    for h in "${HOURS[@]}"; do
        printf '%s\t%s\t%s\n' "$h" "$BASE_URL" "$OUTDIR" >> "$JOBLIST"
    done
done

TOTAL_JOBS=$(wc -l < "$JOBLIST")
echo "Total forecast-hour downloads queued: ${TOTAL_JOBS}"
echo ""

xargs -P "$JOBS" -a "$JOBLIST" -L 1 bash -c 'fetch_hour "$1" "$0" "$2" "$3"' "$CYCLE"

echo ""
echo "=============================================="
echo "Done. ${NUM_DAYS} daily forecast cycle(s) written under ${OUTBASE}_*_${CYCLE}z/"
echo "Pass these directories (in date order) to Julia's rolling_gfs_dirs() helper"
echo "in rolling_replan.jl to build the gfs_dirs/gfs_init_dates arrays automatically."
echo "=============================================="