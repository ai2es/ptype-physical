#!/bin/bash
# Keep at most MAX p-type jobs in the queue until every run in the list is in the index.
#
# Why this exists: on 2026-11-17 all 188 jobs of the 2025-26 list were submitted at once and about 96 nodes ran
# together. Downloads and uploads then failed across the board and 157 jobs hit their walltime (see
# benchmarks/results/derecho/generation.md). With 10 or fewer nodes at a time every job succeeded.
#
# Each pass submits only as many jobs as there are free slots; the launcher itself skips runs that are already
# indexed or in flight. Run it under nohup on a login node:
#   AWS_PROFILE=<profile> ACCOUNT=<project> nohup bash benchmarks/pbs/throttle_submit.sh <work dir> >> throttle.log 2>&1 &
# <work dir> holds runs.txt and gets a jobs/ directory.
set -u
P=${1:?usage: throttle_submit.sh <work dir with runs.txt>}
MAX=${MAX:-10}
ACCOUNT=${ACCOUNT:?set ACCOUNT to the PBS project code}
INDEX=${INDEX:-s3://wwe-ptype/ptype/inits.json}
: "${AWS_PROFILE:?set AWS_PROFILE to a profile that can write the bucket}"
export AWS_PROFILE
bucket_and_key=${INDEX#s3://}
while true; do
  # Test the bucket itself. `aws sts get-caller-identity` kept passing for 3.5 hours after the SSO login had
  # expired and index reads were failing, and 290 jobs were submitted that exited at their credential check.
  if ! aws s3api head-object --bucket "${bucket_and_key%%/*}" --key "${bucket_and_key#*/}" >/dev/null 2>&1; then
    echo "$(date -u +%FT%TZ) credentials rejected; stopping"
    exit 1
  fi
  active=$(qstat -u "$USER" 2>/dev/null | grep -c "ptype_")
  free=$((MAX - active))
  if [ "$free" -gt 0 ]; then
    out=$(python applications/hrrr_ptype_submit.py --runs-file "$P/runs.txt" --jobs-dir "$P/jobs" --index "$INDEX" \
          --node-type cpu --seconds-per-forecast-hour-cpu 4 --walltime 00:30:00 --max-jobs "$free" --account "$ACCOUNT" 2>&1)
    echo "$(date -u +%FT%TZ) active=$active $(echo "$out" | head -1) submitted=$(echo "$out" | grep -c desched1)"
    if echo "$out" | grep -q "could not read the index"; then
      echo "$(date -u +%FT%TZ) index unreadable; stopping"
      exit 1
    fi
    # A rejected qsub (for example an overspent project) shows up as a traceback with nothing submitted.
    if echo "$out" | grep -q "CalledProcessError"; then
      echo "$(date -u +%FT%TZ) qsub failed; stopping. Last line: $(echo "$out" | tail -1)"
      exit 1
    fi
    if echo "$out" | head -1 | grep -q " 0 to submit" && [ "$active" -eq 0 ]; then
      echo "$(date -u +%FT%TZ) all runs done or indexed; stopping"
      exit 0
    fi
  else
    echo "$(date -u +%FT%TZ) active=$active no free slot"
  fi
  sleep 180
done
