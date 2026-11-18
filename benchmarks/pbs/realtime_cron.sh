#!/bin/bash
# One pass of the realtime launcher, for cron (see applications/hrrr_ptype_realtime_submit.py and docs/realtime.md).
#
# Cron starts with an almost empty environment, so everything is spelled out here: the PBS commands' location,
# the Casper Python environment, and the three settings a person decides (project code, AWS profile name, and
# whether catch-up runs). Settings live in $HOME/.ptype_realtime.env so they are not in the repository:
#     PTYPE_ACCOUNT=NAML0001
#     PTYPE_AWS_PROFILE=wwe-ptype-realtime
# The profile is only a NAME passed to the jobs; this script never reads a credential.
set -uo pipefail
export PATH=/opt/pbs/bin:/usr/local/bin:/usr/bin:/bin:$PATH
PTYPE_FAST=/glade/work/dgagne/ptype_fast
JOBS=${PTYPE_JOBS_DIR:-/glade/derecho/scratch/$USER/ptype_fast/realtime/jobs}
[ -f "$HOME/.ptype_realtime.env" ] && . "$HOME/.ptype_realtime.env"
: "${PTYPE_ACCOUNT:?set PTYPE_ACCOUNT in ~/.ptype_realtime.env}"
: "${PTYPE_AWS_PROFILE:?set PTYPE_AWS_PROFILE in ~/.ptype_realtime.env}"
mkdir -p "$JOBS"
cd "$PTYPE_FAST/ptype-physical" || exit 1
# One line per pass with a timestamp, so a silent cron is distinguishable from a quiet one.
{
    echo "$(date -u +%FT%TZ) pass"
    "$PTYPE_FAST/venv_casper/bin/python" applications/hrrr_ptype_realtime_submit.py \
        --jobs-dir "$JOBS" --account "$PTYPE_ACCOUNT" --aws-profile "$PTYPE_AWS_PROFILE" ${PTYPE_SUBMIT_ARGS:-}
} >> "$JOBS/cron.log" 2>&1
