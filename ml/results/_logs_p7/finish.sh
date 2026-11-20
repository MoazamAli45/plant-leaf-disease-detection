#!/usr/bin/env bash
# Runs after the Phase 7 sweep and the CD&S download: CD&S scores, overfitting checks, report.
set -uo pipefail
PY=.venv/bin/python
while pgrep -f "_logs_p7/sweep.sh" >/dev/null || pgrep -f "extra_field cds" >/dev/null; do sleep 60; done
[ -f manifests/cds/all.csv ] && $PY -m leafdx.field.phase7 add-cds
$PY -m leafdx.field.phase7 diagnose
$PY -m leafdx.field.phase7 report
echo FINISHED
