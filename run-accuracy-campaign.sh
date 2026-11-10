#!/usr/bin/env bash
# Run the accuracy campaign from a plain terminal, with nothing else open.
# Close the editor, the browser and any assistant first: these columns are timing
# measurements and the numbers belong to whatever state the machine is in.
#
#   ./run-accuracy-campaign.sh [phases...]
#
# The Windows wrapper of the same name and this one call the same campaign; only
# the interpreter lookup differs.
set -u
cd "$(dirname "$0")"

# The modelling venv is a Windows layout here; on a POSIX host a sibling
# `.venv/bin/python` is the same environment, and a bare `python3` is the last
# resort. Whatever is chosen is printed, so a log never leaves it ambiguous.
# `PYTHON` wins over all three: a rented box or a WSL install usually keeps its
# venv somewhere else, and the bare `python3` that would be found first has no
# numpy -- every column would then refuse for the wrong reason, and the reason
# it gives (a missing artifact, an unchecked resident list) points the reader at
# the probes instead of at the interpreter.
PY="${PYTHON:-}"
if [ -n "$PY" ]; then
  # A name on PATH is as good as a path here (`PYTHON=python3.12`), but a value
  # that resolves to nothing is refused rather than quietly replaced: the whole
  # reason the variable exists is that the fallback is usually the wrong one.
  case "$PY" in
    */*) [ -x "$PY" ] || { echo "PYTHON=$PY is not an executable file" >&2; exit 2; } ;;
    *) PY="$(command -v "$PY" || true)"
       if [ -z "$PY" ]; then
         echo "PYTHON names no interpreter on PATH" >&2
         exit 2
       fi ;;
  esac
fi
if [ -z "$PY" ]; then
  for candidate in modeling/.venv/bin/python .venv/bin/python; do
    if [ -x "$candidate" ]; then
      PY="$candidate"
      break
    fi
  done
fi
if [ -z "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PY" ]; then
  echo "no python interpreter found: create modeling/.venv or install python3" >&2
  exit 2
fi
echo "interpreter: $PY"

# The campaign imports numpy before it can preflight a single column, so an
# interpreter without it is a wrong-interpreter answer rather than an empty
# machine: say which interpreter and how to name the right one.
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
  echo "this interpreter has no numpy: $PY" >&2
  echo "name the venv's interpreter instead, for example:" >&2
  echo "  PYTHON=/path/to/venv/bin/python $0 $*" >&2
  exit 3
fi

echo
echo "  accuracy campaign : model, dense, e2a"
echo "  output            : build/accuracy-YYYYMMDD/campaign.log"
echo
echo "  Anything still running will show up in the log's resident list."
echo "  Press Ctrl+C now if a program you meant to close is still open."
echo
sleep 10

"$PY" tools/run_accuracy_campaign.py "$@"
RC=$?

echo
if [ "$RC" -eq 0 ]; then
  echo "Campaign finished. Read build/accuracy-*/CAMPAIGN_SUMMARY.md"
else
  echo "Campaign stopped with code $RC. The last phase in build/accuracy-*/campaign.log failed."
fi
echo
exit "$RC"
