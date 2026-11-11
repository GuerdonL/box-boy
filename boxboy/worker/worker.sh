#!/usr/bin/env bash
# Runs ONE job on a rented GPU instance. Started by the instance's onstart command, which `bb launch`
# writes; not meant to be run by hand (set BB_ROOT and on_done=keep to try it on a machine you own).
#
# Reads: BB_JOB_B64 (base64 JSON, see cli.py:make_job), HF_TOKEN, and Vast's CONTAINER_ID / CONTAINER_API_KEY.
# Does:  clone the project at the launch commit, run its setup and check commands, restore any earlier
#        state of this run from the Hugging Face repo, run the job, upload <runs>/<name> while it runs and
#        when it ends, then destroy this instance (or stop it if the job failed, so the disk can be read).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${BB_ROOT:-/root}
LOG=$HERE/worker.log
exec > >(tee -a "$LOG") 2>&1
echo "[worker] start $(date -u +%FT%TZ)"

[ -f /etc/environment ] && { set -a; . /etc/environment 2>/dev/null; set +a; }   # some lines are not valid shell
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

job() { python3 -c "import base64,json,os,sys; print(json.loads(base64.b64decode(os.environ['BB_JOB_B64']))[sys.argv[1]])" "$1"; }
NAME=$(job name); CMD=$(job cmd); REF=$(job ref); REPO_URL=$(job repo_url); PROJECT=$(job project)
HF_REPO=$(job hf_repo); ON_DONE=$(job on_done); SYNC_S=$(job sync_s); RUNS=$(job runs_dir)
SETUP=$(job setup); CHECK=$(job check); MAX_HOURS=$(job max_hours)
eval "$(python3 -c "import base64,json,os,shlex; [print(f'export {k}={shlex.quote(str(v))}') for k,v in json.loads(base64.b64decode(os.environ['BB_JOB_B64']))['env'].items()]")"
export HF_TOKEN BB_HF_REPO=$HF_REPO
echo "[worker] job=$NAME project=$PROJECT ref=$REF hf=$HF_REPO on_done=$ON_DONE max_hours=$MAX_HOURS"
sync() { python "$HERE/sync.py" "$@" --runs "$RUNS"; }

finish() {   # $1 = exit code of the job
    sync push "$HF_REPO" "$NAME" --all --status "$1" --log "$LOG" || echo "[worker] final upload FAILED"
    if [ "$ON_DONE" = "keep" ]; then return; fi
    if [ "$1" = "0" ] && [ "$ON_DONE" = "destroy" ]; then
        echo "[worker] done, destroying instance $CONTAINER_ID"
        # newer CLIs ask for confirmation (and exit 0 when it is not given), older ones have no --yes
        vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --yes \
            || echo y | vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
        sleep 90   # a destroyed instance never gets here
        echo "[worker] still alive after destroy, stopping instead"
    else
        echo "[worker] stopping instance $CONTAINER_ID (exit code $1; disk kept for inspection)"
    fi
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
}

PIP="pip install -q"; command -v uv >/dev/null && PIP="uv pip install -q"
cd "$ROOT"
if [ ! -d "$PROJECT/.git" ]; then git clone -q "$REPO_URL" "$PROJECT" || { echo "[worker] clone failed"; finish 96; exit 96; }; fi
cd "$PROJECT" && git fetch -q origin && git checkout -q "$REF" || { echo "[worker] checkout $REF failed"; finish 96; exit 96; }
if [ -n "$SETUP" ]; then bash -c "$SETUP" || { echo "[worker] setup failed: $SETUP"; finish 97; exit 97; }; fi
python -c "import huggingface_hub" 2>/dev/null || $PIP huggingface_hub
command -v vastai >/dev/null || $PIP vastai
if [ -n "$CHECK" ]; then bash -c "$CHECK" || { echo "[worker] check failed: $CHECK"; finish 98; exit 98; }; fi

# `bb-pull <other run>` inside a job fetches that run's files into <runs>/<other run> (for evaluation jobs)
mkdir -p "$HERE/bin" && printf '#!/usr/bin/env bash\nexec python %q pull %q "$1" --any --runs %q\n' "$HERE/sync.py" "$HF_REPO" "$RUNS" > "$HERE/bin/bb-pull"
chmod +x "$HERE/bin/bb-pull"; export PATH="$HERE/bin:$PATH"

# an earlier attempt at this run (interrupted instance, or a rerun): restore it, or stop if it already finished
sync pull "$HF_REPO" "$NAME"; rc=$?
if [ $rc = 3 ]; then echo "[worker] run already finished"; finish 0; exit 0; fi

( while sleep "$SYNC_S"; do sync push "$HF_REPO" "$NAME" --log "$LOG"; done ) &
SYNC_PID=$!
echo "[worker] running: $CMD"
timeout --kill-after=60 "${MAX_HOURS}h" bash -c "$CMD"; rc=$?
kill $SYNC_PID 2>/dev/null
[ $rc = 124 ] && echo "[worker] job hit the ${MAX_HOURS} h limit (max_hours) and was stopped"
echo "[worker] command exited with $rc at $(date -u +%FT%TZ)"
finish $rc
