# CLAUDE.md — box-boy

Read the README first; it is the user-facing description and it is accurate. This file is what the README
leaves out: how the pieces fit, what has and has not been exercised, and the traps.

Owner: Guerdon (GuerdonL on GitHub). He prefers decisions made and flagged over questions asked. Commit messages
end with a Co-Authored-By line for Claude.

## What this is for

The control machine (a small droplet, no GPU) holds this repo and one or more project checkouts. The first
project is `tribe` (`github.com/GuerdonL/tribe`): its `CLAUDE.md` is the handoff for the research itself, its
`boxboy.toml` and `jobs/` are how it uses this harness. Keep project-specific things out of this repo.

## Pieces

```
boxboy/cli.py            the `bb` command: launch, status, logs, ssh, wait, collect, destroy, doctor
boxboy/worker/worker.sh  runs on the rented instance: clone, setup, check, restore, run, upload, destroy
boxboy/worker/sync.py    runs on the instance: push/pull one run folder to/from the Hugging Face repo
setup.sh                 control-machine install (virtualenv in .venv, symlinks in ~/.local/bin)
tests/test_cli.py        pure-function tests; nothing in them rents or uploads
```

How a launch travels:

1. `bb launch` reads `boxboy.toml` and the jobs file, checks HEAD is pushed, searches offers (cheapest first).
2. It uploads `worker.sh` and `sync.py` to `<hf repo>/_boxboy/<hash of both files>/` if that hash is not there yet.
3. Per job it creates an instance whose onstart command carries the job as base64 JSON (`BB_JOB_B64`), downloads
   the two files from the HF repo with the instance's `HF_TOKEN`, and starts `worker.sh` under `nohup`.
4. The worker's log is `/root/boxboy/worker.log` on the instance. It is not in `vastai logs` (that shows the
   container's own output). `bb logs` reads it over SSH, or falls back to the copy uploaded as `<name>/worker.log`.
5. `bb launch` attaches this machine's `~/.ssh/id_*.pub` to each instance so `bb logs` and `bb ssh` work.

The job JSON's fields are made in `cli.py:make_job` and read by name in `worker.sh` (`$(job field)`); a test checks
the worker asks only for fields that exist. Change both together. Any change to the worker files changes the
bundle hash, so running instances keep the version they started with.

## What has been exercised (2026-11-11)

- A real rented 4090 ran an earlier version of the worker end to end (20 training updates of tribe): clone,
  install, GPU check, training, uploads, and self-destroy with the instance's own key.
- The current worker ran end to end on a machine we already had (`BB_ROOT=<dir>`, `on_done=keep`), started
  from the exact onstart command: download from the HF repo, clone, setup, check, restore, a short job,
  periodic and final upload, `bb-pull`. `bb logs` (uploaded copy), `bb collect`, `bb status`, `bb doctor`
  and `bb launch --dry-run` were run against the live Vast and Hugging Face accounts with `vastai` 1.11.
- **Real launches, 2026-11-11** (`gpu_xl`, 20 updates, snapshot at 10):
  - `smoke_u20` on a host with driver 550 / CUDA 12.4: worker fetched from HF in a real onstart, SSH key attach and
    `bb logs` over SSH worked, the GPU `check` failed (driver older than the image's CUDA 13.2), log uploaded,
    instance stopped: the failure path works. `bb destroy` worked on it. This led to `min_host_cuda`.
  - `smoke_b` on a CUDA 13.2 host: setup, check, 82 s compile, about 4 s/update, periodic and final upload, self-destroy.
    `bb collect --ckpt` returned STATUS.json (exit 0), log.csv, params.pkl, ckpt.pkl, snap/u00010.pkl, worker.log.
- **Not yet exercised:** `bb wait`, restore after an instance restart, the `max_hours` timeout, the final-upload
  retry (simulated only), and the `flock` path under a real collision.
- **Known wart:** `bb status` prints `cur_state` (Vast's intended state) when `actual_status` is empty, so a loading
  or stopping instance can read "running". `bb logs` on a loading instance says "no uploaded log" rather than "loading".

## Traps

- **`vastai destroy` asks for confirmation** in current CLIs and exits 0 when it gets none. Older CLIs have no
  `--yes`. Both the worker and `bb destroy` try `--yes` first, then pipe `y`. The worker then waits 90 s and stops
  the instance if it is still alive; `bb destroy` re-lists and says so if anything is left.
- **An API key can list and search but not rent.** The error is `401 ... lacks the api.ask route`. `bb doctor`
  cannot detect this; only a launch does, and it stops at the first such error.
- **`/etc/environment` on the instances is not valid shell** (one line fails to source). The worker sources it with
  errors ignored. Do not `set -e` around that.
- **The server-side `cuda_max_good` offer filter drops valid offers**; the CUDA floor is applied client-side
  (`offer.min_cuda`, raised to the CUDA the image tag names: `min_host_cuda`).
- **A host driver older than the image's CUDA fails at `cuInit` with `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`** on consumer
  GPUs (no forward compatibility). Seen 2026-11-11: driver 550 (CUDA 12.4) with `cuda-13.2.1` image; jax fell back
  to CPU and the `check` command caught it.
- **Onstart commands have a size limit.** A test keeps the command under 4000 characters; the job's command,
  `setup`, `check` and `env` all ride in it.
- **`HF_TOKEN` reaches the instance through `--env`.** The onstart command falls back to grepping
  `/etc/environment` for it. If the worker download ever fails, the instance sits idle and billed with no log:
  `bb status` will show "no upload yet" long after launch. That is the signal; destroy it.
- **Renting is a purchase.** Show the dry run and get the owner's yes before any launch he has not already
  approved, and say what is still running at the end of a session.
- `snapshot_download` leaves a `.cache` folder inside the runs folder. It is harmless and git-ignored in tribe.

## Ideas not built

- Private project repos (pass a GitHub token to the instance).
- Several jobs in sequence on one instance, for many short jobs where the minute of setup dominates.
- A hard budget: destroy everything after N dollars or hours, enforced from the control machine by cron.
