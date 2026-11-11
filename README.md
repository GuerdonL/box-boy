# box-boy

Run a project's training jobs on rented [Vast.ai](https://vast.ai) GPU instances from a small control machine
with no GPU. One job per instance. Results are uploaded to a private Hugging Face repo while the job runs, and
each instance destroys itself when its job succeeds.

box-boy knows nothing about any particular project. A project opts in with a `boxboy.toml` at its root and
plain-text jobs files; `bb` is run from inside the project's checkout.

## Set up a control machine

On a fresh Ubuntu or Debian server (a small droplet; 2 GB of RAM is comfortable if you also run the project's CPU tests):

```
git clone git@github.com:GuerdonL/box-boy.git && cd box-boy && ./setup.sh
```

`setup.sh` installs box-boy into its own virtualenv and puts `bb`, `vastai` and `hf` on your PATH. Then log in
to the three services, once each:

```
vastai set api-key <key>     # console.vast.ai > Account > API Keys. The key needs permission to create instances.
hf auth login                # a Hugging Face token with write access
gh auth login                # GitHub, to clone and push the project
```

Clone the project and check everything:

```
git clone git@github.com:GuerdonL/tribe.git && cd tribe
bb doctor
```

## Use

```
bb launch jobs/seeds.txt --dry-run    # which instances, where, and the hourly cost. Rents nothing.
bb launch jobs/seeds.txt              # asks, then rents one instance per job
bb status                             # instances, cost, and how long since each last uploaded
bb logs m1_s3 -f                      # follow a job's log
bb wait && bb collect                 # block until the jobs end, then download results into runs/
bb destroy m1_s3                      # or --all
bb ssh m1_s3                          # a shell on the instance
```

A jobs file has one job per line, `name: command`:

```
m1_s3: python -m tribe.train --preset gpu_xl --seed 3 --out runs/m1_s3 --resume
m1_s4: python -m tribe.train --preset gpu_xl --seed 4 --out runs/m1_s4 --resume
```

The command runs from the project root on the instance. It must write to `runs/<name>/`, because that folder is
what gets uploaded and what is restored if the job is launched again. A job that needs another run's files (an
evaluation, say) calls `bb-pull <other name>` first.

What happens on each instance:

1. It clones the project at the commit you launched from. That commit must be pushed, and the project repo must be public.
2. It runs the project's `setup` command (install dependencies) and its `check` command (for example, that the GPU is visible).
3. It restores `runs/<name>/` from the Hugging Face repo if an earlier attempt left anything there.
4. It runs the job, uploading `runs/<name>/` every 10 minutes and once more at the end, with `worker.log` and a `STATUS.json`.
5. On success it destroys itself. On failure, or after `max_hours`, it stops instead: the GPU is no longer billed
   but the disk is, until you read the log and run `bb destroy <name>`.

## boxboy.toml

Every key is optional; these are the defaults, apart from `setup`, `check` and `env`, which are empty.

```toml
name = "tribe"                 # instance labels are <name>:<job>; default is the repo's name
image = "vastai/pytorch:cuda-13.2.1-auto"
setup = "uv pip install -q -r requirements-gpu.txt"
check = "python -c \"import jax; assert jax.devices()[0].platform == 'gpu'\""
runs_dir = "runs"
# hf_repo = "someone/some-repo"   # default: <your hf user>/<name>-runs (private, created on first use)

[env]                          # extra environment variables for the job
XLA_PYTHON_CLIENT_PREALLOCATE = "false"

[offer]
gpu = "RTX_4090"
num_gpus = 1
max_price = 0.60               # $/h per instance
max_total = 5.0                # $/h for one launch; refused above this without --over-budget
disk = 30                      # GB
reliability = 0.98
inet_down = 200                # Mbit/s
min_cuda = 12.0                # what the host driver must support
exclude_geo = "CN"             # Hugging Face is unreachable from some countries

[run]
sync_s = 600                   # seconds between uploads
on_done = "destroy"            # destroy | stop | keep, for a job that succeeds
max_hours = 24                 # a job running longer is stopped as failed
```

`bb launch` flags (`--gpu`, `--num-gpus`, `--max-price`, `--disk`, `--on-done`, `--sync-s`, `--max-hours`)
override the file for one launch.

## Money

- `bb launch` prints the hourly total and asks before renting. `--dry-run` never rents.
- A stopped instance still bills for its disk. `bb status` lists them; destroy them once you have read the log.
- `bb status` shows the time since each job last uploaded. A running instance that has uploaded nothing for much
  longer than `sync_s` is probably stuck: look with `bb logs <name>` or `bb ssh <name>`.
- Your Hugging Face token is passed to each instance as an environment variable, so it is visible in your Vast
  account's instance configuration. Use a token scoped to what the runs need.

## Limits

- The project repo must be public: instances clone it without a login. This repo can stay private, because the
  instance-side scripts are handed over through the Hugging Face repo.
- One job is one instance. Nothing here splits a run across machines.
- Checkpoints are expected at `runs/<name>/ckpt.pkl` (uploaded at most hourly, because they are large; everything
  else in the folder goes up on every sync).

Tests: `.venv/bin/python -m pytest tests -q`.
