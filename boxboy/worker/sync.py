"""Move one run's files between <runs>/<name>/ and a Hugging Face model repo (folder <name>/).
Runs on the rented instance; `bb launch` uploads this file next to worker.sh.

    python sync.py push <hf_repo> <name> [--all] [--status CODE] [--log FILE] [--runs DIR]
    python sync.py pull <hf_repo> <name> [--any] [--runs DIR]     # exit code 3 if the run already finished

Periodic pushes skip ckpt.pkl (large, only needed to resume) unless --all is given or it has not
been uploaded for an hour. --status writes STATUS.json, which marks the run finished (code 0) or failed.
pull --any is for a job that reads another run's files: it always downloads and never returns 3.
"""
import argparse
import json
import os
import sys
import time

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

CKPT_EVERY_S = 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["push", "pull"])
    ap.add_argument("repo")
    ap.add_argument("name")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--any", action="store_true")
    ap.add_argument("--status", type=int, default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    api = HfApi()
    local = os.path.join(a.runs, a.name)

    if a.op == "pull":
        if not a.any and os.path.exists(os.path.join(local, "ckpt.pkl")):
            print("[sync] local checkpoint present (instance was stopped and restarted); keeping it")
            return 0
        try:
            snapshot_download(a.repo, allow_patterns=[f"{a.name}/*"], local_dir=a.runs)
        except RepositoryNotFoundError:
            print(f"[sync] {a.repo} does not exist yet; nothing to restore")
            return 0
        st = os.path.join(local, "STATUS.json")
        if not a.any and os.path.exists(st) and json.load(open(st)).get("exit_code") == 0:
            return 3
        print(f"[sync] restored {sorted(os.listdir(local)) if os.path.isdir(local) else 'nothing'}")
        return 0

    os.makedirs(local, exist_ok=True)
    api.create_repo(a.repo, private=True, exist_ok=True)
    if a.log and os.path.exists(a.log):
        with open(a.log, "rb") as src, open(os.path.join(local, "worker.log"), "wb") as dst:
            dst.write(src.read())
    if a.status is not None:
        json.dump({"exit_code": a.status, "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                   "instance": os.environ.get("CONTAINER_ID")}, open(os.path.join(local, "STATUS.json"), "w"))
    stamp = os.path.join(local, ".ckpt_pushed")
    due = a.all or not os.path.exists(stamp) or time.time() - os.path.getmtime(stamp) > CKPT_EVERY_S
    ignore = [".ckpt_pushed", "*.tmp"] + ([] if due else ["ckpt.pkl"])
    api.upload_folder(repo_id=a.repo, folder_path=local, path_in_repo=a.name, ignore_patterns=ignore,
                      commit_message=f"{a.name}: {'final' if a.status is not None else 'sync'}")
    if due:
        open(stamp, "w").close()
    print(f"[sync] pushed {a.name} ({'with' if due else 'without'} ckpt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
