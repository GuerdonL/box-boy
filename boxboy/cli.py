"""bb: run a project's jobs on rented Vast.ai GPU instances, one job per instance.

Run it from inside the project's git checkout (the project has a boxboy.toml at its root):

    bb doctor                         # check logins, config and that the commit is pushed
    bb launch jobs/seeds.txt --dry-run
    bb launch jobs/seeds.txt          # asks, then rents
    bb status                         # instances, cost, time since each last uploaded
    bb logs <name> [-f]               # the job's log (live over SSH, else the last uploaded copy)
    bb wait [name ...]                # block until the jobs end
    bb collect [name ...]             # download results into the project's runs folder
    bb destroy <name ...> | --all

A jobs file has one job per line, `name: command`. The command runs from the project root on the
instance and must write to <runs_dir>/<name>/ (that folder is what gets uploaded and restored).

Each instance clones the project at the commit you launch from (it must be pushed), runs the project's
setup command, runs the job, uploads its folder to a private Hugging Face repo every few minutes and at
the end, then destroys itself. A failed job stops its instance instead (disk kept, GPU billing off).
"""
import argparse
import base64
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib

WORKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker")
WORKER_FILES = ["worker.sh", "sync.py"]
CONFIG_NAME = "boxboy.toml"

DEFAULTS = {
    "name": None,                       # label prefix for instances; default: the repo's name
    "image": "vastai/pytorch:cuda-13.2.1-auto",
    "setup": "",                        # shell, run once on the instance from the project root (install deps)
    "check": "",                        # shell, must exit 0 before the job starts (e.g. the framework sees the GPU)
    "runs_dir": "runs",
    "hf_repo": None,                    # default: <your hf user>/<name>-runs
    "env": {},                          # extra environment variables for the job
    "offer": {"gpu": "RTX_4090", "num_gpus": 1, "max_price": 0.60, "max_total": 5.0, "disk": 30,
              "reliability": 0.98, "inet_down": 200, "min_cuda": 12.0, "exclude_geo": "CN"},
    "run": {"sync_s": 600, "on_done": "destroy", "max_hours": 24},
}


# ------------------------------------------------------------- helpers ---
def sh(*args, check=True, stdin=subprocess.DEVNULL):
    r = subprocess.run(args, capture_output=True, text=True, stdin=stdin)
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(args[:4])} ...\n{r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def vast(*args, check=True):
    out = sh("vastai", *args, "--raw", check=check)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def parse_jobs(text, runs_dir="runs"):
    """`name: command` per line; blank lines and # comments ignored."""
    jobs = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(.+)$", line)
        if not m:
            raise ValueError(f"line {n}: expected `name: command`, got {line!r}")
        jobs.append((m.group(1), m.group(2)))
    names = [j[0] for j in jobs]
    dup = {x for x in names if names.count(x) > 1}
    if dup:
        raise ValueError(f"duplicate job names: {sorted(dup)}")
    for name, cmd in jobs:
        if not re.search(rf"{re.escape(runs_dir)}/{re.escape(name)}(?![A-Za-z0-9_.-])", cmd):
            raise ValueError(f"job {name}: the command must write to {runs_dir}/{name} (that folder is what gets saved)")
    return jobs


def merge_config(user):
    """DEFAULTS overlaid with the project's boxboy.toml; unknown keys are an error (typos cost money)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    for k, v in user.items():
        if k not in cfg:
            raise ValueError(f"{CONFIG_NAME}: unknown key {k!r}")
        if k in ("offer", "run"):
            bad = set(v) - set(cfg[k])
            if bad:
                raise ValueError(f"{CONFIG_NAME}: unknown key(s) in [{k}]: {sorted(bad)}")
            cfg[k].update(v)
        else:
            cfg[k] = v
    if cfg["run"]["on_done"] not in ("destroy", "stop", "keep"):
        raise ValueError(f"{CONFIG_NAME}: run.on_done must be destroy, stop or keep")
    return cfg


def project():
    """(root, config) of the project the command is run from."""
    root = sh("git", "rev-parse", "--show-toplevel", check=False)
    if not root:
        sys.exit("run bb from inside the project's git checkout")
    path = os.path.join(root, CONFIG_NAME)
    if not os.path.exists(path):
        sys.exit(f"{root} has no {CONFIG_NAME}; see the box-boy README for a starting one")
    try:
        cfg = merge_config(tomllib.load(open(path, "rb")))
    except ValueError as e:
        sys.exit(str(e))
    slug = repo_slug(sh("git", "-C", root, "remote", "get-url", "origin"))
    cfg["name"] = cfg["name"] or slug.split("/")[-1]
    cfg["slug"] = slug
    return root, cfg


def repo_slug(url):
    return re.sub(r"^(git@github\.com:|https://github\.com/|ssh://git@github\.com/)", "", url).removesuffix(".git")


def pushed_ref(root):
    ref = sh("git", "-C", root, "rev-parse", "HEAD")
    if not sh("git", "-C", root, "branch", "-r", "--contains", ref):
        sys.exit(f"commit {ref[:8]} is not on the remote; push it first (instances clone from GitHub)")
    dirty = sh("git", "-C", root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("warning: uncommitted changes will NOT be on the instances:\n" + dirty)
    return ref


def hf_login(cfg):
    from huggingface_hub import HfApi, get_token
    tok = get_token()
    if not tok:
        sys.exit("no Hugging Face login found; run `hf auth login` first")
    return tok, cfg["hf_repo"] or f"{HfApi().whoami()['name']}/{cfg['name']}-runs"


def bundle_id():
    h = hashlib.sha256()
    for f in WORKER_FILES:
        h.update(open(os.path.join(WORKER_DIR, f), "rb").read())
    return h.hexdigest()[:12]


def upload_bundle(hf_repo):
    """Put worker.sh and sync.py where the instances can fetch them with the HF token (so this repo can stay private)."""
    from huggingface_hub import HfApi
    api, bid = HfApi(), bundle_id()
    api.create_repo(hf_repo, private=True, exist_ok=True)
    if not api.file_exists(hf_repo, f"_boxboy/{bid}/worker.sh"):
        api.upload_folder(repo_id=hf_repo, folder_path=WORKER_DIR, path_in_repo=f"_boxboy/{bid}",
                          allow_patterns=WORKER_FILES, commit_message=f"boxboy worker {bid}")
    return bid


def make_job(name, cmd, ref, cfg, hf_repo, a):
    return dict(name=name, cmd=cmd, ref=ref, project=cfg["name"], repo_url=f"https://github.com/{cfg['slug']}.git",
                hf_repo=hf_repo, runs_dir=cfg["runs_dir"], setup=cfg["setup"], check=cfg["check"], env=cfg["env"],
                on_done=a.on_done or cfg["run"]["on_done"], sync_s=a.sync_s or cfg["run"]["sync_s"],
                max_hours=a.max_hours or cfg["run"]["max_hours"])


def onstart_cmd(job, bid):
    """The instance's onstart command: fetch the worker from the HF repo and hand it the job."""
    b64 = base64.b64encode(json.dumps(job).encode()).decode()
    url = f"https://huggingface.co/{job['hf_repo']}/resolve/main/_boxboy/{bid}"
    tok = "${HF_TOKEN:-$(grep -m1 '^HF_TOKEN=' /etc/environment | cut -d= -f2- | tr -d '\"')}"
    return (f"export BB_JOB_B64={b64}; mkdir -p /root/boxboy && cd /root/boxboy && T={tok} && "
            + " && ".join(f'curl -fsSL -H "Authorization: Bearer $T" {url}/{f} -o {f}' for f in WORKER_FILES)
            + " && nohup bash worker.sh > /dev/null 2>&1 &")


def offer_query(o):
    return (f"gpu_name={o['gpu']} num_gpus={o['num_gpus']} rentable=true reliability>{o['reliability']} "
            f"inet_down>{o['inet_down']} disk_space>={o['disk']} dph_total<={o['max_price']} "
            f"geolocation notin [{o['exclude_geo']}]")


def label_of(cfg):
    return cfg["name"] + ":"


def instances(cfg):
    rows = vast("show", "instances")
    if not isinstance(rows, list):
        sys.exit(f"could not list instances: {rows}")
    return [i for i in rows if (i.get("label") or "").startswith(label_of(cfg))]


def job_name(i, cfg):
    return i["label"][len(label_of(cfg)):]


def by_name(cfg, names):
    rows = instances(cfg)
    if names:
        rows = [i for i in rows if job_name(i, cfg) in names]
        missing = set(names) - {job_name(i, cfg) for i in rows}
        if missing:
            sys.exit(f"no instance for: {sorted(missing)}")
    return rows


def public_key():
    for p in sorted(glob.glob(os.path.expanduser("~/.ssh/id_*.pub"))):
        return open(p).read().strip()
    return None


def ssh_args(i):
    url = sh("vastai", "ssh-url", str(i["id"]), check=False).splitlines()[-1:]   # older CLIs print a warning first
    m = re.match(r"ssh://(\S+?)@(\S+?):(\d+)", url[0]) if url else None
    if not m:
        return None
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-p", m.group(3), f"{m.group(1)}@{m.group(2)}"]


def last_uploads(hf_repo):
    """{run name: seconds since its folder last changed on the HF repo}; {} if the repo is not there yet."""
    from huggingface_hub import HfApi
    try:
        tree = HfApi().list_repo_tree(hf_repo, expand=True)
        return {t.path: time.time() - t.last_commit.date.timestamp() for t in tree if getattr(t, "last_commit", None)}
    except Exception:
        return {}


def ago(s):
    return "no upload yet" if s is None else f"uploaded {s / 60:.0f} min ago" if s < 5400 else f"uploaded {s / 3600:.1f} h ago"


# ------------------------------------------------------------ commands ---
def cmd_launch(a):
    root, cfg = project()
    o = dict(cfg["offer"])
    for k in ("gpu", "num_gpus", "max_price", "disk"):
        if getattr(a, k) is not None:
            o[k] = getattr(a, k)
    try:
        jobs = parse_jobs(open(a.jobs).read(), cfg["runs_dir"])
    except ValueError as e:
        sys.exit(f"{a.jobs}: {e}")
    ref = pushed_ref(root)
    token, hf_repo = hf_login(cfg)
    running = {job_name(i, cfg) for i in instances(cfg)}
    clash = [n for n, _ in jobs if n in running]
    if clash:
        sys.exit(f"already have instances for: {clash} (destroy them or rename the jobs)")
    # the host driver's CUDA is filtered here: the server-side cuda_max_good filter drops valid offers
    found = vast("search", "offers", offer_query(o), "-o", "dph_total+")
    if not isinstance(found, list):
        sys.exit(f"offer search failed: {found}")
    offers = sorted((x for x in found if (x.get("cuda_max_good") or 0) >= o["min_cuda"]), key=lambda x: x["dph_total"])
    if len(offers) < len(jobs):
        sys.exit(f"only {len(offers)} offers match for {len(jobs)} jobs; raise --max-price or relax [offer] in {CONFIG_NAME}")
    picks = offers[: len(jobs)]
    total = sum(x["dph_total"] for x in picks)
    print(f"{cfg['name']}: commit {ref[:8]} of {cfg['slug']}; results to hf:{hf_repo}; image {cfg['image']}")
    for (name, cmd), x in zip(jobs, picks):
        print(f"  {name:20s} {x['num_gpus']}x {x['gpu_name']} ${x['dph_total']:.3f}/h {x.get('geolocation', '?'):18s} | {cmd}")
    print(f"{len(jobs)} instances, ${total:.2f}/h in total while they run. On success each "
          f"{'destroys itself' if (a.on_done or cfg['run']['on_done']) == 'destroy' else 'is kept or stopped'}; "
          f"a failed job stops its instance (disk still billed until you destroy it).")
    if total > o["max_total"] and not a.over_budget:
        sys.exit(f"that is over offer.max_total = ${o['max_total']:.2f}/h; pass --over-budget or raise it in {CONFIG_NAME}")
    if a.dry_run:
        return
    if not a.yes and input("rent these? [y/N] ").strip().lower() != "y":
        sys.exit("nothing rented")
    bid = upload_bundle(hf_repo)
    key = public_key()
    spare = offers[len(jobs):]
    for (name, cmd), x in zip(jobs, picks):
        job = make_job(name, cmd, ref, cfg, hf_repo, a)
        while True:
            r = vast("create", "instance", str(x["id"]), "--image", cfg["image"], "--disk", str(o["disk"]),
                     "--label", label_of(cfg) + name, "--ssh", "--direct", "--cancel-unavail",
                     "--env", f"-e HF_TOKEN={token}", "--onstart-cmd", onstart_cmd(job, bid), check=False)
            if isinstance(r, dict) and r.get("success"):
                iid = r.get("new_contract")
                print(f"  {name}: instance {iid} on offer {x['id']}")
                if key:      # so `bb logs` and `bb ssh` work from this machine
                    sh("vastai", "attach", "ssh", str(iid), key, check=False)
                break
            if re.search(r"\b40[13]\b|Authorization|lacks", str(r)):
                sys.exit(f"  {name}: this Vast API key is not allowed to rent instances: {r}\n"
                         "  Use an account key with instance-create permission (`vastai set api-key <key>`).")
            if not spare:
                print(f"  {name}: FAILED to rent, no offers left. Last error: {r}")
                break
            print(f"  {name}: offer {x['id']} failed ({str(r)[:120]}), trying the next one")
            x = spare.pop(0)
    print("next: bb status | bb logs <name> | bb wait && bb collect")


def cmd_status(a):
    _, cfg = project()
    rows = instances(cfg)
    if not rows:
        print(f"no {cfg['name']} instances")
        return rows
    ups = last_uploads(hf_login(cfg)[1])
    for i in rows:
        up = (i.get("duration") or 0) / 3600
        n = job_name(i, cfg)
        print(f"{n:20s} id {i['id']}  {i.get('actual_status') or i.get('cur_state') or '?':9s} "
              f"{i.get('num_gpus')}x {i.get('gpu_name')}  ${i.get('dph_total', 0):.3f}/h  up {up:.1f} h  {ago(ups.get(n))}")
    live = [i for i in rows if i.get("actual_status") == "running"]
    print(f"total ${sum(i.get('dph_total', 0) for i in live):.2f}/h for {len(live)} running; "
          f"{len(rows) - len(live)} not running (a stopped instance means its job failed: bb logs <name>, then bb destroy <name>)")
    return rows


def cmd_logs(a):
    _, cfg = project()
    rows = [i for i in instances(cfg) if job_name(i, cfg) == a.name]
    if rows and rows[0].get("actual_status") == "running":
        ssh = ssh_args(rows[0])
        if ssh:
            tail = f"tail -n {a.tail} {'-f ' if a.follow else ''}/root/boxboy/worker.log"
            if subprocess.run(ssh + [tail]).returncode == 0:
                return
            print("(could not read the log over SSH; showing the last uploaded copy)")
    from huggingface_hub import hf_hub_download
    _, hf_repo = hf_login(cfg)
    try:
        path = hf_hub_download(hf_repo, f"{a.name}/worker.log", force_download=True)
    except Exception as e:
        sys.exit(f"no uploaded log for {a.name} yet ({type(e).__name__})")
    print("".join(open(path, errors="replace").readlines()[-a.tail:]), end="")


def cmd_ssh(a):
    _, cfg = project()
    ssh = ssh_args(by_name(cfg, [a.name])[0])
    if not ssh:
        sys.exit("no SSH address for that instance yet")
    ssh = [x for x in ssh if x != "BatchMode=yes"]
    if a.command:
        sys.exit(subprocess.run(ssh + [" ".join(shlex.quote(c) for c in a.command)]).returncode)
    os.execvp("ssh", ssh)


def cmd_wait(a):
    _, cfg = project()
    while True:
        rows = [i for i in instances(cfg) if not a.names or job_name(i, cfg) in a.names]
        live = [i for i in rows if i.get("actual_status") in ("running", "loading", "created", None)]
        print(time.strftime("%H:%M:%S"), ", ".join(f"{job_name(i, cfg)}={i.get('actual_status')}" for i in rows) or "none left", flush=True)
        if not live:
            break
        time.sleep(a.every)
    failed = [job_name(i, cfg) for i in rows]
    if failed:
        sys.exit(f"still present and not running (failed jobs keep their instance): {failed}")
    print("all jobs ended and their instances are gone; `bb collect` downloads the results")


def cmd_collect(a):
    from huggingface_hub import snapshot_download
    root, cfg = project()
    _, hf_repo = hf_login(cfg)
    dest = os.path.join(root, cfg["runs_dir"])
    pats = [f"{n}/*" for n in a.names] or None
    snapshot_download(hf_repo, allow_patterns=pats, ignore_patterns=["_boxboy/*"] + ([] if a.ckpt else ["*/ckpt.pkl"]),
                      local_dir=dest)
    for d in sorted(os.listdir(dest)):
        st = os.path.join(dest, d, "STATUS.json")
        if os.path.isdir(os.path.join(dest, d)) and not d.startswith((".", "_")) and (not a.names or d in a.names):
            state = f"exit code {json.load(open(st))['exit_code']}" if os.path.exists(st) else "no STATUS.json (in progress, or not a bb run)"
            print(f"  {cfg['runs_dir']}/{d}: {state}")


def cmd_destroy(a):
    _, cfg = project()
    if not a.names and not a.all:
        sys.exit("name the jobs to destroy, or pass --all")
    rows = by_name(cfg, a.names)
    for i in rows:
        print(f"  {job_name(i, cfg)} (id {i['id']}, {i.get('actual_status')})")
    if rows and (a.yes or input("destroy these instances and their disks? [y/N] ").strip().lower() == "y"):
        for i in rows:
            # newer CLIs want --yes, older ones reject it
            if subprocess.run(["vastai", "destroy", "instance", str(i["id"]), "--yes"], capture_output=True,
                              stdin=subprocess.DEVNULL).returncode != 0:
                subprocess.run(["vastai", "destroy", "instance", str(i["id"])], capture_output=True, input=b"y\n")
        time.sleep(3)
        left = {i["id"] for i in instances(cfg)} & {i["id"] for i in rows}
        print("destroyed" if not left else f"STILL PRESENT: {sorted(left)}; destroy them in the Vast console")


def cmd_doctor(a):
    ok = True

    def line(good, what, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok' if good else '!!'}] {what}" + (f"\n       fix: {fix}" if not good and fix else ""))

    user = vast("show", "user", check=False)
    line(isinstance(user, dict) and "id" in user, "Vast.ai login"
         + (f" (credit ${user.get('credit', 0):.2f})" if isinstance(user, dict) and "credit" in user else ""),
         "vastai set api-key <key>   (console.vast.ai, Account, API Keys; it needs instance-create permission)")
    if isinstance(user, dict) and "credit" in user:
        line(user["credit"] > 1, "Vast.ai credit above $1", "add credit in the Vast console")
    print("       note: only a real launch shows whether the key may rent")
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami()["name"]
    except Exception:
        who = None
    line(who, f"Hugging Face login{f' ({who})' if who else ''}", "hf auth login   (a token with write access)")
    line(public_key(), "SSH key in ~/.ssh (for bb logs and bb ssh)", "ssh-keygen -t ed25519")
    root = sh("git", "rev-parse", "--show-toplevel", check=False)
    line(root and os.path.exists(os.path.join(root, CONFIG_NAME)), f"inside a project with a {CONFIG_NAME}",
         "cd into the project checkout")
    if root and os.path.exists(os.path.join(root, CONFIG_NAME)):
        _, cfg = project()
        ref = sh("git", "-C", root, "rev-parse", "HEAD")
        line(sh("git", "-C", root, "branch", "-r", "--contains", ref), f"commit {ref[:8]} is pushed", "git push")
        code = sh("curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"https://github.com/{cfg['slug']}", check=False)
        line(code == "200", f"github.com/{cfg['slug']} is public (instances clone it without a login)",
             "make the project repo public; private project repos are not supported yet")
        rows = instances(cfg)
        print(f"  {len(rows)} {cfg['name']} instance(s) exist right now" + (": " + ", ".join(job_name(i, cfg) for i in rows) if rows else ""))
    print("all good" if ok else "fix the lines marked !! and run `bb doctor` again")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(prog="bb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="op", required=True)
    p = sub.add_parser("launch", help="rent one instance per job in a jobs file")
    p.add_argument("jobs")
    p.add_argument("--dry-run", action="store_true", help="show what would be rented and the cost, rent nothing")
    p.add_argument("--yes", action="store_true", help="do not ask before renting")
    p.add_argument("--gpu", default=None, help=f"override [offer] gpu in {CONFIG_NAME}, e.g. RTX_5090")
    p.add_argument("--num-gpus", type=int, default=None)
    p.add_argument("--max-price", type=float, default=None, help="$/h per instance")
    p.add_argument("--disk", type=int, default=None, help="GB")
    p.add_argument("--over-budget", action="store_true", help="allow a total above offer.max_total")
    p.add_argument("--on-done", choices=["destroy", "stop", "keep"], default=None)
    p.add_argument("--sync-s", type=int, default=None, help="seconds between uploads")
    p.add_argument("--max-hours", type=float, default=None, help="stop a job that runs longer than this")
    p.set_defaults(fn=cmd_launch)
    sub.add_parser("status", help="instances, cost and time since each last uploaded").set_defaults(fn=cmd_status)
    p = sub.add_parser("logs", help="a job's log")
    p.add_argument("name"); p.add_argument("--tail", type=int, default=40); p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(fn=cmd_logs)
    p = sub.add_parser("ssh", help="open a shell on a job's instance, or run a command there")
    p.add_argument("name"); p.add_argument("command", nargs="*"); p.set_defaults(fn=cmd_ssh)
    p = sub.add_parser("wait", help="block until the jobs end")
    p.add_argument("names", nargs="*"); p.add_argument("--every", type=int, default=120); p.set_defaults(fn=cmd_wait)
    p = sub.add_parser("collect", help="download results into the project's runs folder")
    p.add_argument("names", nargs="*"); p.add_argument("--ckpt", action="store_true", help="also download ckpt.pkl files")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("destroy", help="destroy instances and their disks")
    p.add_argument("names", nargs="*"); p.add_argument("--all", action="store_true"); p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_destroy)
    sub.add_parser("doctor", help="check logins, config and the pushed commit").set_defaults(fn=cmd_doctor)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
