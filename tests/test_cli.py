import argparse
import base64
import json
import subprocess
import os

import pytest

from boxboy import cli


def test_parse_jobs():
    jobs = cli.parse_jobs("# c\n\na_1: python -m x.train --out runs/a_1 --lr 1e-4\nb: echo hi > runs/b/x\n")
    assert jobs == [("a_1", "python -m x.train --out runs/a_1 --lr 1e-4"), ("b", "echo hi > runs/b/x")]
    assert cli.parse_jobs("a: train --out out/a", runs_dir="out") == [("a", "train --out out/a")]


@pytest.mark.parametrize("text", ["no colon here", "a: x runs/a\na: y runs/a", "a: python train.py --out runs/other",
                                  "a: python train.py --out runs/a2"])
def test_parse_jobs_rejects(text):
    with pytest.raises(ValueError):
        cli.parse_jobs(text)


def test_config_overlays_defaults_and_rejects_typos():
    cfg = cli.merge_config({"name": "p", "offer": {"gpu": "RTX_5090"}, "env": {"A": "1"}})
    assert cfg["offer"]["gpu"] == "RTX_5090" and cfg["offer"]["disk"] == 30 and cfg["run"]["on_done"] == "destroy"
    assert cli.DEFAULTS["offer"]["gpu"] == "RTX_4090"            # defaults not mutated
    for bad in ({"imge": "x"}, {"offer": {"max_prise": 1}}, {"run": {"on_done": "explode"}}):
        with pytest.raises(ValueError):
            cli.merge_config(bad)


def _job(**kw):
    a = argparse.Namespace(on_done=None, sync_s=None, max_hours=None)
    cfg = cli.merge_config({"name": "p", "setup": "pip install -r r.txt", "env": {"K": "v w"}})
    cfg["slug"] = "u/p"
    return cli.make_job("a", "python x.py --out 'runs/a' && echo \"done\"", "abc123", cfg, "u/p-runs", a)


def test_onstart_carries_the_job_and_fetches_the_worker_with_the_token():
    job = _job()
    cmd = cli.onstart_cmd(job, "deadbeef0000")
    assert "https://huggingface.co/u/p-runs/resolve/main/_boxboy/deadbeef0000/worker.sh" in cmd
    assert "Authorization: Bearer $T" in cmd and "hf_" not in cmd       # the token itself is never in the command
    b64 = cmd.split("BB_JOB_B64=")[1].split(";")[0]
    assert json.loads(base64.b64decode(b64)) == job
    assert job["repo_url"] == "https://github.com/u/p.git" and job["max_hours"] == 24
    assert len(cmd) < 4000          # onstart commands have a size limit
    assert subprocess.run(["bash", "-n", "-c", cmd]).returncode == 0


def test_worker_reads_every_field_the_job_has():
    """worker.sh pulls fields out of the job by name; a field it asks for must exist."""
    src = open(os.path.join(cli.WORKER_DIR, "worker.sh")).read()
    import re
    asked = set(re.findall(r"\$\(job (\w+)\)", src))
    assert asked and asked <= set(_job()), asked - set(_job())
    assert subprocess.run(["bash", "-n", os.path.join(cli.WORKER_DIR, "worker.sh")]).returncode == 0


def test_offer_query_and_slug():
    q = cli.offer_query(cli.DEFAULTS["offer"])
    assert "gpu_name=RTX_4090" in q and "dph_total<=0.6" in q and "notin [CN]" in q
    for url in ("git@github.com:u/p.git", "https://github.com/u/p.git", "https://github.com/u/p"):
        assert cli.repo_slug(url) == "u/p"


def test_bundle_id_is_stable():
    assert cli.bundle_id() == cli.bundle_id() and len(cli.bundle_id()) == 12


def test_ssh_args_batch_and_interactive(monkeypatch):
    monkeypatch.setattr(cli, "sh", lambda *a, **k: "warning: old cli\nssh://root@1.2.3.4:2222")
    batch, inter = cli.ssh_args({"id": 1}), cli.ssh_args({"id": 1}, batch=False)
    assert "BatchMode=yes" in batch and "BatchMode=yes" not in inter
    for args in (batch, inter):           # every -o has a value after it
        assert all(not args[k + 1].startswith("-") for k, x in enumerate(args) if x == "-o")
        assert args[-3:] == ["-p", "2222", "root@1.2.3.4"]


def test_host_cuda_floor_follows_the_image():
    assert cli.image_cuda("vastai/pytorch:cuda-13.2.1-auto") == 13.2 and cli.image_cuda("python:3.12") is None
    o = dict(cli.DEFAULTS["offer"])
    assert cli.min_host_cuda({"image": "vastai/pytorch:cuda-13.2.1-auto"}, o) == 13.2
    assert cli.min_host_cuda({"image": "python:3.12"}, o) == o["min_cuda"]
