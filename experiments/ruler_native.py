"""Pinned RULER-v1 data preparation for local development comparisons.

Copyright (c) 2026 Prashant Jagtap. MIT License.
The NVIDIA generators remain separate, unmodified Apache-2.0 source files.
Downloaded corpora and generated prompts stay outside this repository.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import runpy
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

SOURCE_REV = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
NEMO_REV = "f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3"
TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
         "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
         "vt", "cwe", "fwe", "qa_1", "qa_2")
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def outside_repo(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Licensed raw data must remain outside the publishing repository")
    return path


def source_files(source):
    source = outside_repo(source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source, text=True).strip()
    if revision != SOURCE_REV or changed:
        raise ValueError("Pinned unmodified RULER generator checkout required")
    return {str(p.relative_to(source)).replace("\\", "/"): sha(p)
            for p in sorted((source / "scripts").rglob("*"))
            if p.is_file() and p.suffix in (".py", ".yaml", ".txt")}


def download(url, destination, cap):
    """Bounded HTTPS download, fresh files only; record the resolved URL."""
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("HTTPS required")
    with urllib.request.urlopen(url, timeout=45) as response:
        final_url = response.geturl()
        if urllib.parse.urlparse(final_url).scheme != "https":
            raise ValueError("HTTPS downgrade refused")
        data = response.read(cap + 1)
    if len(data) > cap:
        raise ValueError("Download exceeded the declared size bound")
    with Path(destination).open("xb") as stream:
        stream.write(data)
    return dict(url=url, resolved_url=final_url, bytes=len(data), sha256=sha(destination))


def fetch(source, output):
    """Download the native sources; never silently omit a failed essay."""
    import html2text
    from bs4 import BeautifulSoup

    source, output = outside_repo(source), outside_repo(output)
    hashes = source_files(source)
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    native = source / "scripts/data/synthetic/json"
    urls = (native / "PaulGrahamEssays_URLs.txt").read_text().splitlines()
    started = time.perf_counter()

    def essay(item):
        index, original = item
        url = original.strip().replace("http://", "https://", 1)
        path = raw / f"essay-{index:03d}.bin"
        try:
            record = download(url, path, 4 * 1024 * 1024)
            html = ".html" in url
            if html:
                # Match the native downloader's decoding and HTML conversion.
                content = path.read_bytes().decode("unicode_escape", "utf-8")
                tag = BeautifulSoup(content, "html.parser").find("font")
                if tag is None:
                    raise ValueError("Native font element missing")
                parser = html2text.HTML2Text()
                parser.ignore_images = parser.ignore_tables = parser.escape_all = True
                parser.reference_links = parser.mark_code = False
                converted = parser.handle(str(tag))
            else:
                converted = path.read_bytes().decode("utf-8")
            filename = url.split("/")[-1].replace(".html", ".txt")
            return dict(index=index, original_url=original, **record, html=html,
                        filename=filename, text=converted, error=None)
        except Exception as exc:
            return dict(index=index, original_url=original, error=f"{type(exc).__name__}: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(essay, enumerate(urls)))
    write_new(output / "essay-downloads.json", [{k: v for k, v in r.items() if k != "text"} for r in records])
    errors = [r for r in records if r["error"]]
    if errors:
        raise ValueError(f"{len(errors)}/{len(records)} essays failed; retained raw files and error manifest")
    # Upstream concatenates sorted repository text files, then sorted HTML files.
    ordered = sorted(records, key=lambda r: (r["html"], r["filename"]))
    if len({(r["html"], r["filename"]) for r in ordered}) != len(ordered):
        raise ValueError("Upstream essay filename collision")
    write_new(native / "PaulGrahamEssays.json", {"text": "".join(r["text"] for r in ordered)})
    qa = {}
    for name, url in (
        ("squad", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"),
        ("hotpotqa", "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json"),
    ):
        qa[name] = download(url, native / f"{name}.json", 64 * 1024 * 1024)
    punkt = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip",
                     output / "punkt_tab.zip", 8 * 1024 * 1024)
    nltk_data = output / "nltk_data/tokenizers"
    nltk_data.mkdir(parents=True)
    with zipfile.ZipFile(output / "punkt_tab.zip") as archive:
        for name in archive.namelist():
            target = (nltk_data / name).resolve()
            if nltk_data.resolve() not in target.parents or ".." in Path(name).parts:
                raise ValueError("Unsafe tokenizer archive path")
        archive.extractall(nltk_data)
    manifest = dict(schema=1, source_revision=SOURCE_REV, nemo_revision=NEMO_REV, source_hashes=hashes,
                    essays=len(records), qa=qa, punkt=punkt,
                    corpora={name: sha(native / name) for name in
                             ("PaulGrahamEssays.json", "squad.json", "hotpotqa.json", "english_words.json")},
                    seconds=time.perf_counter()-started,
                    distribution="Raw essays/prompts remain local. SQuAD and HotpotQA: CC-BY-SA-4.0.",
                    deviations=["HTTPS for upstream HTTP essay URLs; download failures are fatal",
                                "Pinned upstream-listed HotpotQA HTTPS mirror; no insecure fallback",
                                "Data URLs may be mutable; exact downloaded bytes are locally hash-pinned"])
    write_new(output / "manifest.json", manifest)
    print(json.dumps({"essays": len(records), "corpora": manifest["corpora"], "seconds": manifest["seconds"]}))


def prepare(source, assets, tokenizer, output, count, seed, lengths, qa_offset=0, seed_stride=0, qa_offset_stride=0):
    """Call original generators using argument arrays, with bounded subprocesses."""
    import yaml
    from transformers import AutoTokenizer

    source, assets, output = map(outside_repo, (source, assets, output))
    hashes = source_files(source)
    manifest = json.loads((assets / "manifest.json").read_bytes())
    native = source / "scripts/data/synthetic/json"
    if any(sha(native / name) != expected for name, expected in manifest["corpora"].items()):
        raise ValueError("Corpora no longer match downloaded artifacts")
    if not 1 <= count <= 500 or not lengths or any(n not in (4096, 8192, 16384, 32768) for n in lengths):
        raise ValueError("Bounded registered length and sample count required")
    if any(type(v) is not int or not 0 <= v <= 100000 for v in (qa_offset, seed_stride, qa_offset_stride)):
        raise ValueError("Nonnegative bounded offsets/seed strides required")
    tokenizer = Path(tokenizer).resolve()
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True, trust_remote_code=False)
    base_tokens = len(tok.tokenize("{task_template}"))
    constants = runpy.run_path(str(source / "scripts/data/synthetic/constants.py"))["TASKS"]
    config = yaml.safe_load((source / "scripts/synthetic.yaml").read_text())
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1", NLTK_DATA=str(assets / "nltk_data"))
    plan = dict(schema=1, purpose="RULER-v1 development data; custom local orchestration, no leaderboard claim",
                source_revision=SOURCE_REV, nemo_revision=NEMO_REV, source_hashes=hashes,
                runner_sha256=sha(__file__), assets_sha256=sha(assets / "manifest.json"),
                tokenizer_sha256={n: sha(tokenizer / n) for n in ("tokenizer.json", "tokenizer_config.json", "config.json")},
                count_per_task_length=count, seed=seed, lengths=lengths, tasks=TASKS,
                qa_offset=qa_offset, seed_stride=seed_stride, qa_offset_stride=qa_offset_stride,
                template_reserve=50, native_base_template_tokens=base_tokens,
                packages={p: importlib.metadata.version(p) for p in
                          ("transformers", "tokenizers", "numpy", "nltk", "wonderwords", "scipy", "tenacity", "PyYAML")},
                answer_isolation="Separate public inputs and scoring keys. Never pass outputs or token_position_answer to selectors.",
                deviations=["Original generators called directly without NeMo shell/pip wrapper",
                            "Local pinned Qwen tokenizer; offline, no remotely loaded model code",
                            "Same NeMo 50-token reserve and native base-template-token subtraction",
                            "Smaller development sample count; not a complete official benchmark run"])
    write_new(output / "plan.json", plan)
    inventory = []
    for length_index, length in enumerate(lengths):
        for task in TASKS:
            spec = config[task]
            constant = constants[spec["task"]]
            directory = output / str(length)
            args = [sys.executable, str(source / f"scripts/data/synthetic/{spec['task']}.py"),
                    "--save_dir", str(directory), "--save_name", task, "--subset", "test",
                    "--tokenizer_path", str(tokenizer), "--tokenizer_type", "hf",
                    "--max_seq_length", str(length-50), "--num_samples", str(count),
                    "--random_seed", str(seed + length_index * seed_stride), "--model_template_token", str(base_tokens),
                    "--tokens_to_generate", str(constant["tokens_to_generate"]),
                    "--template", constant["template"] + constant.get("answer_prefix", "")]
            for key, value in spec["args"].items():
                args.extend([f"--{key}", str(value)])
            if spec["task"] == "qa":
                args.extend(["--pre_samples", str(qa_offset + length_index * qa_offset_stride)])
            start = time.perf_counter()
            with (output / f"{length}-{task}.log").open("xb") as log:
                completed = subprocess.run(args, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=240, check=False)
            row = dict(length=length, task=task, seconds=time.perf_counter()-start, exit_code=completed.returncode,
                       seed=seed + length_index * seed_stride,
                       qa_offset=qa_offset + length_index * qa_offset_stride if spec["task"] == "qa" else None)
            path = directory / task / "test.jsonl"
            if completed.returncode != 0 or not path.is_file():
                write_new(output / "failure.json", row)
                raise ValueError(f"Native generation failed: {length}/{task}; log retained")
            samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(samples) != count:
                raise ValueError("Native task count differs")
            # Use row position: upstream NIAH 'index' is the answer character offset, not a unique ID.
            public, keys = [], []
            for i, sample in enumerate(samples):
                key = f"{length}-{task}-{i:04d}"
                question = sample["input"] + sample["answer_prefix"]
                public.append(dict(id=key, task=task, length=length, question=question))
                keys.append(dict(id=key, expected_answer=sample["outputs"],
                                 match_type="part" if task.startswith("qa_") else "all"))
            write_new(directory / task / "inputs.json", public)
            write_new(directory / task / "keys.json", keys)
            row.update(raw_sha256=sha(path), inputs_sha256=sha(directory/task/"inputs.json"),
                       keys_sha256=sha(directory/task/"keys.json"), count=len(samples))
            inventory.append(row)
            print(json.dumps(row), flush=True)
    write_new(output / "inventory.json", inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "prepare"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--qa-offset", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=0)
    parser.add_argument("--qa-offset-stride", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.source, args.output)
    else:
        if args.assets is None or args.tokenizer is None:
            parser.error("prepare requires --assets and --tokenizer")
        prepare(args.source, args.assets, args.tokenizer, args.output, args.count, args.seed, args.lengths,
                args.qa_offset, args.seed_stride, args.qa_offset_stride)
