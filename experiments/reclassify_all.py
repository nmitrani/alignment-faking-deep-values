#!/usr/bin/env python3
"""Reclassify every results_*.json under outputs/ with a vLLM-served classifier.

Walks the entire source tree (default: outputs/), mirrors its directory
structure into outputs_<classifier_name>/, and runs the NewClassify pipeline
(strong_reject + deceptive_alignment + partial_compliance + helpfulness_categories
+ deceptive_alignment_categories) on every trace, optionally also
instrumental_goal_guarding + contractor_sycophancy on AW paths.

The destination JSON preserves every original field; only classifier outputs
and `timestamp` are overwritten. Traces that are missing required fields (or
that hit a classifier error) are passed through unchanged with a
`reclassify_skipped` sentinel, so destination file length and order match the
source — downstream tools (compliance_gap.py et al.) keep working.

The classifier endpoint is passed explicitly via --base_url, so multiple
instances of this script can run in parallel against different vLLM servers
without colliding on the singleton `.vllm_endpoint` file.

Usage:
    python -m experiments.reclassify_all \\
        --classifier_model google/gemma-4-31b-it \\
        --classifier_name gemma-4-31b-it \\
        --base_url http://localhost:8235/v1
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.inference import InferenceAPI
from src.pipeline.classify import NewClassify
from src.pipeline.evals import MajorityVoteClassifier

from experiments.classify_results import create_igg_classifier, create_cs_classifier


CORE_CLASSIFIER_KEYS = {
    "strong_reject_score",
    "strong_reject_raw_output",
    "deceptive_alignment",
    "partial_compliance",
    "helpfulness_categories",
    "deceptive_alignment_categories",
}


def matches_any(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch(rel, p) for p in patterns)


def discover_files(
    source_root: Path,
    include: list[str],
    exclude: list[str],
) -> list[Path]:
    """Walk source_root for results_*.json, filtered by include/exclude globs."""
    files = []
    for p in source_root.rglob("results_*.json"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(source_root))
        if include and not matches_any(rel, include):
            continue
        if exclude and matches_any(rel, exclude):
            continue
        files.append(p)
    return sorted(files)


def dest_path(src: Path, source_root: Path, dest_root: Path) -> Path:
    return dest_root / src.relative_to(source_root)


def is_complete(dest: Path, src_len: int, expected_keys: set[str]) -> bool:
    """A destination file is complete iff it parses, length matches, and every
    record either has all expected classifier keys or is a recorded skip."""
    if not dest.exists():
        return False
    try:
        data = json.loads(dest.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, list) or len(data) != src_len:
        return False
    for r in data:
        if not isinstance(r, dict):
            return False
        if r.get("reclassify_skipped"):
            continue
        if not expected_keys.issubset(r.keys()):
            return False
    return True


def is_animal_welfare_path(rel_path: Path) -> bool:
    """Auto-detect AW vs helpful-only from a path under source_root."""
    s = str(rel_path).lower()
    if "animal-welfare" in s or "animal_welfare" in s:
        return True
    if "helpful-only" in s:
        return False
    if "sycophancy" in s:
        return True
    return True


def resolve_dest_root(args) -> Path:
    if args.dest_root:
        return Path(args.dest_root)
    safe_name = args.classifier_name.replace("/", "_")
    return Path(f"{args.dest_root_prefix}{safe_name}")


async def reclassify_trace(
    record: dict,
    classifier: NewClassify,
    igg_classifier: Optional[MajorityVoteClassifier] = None,
    cs_classifier: Optional[MajorityVoteClassifier] = None,
) -> dict:
    """Reclassify a single trace.

    On a required-field error, return the original record with a
    `reclassify_skipped` sentinel and a fresh timestamp, so the destination
    file shape matches the source.
    """
    try:
        out = await classifier.classify(dict(record))
    except ValueError as e:
        return {
            **record,
            "reclassify_skipped": f"missing_field:{e}",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            **record,
            "reclassify_skipped": f"error:{type(e).__name__}:{e}",
            "timestamp": datetime.now().isoformat(),
        }

    output_text = record.get("output")
    if (igg_classifier or cs_classifier) and output_text:
        try:
            tasks = []
            if igg_classifier:
                tasks.append(igg_classifier.process_single_response(output_text))
            if cs_classifier:
                tasks.append(cs_classifier.process_single_response(output_text))
            results = await asyncio.gather(*tasks)
            idx = 0
            if igg_classifier:
                out["instrumental_goal_guarding"] = results[idx]
                idx += 1
            if cs_classifier:
                out["contractor_sycophancy"] = results[idx]
                idx += 1
        except Exception:
            pass
    return out


async def reclassify_file(
    src: Path,
    dest: Path,
    classifier: NewClassify,
    igg_classifier: Optional[MajorityVoteClassifier] = None,
    cs_classifier: Optional[MajorityVoteClassifier] = None,
) -> tuple[int, int]:
    """Reclassify one file. Returns (n_classified, n_skipped)."""
    with open(src) as f:
        src_records = json.load(f)
    if not isinstance(src_records, list):
        raise ValueError(f"{src}: expected JSON array, got {type(src_records).__name__}")

    new_records = await asyncio.gather(
        *[reclassify_trace(r, classifier, igg_classifier, cs_classifier) for r in src_records]
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(new_records, f, indent=2, default=str)
    os.replace(tmp, dest)

    n_skipped = sum(1 for r in new_records if r.get("reclassify_skipped"))
    return (len(new_records) - n_skipped, n_skipped)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reclassify every results_*.json under outputs/ with a vLLM-served classifier",
    )
    parser.add_argument("--classifier_model", required=True, help="Model id served by vLLM (e.g. google/gemma-4-31b-it)")
    parser.add_argument("--classifier_name", required=True, help="Short name used in outputs_<name>/")
    parser.add_argument("--base_url", required=True, help="vLLM endpoint URL (e.g. http://host:8235/v1)")
    parser.add_argument("--source_root", default="outputs", help="Source tree to walk (default: %(default)s)")
    parser.add_argument("--dest_root_prefix", default="outputs_", help="Destination root prefix (default: %(default)s)")
    parser.add_argument("--dest_root", default=None, help="Override destination root entirely")
    parser.add_argument("--threads", type=int, default=64, help="Concurrent API calls (default: %(default)s)")
    parser.add_argument("--include", action="append", default=[], help="Glob filter relative to source_root; repeatable")
    parser.add_argument("--exclude", action="append", default=[], help="Glob exclusion relative to source_root; repeatable")
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true", default=True,
                        help="Skip files whose destination is already complete (default)")
    parser.add_argument("--force", dest="skip_existing", action="store_false",
                        help="Reprocess every matching file even if destination is complete")
    parser.add_argument("--dry_run", action="store_true", help="List files that would be processed; no LLM calls or writes")
    parser.add_argument("--max_files", type=int, default=0, help="Cap on files processed (0 = unlimited; for benchmarks)")
    parser.add_argument("--animal_welfare_mode", choices=["auto", "always", "never"], default="auto",
                        help="Choose AW vs helpful-only classifier prompts (default: %(default)s)")
    parser.add_argument("--include_extra", action="store_true",
                        help="Also run instrumental_goal_guarding + contractor_sycophancy on AW traces")
    return parser.parse_args()


async def main_async(args):
    source_root = Path(args.source_root).resolve()
    dest_root = resolve_dest_root(args).resolve()

    if not source_root.exists():
        print(f"ERROR: source root does not exist: {source_root}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  Reclassify-all Pipeline")
    print("=" * 70)
    print(f"  Classifier:    {args.classifier_model}")
    print(f"  Source root:   {source_root}")
    print(f"  Dest root:     {dest_root}")
    print(f"  Base URL:      {args.base_url}")
    print(f"  Threads:       {args.threads}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  AW mode:       {args.animal_welfare_mode}")
    print(f"  Include extra: {args.include_extra}")
    if args.include:
        print(f"  Include globs: {args.include}")
    if args.exclude:
        print(f"  Exclude globs: {args.exclude}")
    print()

    all_files = discover_files(source_root, args.include, args.exclude)
    print(f"  Discovered {len(all_files)} files matching filters.")

    todo: list[tuple[Path, Path]] = []
    skipped_complete = 0
    for src in all_files:
        dest = dest_path(src, source_root, dest_root)
        if args.skip_existing and dest.exists():
            try:
                with open(src) as f:
                    src_data = json.load(f)
                if isinstance(src_data, list) and is_complete(dest, len(src_data), CORE_CLASSIFIER_KEYS):
                    skipped_complete += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        todo.append((src, dest))

    print(f"  Already complete (skip_existing): {skipped_complete}")
    print(f"  To process: {len(todo)}")

    if args.max_files > 0:
        todo = todo[: args.max_files]
        print(f"  Limited to first {len(todo)} files (--max_files)")

    if args.dry_run:
        print("\n[DRY-RUN] Files that would be processed:")
        for src, dest in todo:
            print(f"  {src.relative_to(source_root)}  ->  {dest}")
        return

    if not todo:
        print("Nothing to do.")
        return

    api = InferenceAPI(
        base_url=args.base_url,
        api_key="not-needed",
        num_threads=args.threads,
        use_local_vllm=False,
    )

    cls_aw = NewClassify(
        api=api,
        use_hidden_scratchpad=True,
        classifier_model_id=args.classifier_model,
        animal_welfare=True,
    )
    cls_ho = NewClassify(
        api=api,
        use_hidden_scratchpad=True,
        classifier_model_id=args.classifier_model,
        animal_welfare=False,
    )
    igg = create_igg_classifier(api, args.classifier_model) if args.include_extra else None
    cs = create_cs_classifier(api, args.classifier_model) if args.include_extra else None

    t0 = time.time()
    n_files_done = 0
    n_traces = 0
    n_skipped_traces = 0
    n_file_errors = 0

    for i, (src, dest) in enumerate(todo, 1):
        rel = src.relative_to(source_root)
        if args.animal_welfare_mode == "auto":
            is_aw = is_animal_welfare_path(rel)
        else:
            is_aw = args.animal_welfare_mode == "always"

        classifier = cls_aw if is_aw else cls_ho
        igg_use = igg if is_aw else None
        cs_use = cs if is_aw else None

        elapsed = time.time() - t0
        rate = n_files_done / elapsed if elapsed > 0 else 0.0
        eta_min = ((len(todo) - n_files_done) / rate / 60.0) if rate > 0 else float("inf")
        print(f"[{i}/{len(todo)}] ({rate:.2f} files/s, ETA {eta_min:.1f}m) AW={is_aw}  {rel}")

        try:
            n_cls, n_skp = await reclassify_file(src, dest, classifier, igg_use, cs_use)
            n_traces += n_cls
            n_skipped_traces += n_skp
            n_files_done += 1
            if n_skp > 0:
                print(f"    [warn] {n_skp} traces skipped (missing fields or classifier error)", file=sys.stderr)
        except Exception as e:
            n_file_errors += 1
            print(f"    [error] {type(e).__name__}: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("  Done")
    print("=" * 70)
    print(f"  Files processed: {n_files_done}/{len(todo)}")
    print(f"  Files errored:   {n_file_errors}")
    print(f"  Traces classified: {n_traces}")
    print(f"  Traces skipped:    {n_skipped_traces}")
    print(f"  Wall-clock:        {elapsed / 60:.1f} min")
    if n_files_done > 0:
        print(f"  Avg per-file:      {elapsed / n_files_done:.1f}s")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
