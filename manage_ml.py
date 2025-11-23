# -*- coding: utf-8 -*-
# manage_ml.py
"""
ML-оркестратор с прогресс-барами и ETA (ASCII-совместимый вывод).
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TimeElapsedColumn,
    TimeRemainingColumn, SpinnerColumn, TextColumn
)

import config

console = Console()
ROOT = Path(__file__).resolve().parent

DUR_DB = ROOT / ".manage_ml_durations.json"
STEP_KEYS = {
    "dataset": "build_dataset",
    "train": "train_model",
    "sanity": "sanity_check",
    "fetch": "fetch_history",
}

SCRIPT_DATASET = ROOT / "build_ml_dataset_from_fills.py"
SCRIPT_TRAIN   = ROOT / "retrain_model_from_dataset.py"
SCRIPT_SANITY  = ROOT / "sanity_check.py"

MODEL_FILE = getattr(config, "MODEL_FILE", "rf_model.pkl")
MODEL_META = getattr(config, "MODEL_META", "model_meta.json")

def load_durations() -> dict:
    if DUR_DB.exists():
        try:
            return json.loads(DUR_DB.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_durations(db: dict):
    try:
        DUR_DB.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def run_with_progress(
    title: str,
    cmd: List[str],
    step_key: str,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> Tuple[int, str]:
    durations = load_durations()
    avg = float(durations.get(step_key, 0.0)) or 0.0

    columns = [
        SpinnerColumn(),
        TextColumn("[bold]" + title + "[/bold]"),
        BarColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ]
    progress = Progress(*columns, transient=True)

    task_id = progress.add_task(title, total=100.0)
    start = time.time()

    sp_env = os.environ.copy()
    if env:
        sp_env.update(env)
    # гарантируем текстовый режим и мягкую декодировку
    sp_env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=getattr(config, "SUBPROC_ENCODING", "utf-8"),
        errors="replace",
        bufsize=1,
        env=sp_env,
        cwd=cwd
    )

    captured_lines: List[str] = []
    last_update = time.time()

    with progress:
        while True:
            line = proc.stdout.readline()
            now = time.time()
            if avg > 0:
                elapsed = now - start
                pct = min(99.0, (elapsed / avg) * 100.0)
            else:
                elapsed = now - start
                pct = min(80.0, elapsed * 2.5)

            if now - last_update > 0.1:
                progress.update(task_id, completed=pct)
                last_update = now

            if line == "" and proc.poll() is not None:
                break
            if line:
                captured_lines.append(line.rstrip())
                console.print(line.rstrip())

        progress.update(task_id, completed=100.0)

    rc = proc.wait()

    elapsed_total = time.time() - start
    if rc == 0 and elapsed_total > 0.5:
        if avg > 0:
            durations[step_key] = 0.7 * avg + 0.3 * elapsed_total
        else:
            durations[step_key] = elapsed_total
        save_durations(durations)

    return rc, "\n".join(captured_lines)

def step_dataset(csv_path: Optional[str], since: Optional[str]) -> None:
    if not SCRIPT_DATASET.exists():
        console.print("[yellow]build_ml_dataset_from_fills.py ne naiden. Propusk dataset.[/]")
        return
    cmd = ["python", str(SCRIPT_DATASET)]
    env = {}
    if csv_path:
        env["FILLS_PATH"] = csv_path
        env["BYBIT_CSV_PATH"] = csv_path
    if since:
        env["DATASET_SINCE"] = since
    rc, _ = run_with_progress("dataset", cmd, STEP_KEYS["dataset"], env=env or None)
    if rc != 0:
        raise SystemExit("[red]Shag dataset zavershilsya s oshibkoi.[/]")

def step_train() -> None:
    if not SCRIPT_TRAIN.exists():
        console.print("[yellow]retrain_model_from_dataset.py ne naiden. Propusk train.[/]")
        return
    cmd = ["python", str(SCRIPT_TRAIN)]
    rc, _ = run_with_progress("train", cmd, STEP_KEYS["train"])
    if rc != 0:
        raise SystemExit("[red]Shag train zavershilsya s oshibkoi.[/]")

def step_sanity() -> None:
    if not SCRIPT_SANITY.exists():
        console.print("[yellow]sanity_check.py ne naiden — shag sanity propushchen.[/]")
        return
    cmd = ["python", str(SCRIPT_SANITY)]
    rc, _ = run_with_progress("sanity", cmd, STEP_KEYS["sanity"])
    if rc != 0:
        console.print("[yellow]sanity vernul oshibku. Prover provodku vyshe.[/]")

def show_stats() -> None:
    table = Table(title="ML / Dataset Stats", show_lines=True)
    table.add_column("Chto", justify="left", style="cyan", no_wrap=True)
    table.add_column("Znachenie", justify="left", style="white")

    meta_path = ROOT / MODEL_META
    model_path = ROOT / MODEL_FILE

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}

    feats = meta.get("features", [])
    thresholds = meta.get("thresholds", {})
    thr_used = thresholds.get("used") or thresholds.get("global")

    table.add_row("Model", str(model_path if model_path.exists() else f"{MODEL_FILE} (net faila)"))
    table.add_row("Meta", str(meta_path if meta_path.exists() else f"{MODEL_META} (net faila)"))
    table.add_row("Kol-vo fich", str(len(feats)))
    table.add_row("Porog (thr)", str(thr_used) if thr_used is not None else "-")
    if thresholds:
        table.add_row("Porogi (raw)", json.dumps(thresholds, ensure_ascii=False))

    durs = load_durations()
    if durs:
        pretty = ", ".join([f"{k}≈{int(v)}s" for k, v in durs.items()])
        table.add_row("ETA-pamyat", pretty)

    console.print(table)

def cli():
    parser = argparse.ArgumentParser(description="ML-orkestrator s progress-barami")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="polnyi cikl: dataset -> train -> (opc) sanity")
    p_run.add_argument("--csv", type=str, default=None, help="put k fills_all.csv (ili drugomu CSV)")
    p_run.add_argument("--since", type=str, default=None, help="ogranichit istoriyu po date (YYYY-MM-DD)")
    p_run.add_argument("--no-sanity", action="store_true", help="ne zapuskat sanity_check.py")

    p_ds = sub.add_parser("dataset", help="tolko sobrat dataset")
    p_ds.add_argument("--csv", type=str, default=None, help="put k fills_all.csv (ili drugomu CSV)")
    p_ds.add_argument("--since", type=str, default=None, help="ogranichit istoriyu po date (YYYY-MM-DD)")

    sub.add_parser("train", help="tolko trenirovku modeli")
    sub.add_parser("sanity", help="tolko sanity_check.py")
    sub.add_parser("stats", help="korotkaya svodka po datasetu/modeli")
    sub.add_parser("clean", help="sbrosit ETA-pamyat")

    args = parser.parse_args()

    if args.cmd == "clean":
        if DUR_DB.exists():
            DUR_DB.unlink(missing_ok=True)
        console.print("[green]OK. Istoriya dlitelnostei ochishchena.[/]")
        return

    if args.cmd == "stats":
        show_stats()
        return

    if args.cmd == "dataset":
        step_dataset(args.csv, args.since)
        console.print(Panel.fit("[green]OK: Dataset gotov.[/]", title="Gotovo"))
        return

    if args.cmd == "train":
        step_train()
        console.print(Panel.fit("[green]OK: Trening zavershen.[/]", title="Gotovo"))
        return

    if args.cmd == "sanity":
        step_sanity()
        return

    if args.cmd == "run":
        console.print(Panel.fit("Zapusk polnogo cikla", title="manage_ml"))
        step_dataset(args.csv, args.since)
        step_train()
        if not args.no_sanity:
            step_sanity()
        console.print(Panel.fit("[bold green]Gotovo![/] Model i meta obnovleny.", title="FINISH"))
        show_stats()
        return

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[red]Ostanovleno polzovatelem.[/]")
