
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kdagent.analysis.signals import (
    free_observations,
    signal_s1,
    signal_s2,
    signal_s3,
    signal_s4,
    signal_s5,
    signal_s6,
    signal_s7,
    signal_s8,
)
from kdagent.analysis.sources import (
    budget_params,
    console_summary,
    cost_classes,
    facade_defaults,
    llm_facts,
    load_trace,
    search_rows,
    search_seconds,
    tool_calls,
    truncated_replies,
    wallclock_seconds,
)


def report(root: Path) -> str:
    classes, budget, defaults = cost_classes(), budget_params(), facade_defaults()
    out: list[str] = [f"# Signal extraction for {root}\n"]
    facts_rows, signal_rows, new_rows, table_rows = [], [], [], []

    for workspace in sorted(p.parent for p in root.glob("*/trace.jsonl")):
        name = workspace.name
        rows = load_trace(workspace)
        calls = tool_calls(rows)
        llm = llm_facts(rows)
        console = console_summary(root / f"{name}.log")
        free = free_observations(calls)
        s1 = signal_s1(calls)
        s2 = signal_s2(calls)
        s3 = signal_s3(calls, budget, defaults)
        s4 = signal_s4(calls, llm["texts"], classes)


        rows_search = search_rows(workspace)
        s5 = signal_s5(calls)
        s6 = signal_s6(name, calls)
        s7 = signal_s7(name, console, rows_search, workspace)
        s8 = signal_s8(calls)

        minutes = wallclock_seconds(rows) / 60
        inp, outp = llm["input_tokens"] / 1000, llm["output_tokens"] / 1000
        tokens = f"{inp:.0f}k / {outp:.0f}k"
        stop = console.get("stop_reason", "(running)")
        facts_rows.append(
            f"| {name} | {minutes:.1f} min | {llm['model_calls']} | "
            f"{truncated_replies(rows)} | {tokens} | "
            f"{stop} | {console.get('answer_source', '')} | {free['tool_calls']} | "
            f"{free['path_not_found']} | {free['skill_reads']} |"
        )
        budget_note = (
            f"{len(s3['requested'])} searches, {s3['with_budget']} with a budget"
        )
        signal_rows.append(
            f"| {name} | {s1['verdict']} ({s1['submits']} submitted, "
            f"{s1['repeats']} repeated) | "
            f"{s2['verdict']} ({len(s2['events'])} named) | "
            f"{s3['verdict']} ({budget_note}) | "
            f"{s4['verdict']} ({'/'.join(s4['cost_classes']) or 'none'}) |"
        )
        new_rows.append(
            f"| {name} | {s5['verdict']} ({len(s5['reads'])} read) | "
            f"{s6['verdict']} ({len(s6['bad'])} invented) | "
            f"{s7['verdict']} (answer from {s7['answer_from'] or 'none'}, "
            f"nmse {s7['answer_nmse']}) | "
            f"{s8['verdict']} ({s8['shell_calls']} shell calls, "
            f"{s8['api_guess_failures']} wrong guesses, "
            f"{s8['file_hunting']} file hunts) |"
        )
        for row in rows_search:
            seconds = search_seconds(row)

            expression = row.get("best_expression") or " + ".join(
                (row.get("law") or {}).get("support") or ()
            )
            table_rows.append(
                f"| {name} | {row['instrument']} | "
                f"{'' if seconds is None else f'{seconds:.0f}s'} | "
                f"{row.get('nmse')} | `{expression}` |"
            )

        out.append(f"\n## {name}\n")
        out.append(f"- answer: `{console.get('answer', '(running)')}`")


        served = f"{console.get('endpoint')} / {console.get('model')}"
        out.append(f"- endpoint/model: {served}")
        out.append(f"- S2 events: {json.dumps(s2['events'], ensure_ascii=False)}")
        out.append(f"- S3 requested: {json.dumps(s3['requested'], ensure_ascii=False)}")
        escalations = json.dumps(s3["escalations"], ensure_ascii=False)
        out.append(f"- S3 after an underfit: {escalations}")
        out.append(
            f"- S4 cost mentions: {json.dumps(s4['mentions'], ensure_ascii=False)}"
        )
        out.append(f"- S5 cards read: {json.dumps(s5, ensure_ascii=False)}")
        out.append(f"- S6 terms written: {json.dumps(s6['terms'], ensure_ascii=False)}")
        out.append(
            f"- S6 invented spellings: {json.dumps(s6['bad'], ensure_ascii=False)}"
        )
        out.append(f"- S7: {json.dumps(s7, ensure_ascii=False)}")

    out.insert(1, "\n## Facts per run\n")
    out.insert(
        2,
        "| Dataset | Wall clock | Model calls | Truncated replies | Tokens in/out "
        "| stop_reason | answer_source | Tool calls | path_not_found | Card reads |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(facts_rows),
    )
    out.insert(3, "\n## The four signals\n")
    out.insert(
        4,
        "| Dataset | S1 repeat submission | S2 retried as told | S3 budget passed "
        "| S4 cost consumed |\n"
        "|---|---|---|---|---|\n" + "\n".join(signal_rows),
    )
    out.insert(5, "\n## The four signals added this round\n")
    out.insert(
        6,
        "| Dataset | S5 card read | S6 term spelling | S7 algorithm-dataset match "
        "| S8 data groping |\n"
        "|---|---|---|---|---|\n" + "\n".join(new_rows),
    )
    out.insert(7, "\n## Every search\n")
    out.insert(
        8,
        "| Dataset | Instrument | Wall clock | nmse | Expression |\n"
        "|---|---|---|---|---|\n" + "\n".join(table_rows),
    )
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="kdagent.analysis")
    parser.add_argument(
        "root", type=Path, help="root directory of one experiment round"
    )
    args = parser.parse_args()
    sys.stdout.write(report(args.root))
