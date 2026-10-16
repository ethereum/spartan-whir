#!/usr/bin/env python3
"""Rank Poseidon Plonky3-WHIR direct-sparse schedules.

The Rust candidate dumper derives WHIR security/query/PoW values from the
backend. This script only scores already-derived candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_MAX_POW_BITS = 22
DEFAULT_SECURITY_BITS = 116
DEFAULT_ZK_ELL_SWEEP = [3, 4, 8, 16]
DEFAULT_ZK_MASK_LOG_INV_RATE_SWEEP = [1, 2, 3, 4, 5]
DEFAULT_VALIDATION_TOLERANCE = 0.20
DEFAULT_SHORTLIST_MARGIN_RATIO = 0.01
TIME_TIE_RELATIVE = 0.01
TIME_TIE_EPSILON = 1.0e-12

COMPONENTS = (
    ("spartan", "constraint_work"),
    ("dft", "dft_work"),
    ("merkle", "merkle_work"),
    ("merkle_path", "merkle_path_work"),
    ("row_opening", "row_work"),
    ("sumcheck", "sumcheck_work"),
    ("pow", "pow_work_units"),
)

VERIFIER_COMPONENTS = (
    ("merkle_hash", "verifier_merkle_hashes"),
    ("leaf_field_element", "verifier_leaf_field_elements"),
    ("row_field_element", "verifier_row_field_elements"),
    ("extension_operation", "verifier_extension_operations"),
    ("pow_check", "verifier_pow_checks"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", help="Candidate JSON from poseidon-schedule-candidates")
    parser.add_argument("--num-variables", type=int, help="Generate candidates for this size")
    parser.add_argument("--field", default="koalabear", help="Field profile: koalabear or babybear")
    parser.add_argument(
        "--extension",
        choices=("all", "quartic", "quintic", "octic"),
        default="all",
        help="Extension candidates to generate or retain; default: all",
    )
    parser.add_argument("--calibration", required=True, help="Component calibration JSON")
    parser.add_argument("--out-report", required=True, help="Ranked report JSON path")
    parser.add_argument(
        "--out-config",
        help="Selected DirectSparse setup config; omit for component searches",
    )
    parser.add_argument("--max-pow-bits", type=int, default=DEFAULT_MAX_POW_BITS)
    parser.add_argument(
        "--round-log-inv-rate-offset-max",
        type=int,
        default=0,
        help=(
            "Search uniform explicit round-rate offsets from 1 through this value; "
            "offset zero retains backend-derived rates"
        ),
    )
    parser.add_argument("--security-bits", type=int, default=DEFAULT_SECURITY_BITS)
    parser.add_argument("--merkle-security-bits", type=int)
    parser.add_argument("--component-security-bits", type=int)
    parser.add_argument("--component-merkle-security-bits", type=int)
    parser.add_argument("--constraint-work", type=int, help="Circuit constraint count for full-prover scoring")
    parser.add_argument("--case-label", help="Optional case label copied into report rows")
    parser.add_argument(
        "--workload-r1cs",
        help=(
            "R1CS file used to bind a directly measured workload; requires "
            "--constraint-work and --case-label"
        ),
    )
    parser.add_argument("--cargo", default="cargo", help="Cargo binary used when --num-variables is set")
    parser.add_argument(
        "--features",
        default="parallel",
        help="Cargo features used to build generated candidates; default: parallel",
    )
    parser.add_argument(
        "--proof-mode",
        choices=("no-zk", "full-zk"),
        default="no-zk",
        help="Proof mode used for candidate validation and cost estimates",
    )
    parser.add_argument(
        "--zk-ell-values",
        help=(
            "Comma-separated ell_zk values to sweep in full-zk mode; "
            f"default: {','.join(map(str, DEFAULT_ZK_ELL_SWEEP))}"
        ),
    )
    parser.add_argument(
        "--zk-mask-log-inv-rate-values",
        help=(
            "Comma-separated mask_log_inv_rate values to sweep in full-zk mode; "
            f"default: {','.join(map(str, DEFAULT_ZK_MASK_LOG_INV_RATE_SWEEP))}"
        ),
    )
    parser.add_argument(
        "--max-report-rows",
        type=int,
        help="Limit stored score rows after ranking; selection still uses all rows",
    )
    parser.add_argument(
        "--measurement-shortlist-margin-seconds",
        type=float,
        help="Include accepted rows within this many projected seconds of the model best",
    )
    parser.add_argument(
        "--measurement-shortlist-margin-ratio",
        type=float,
        help=(
            "Include accepted rows within this fraction of the model best projected time; "
            f"default derives from heldout residuals or falls back to {DEFAULT_SHORTLIST_MARGIN_RATIO:g}"
        ),
    )
    parser.add_argument(
        "--measurements",
        help="Heldout JSON from poseidon-schedule-heldout; adds selected_measured while keeping selected model-based",
    )
    parser.add_argument(
        "--reference-label",
        action="append",
        default=[],
        help="Accepted candidate label to include in the measurement shortlist; repeatable",
    )
    args = parser.parse_args()

    if bool(args.candidates) == bool(args.num_variables):
        raise SystemExit("pass exactly one of --candidates or --num-variables")
    if args.round_log_inv_rate_offset_max < 0:
        raise SystemExit("--round-log-inv-rate-offset-max must be non-negative")

    candidates = (
        read_json(Path(args.candidates))
        if args.candidates
        else generate_candidates(
            args.cargo,
            args.num_variables,
            args.max_pow_bits,
            args.field,
            args.proof_mode,
            args.constraint_work,
            parse_int_list(args.zk_ell_values),
            parse_int_list(args.zk_mask_log_inv_rate_values),
            args.features,
            args.security_bits,
            args.merkle_security_bits,
            args.component_security_bits,
            args.component_merkle_security_bits,
            args.extension,
            args.round_log_inv_rate_offset_max,
        )
    )
    candidates = filter_candidate_extensions(candidates, args.extension)
    workload_identity = build_workload_identity(
        args.workload_r1cs,
        args.constraint_work,
        args.case_label,
    )
    apply_case_metrics(
        candidates,
        args.constraint_work,
        args.case_label,
        workload_identity,
    )
    calibration = read_json(Path(args.calibration))
    measurements = read_json(Path(args.measurements)) if args.measurements else None
    report = score_dump(
        candidates,
        calibration,
        args.max_pow_bits,
        args.max_report_rows,
        args.measurement_shortlist_margin_seconds,
        args.measurement_shortlist_margin_ratio,
        measurements,
        args.reference_label,
    )

    write_json(Path(args.out_report), report)
    selected = report.get("selected")
    if selected is None:
        raise SystemExit("no valid schedule found")
    if args.out_config:
        setup_config = selected_setup_config(report)
        if setup_config is None:
            raise SystemExit(
                "component searches do not produce standalone setup configs; "
                "compose their reports with poseidon_spark_schedule_scorer.py"
            )
        write_json(Path(args.out_config), setup_config)

    trust = "trusted" if report["model_validation"]["trusted"] else "untrusted"
    message = (
        "selected "
        f"label={selected['label']} "
        f"extension={selected['extension']} "
        f"projected_seconds={selected['projected_seconds']:.9g} "
        f"security_bits={selected['security_bits_achieved']:.3f} "
        f"max_pow_bits={selected['max_derived_pow_bits']} "
        f"model={trust}"
    )
    if report.get("selected_measured") is not None:
        measured = report["selected_measured"]
        message += (
            " measured_selected="
            f"{measured['label']} measured_seconds={measured['measured_seconds']:.9g}"
        )
    print(message)


def selected_setup_config(report: dict[str, Any]) -> dict[str, Any] | None:
    selected = report.get("selected_measured") or report.get("selected")
    if selected is None:
        return None
    return selected.get("setup_config")


def score_dump(
    dump: dict[str, Any],
    calibration: dict[str, Any],
    max_pow_bits: int,
    max_report_rows: int | None = None,
    measurement_shortlist_margin_seconds: float | None = None,
    measurement_shortlist_margin_ratio: float | None = None,
    measurements: dict[str, Any] | None = None,
    reference_labels: list[str] | None = None,
) -> dict[str, Any]:
    require_matching_code_provenance(dump, calibration, "candidate dump", "calibration")
    if measurements is not None:
        require_matching_code_provenance(
            dump, measurements, "candidate dump", "heldout measurements"
        )
        require_matching_workload_identity(dump, measurements)
    require_consistent_workload_identity(dump, "candidate dump", "candidates")
    if measurements is not None:
        require_consistent_workload_identity(
            measurements, "heldout measurements", "rows"
        )
    proof_mode = dump.get("proof_mode")
    if proof_mode not in ("no-zk", "full-zk"):
        raise SystemExit("candidate dump must declare proof_mode as no-zk or full-zk")
    coeffs = normalized_coefficients(calibration)
    verifier_coeffs = normalized_verifier_coefficients(calibration, coeffs)
    validation = validate_model(calibration, coeffs, proof_mode)
    require_row_proof_modes(dump.get("candidates", []), proof_mode, "candidate")
    if measurements is not None and measurements.get("proof_mode") != proof_mode:
        raise SystemExit("heldout measurements must use the scorer report's proof_mode")
    if measurements is not None:
        require_row_proof_modes(measurements.get("rows", []), proof_mode, "heldout")
    use_zk_metrics = proof_mode == "full-zk"
    component_search = dump.get("component_security_override_bits") is not None
    scored = []
    for candidate in dump.get("candidates", []):
        row = dict(candidate)
        rejection_reasons = list(filter(None, [candidate.get("rejection_reason")]))
        if not candidate.get("valid"):
            rejection_reasons.append("candidate marked invalid by backend")
        derived_pow = candidate.get("max_derived_pow_bits")
        if derived_pow is None:
            rejection_reasons.append("missing derived PoW")
        elif int(derived_pow) > max_pow_bits:
            rejection_reasons.append(f"derived PoW {derived_pow} exceeds max {max_pow_bits}")
        if candidate.get("setup_config") is None and not component_search:
            rejection_reasons.append("missing setup config")
        projected = projected_seconds(candidate, coeffs, use_zk_metrics)
        row["projected_seconds"] = projected
        row["cost_breakdown"] = cost_breakdown(candidate, coeffs, use_zk_metrics)
        row["verifier_projected_seconds"] = verifier_projected_seconds(
            candidate, verifier_coeffs, use_zk_metrics
        )
        row["verifier_cost_breakdown"] = verifier_cost_breakdown(
            candidate, verifier_coeffs, use_zk_metrics
        )
        row["accepted_for_ranking"] = not rejection_reasons
        row["rejection_reasons"] = rejection_reasons
        scored.append(row)

    accepted = [row for row in scored if row["accepted_for_ranking"]]
    accepted.sort(
        key=lambda row: (
            float(row["projected_seconds"]),
            pow_tie_break_key(row),
            str(row.get("label") or ""),
        )
    )
    selected = accepted[0] if accepted else None
    measurement_shortlist, shortlist_meta = build_measurement_shortlist(
        accepted,
        validation,
        measurement_shortlist_margin_seconds,
        measurement_shortlist_margin_ratio,
    )
    measurement_shortlist, reference_meta = include_reference_rows(
        measurement_shortlist, accepted, reference_labels or []
    )
    shortlist_meta.update(reference_meta)
    measurement_shortlist, coverage_meta = include_schedule_coverage_rows(
        measurement_shortlist, accepted
    )
    shortlist_meta.update(coverage_meta)
    shortlist_meta["required_labels"] = list(
        dict.fromkeys(
            [
                *shortlist_meta.get("reference_labels", []),
                *shortlist_meta.get("coverage_labels", []),
            ]
        )
    )
    if measurements is not None:
        require_measurement_rows_match_candidates(measurements, accepted)
    selected_measured, measurement_summary = measured_selection(measurements)

    sorted_scores = sorted(
        scored,
        key=lambda row: (
            not row["accepted_for_ranking"],
            float(row["projected_seconds"]),
            pow_tie_break_key(row),
            str(row.get("label") or ""),
        ),
    )
    retention = None
    if max_report_rows is not None and component_search:
        frontier = pareto_rows_3d(
            accepted,
            lambda row: row["projected_seconds"],
            lambda row: row["verifier_projected_seconds"],
            lambda row: row.get("proof_size_bytes_estimate") or 0,
        )
        required = unique_rows(
            [
                *frontier,
                *measurement_shortlist,
                *([selected] if selected is not None else []),
            ]
        )
        if len(required) > max_report_rows:
            raise SystemExit(
                "--max-report-rows is too small for the component verifier/proof-size "
                f"frontier and required rows: need at least {len(required)}"
            )
        required_keys = {measurement_shortlist_key(row) for row in required}
        fill = [
            row
            for row in sorted_scores
            if measurement_shortlist_key(row) not in required_keys
        ][: max_report_rows - len(required)]
        sorted_scores = unique_rows([*required, *fill])
        sorted_scores.sort(
            key=lambda row: (
                not row["accepted_for_ranking"],
                float(row["projected_seconds"]),
                pow_tie_break_key(row),
                str(row.get("label") or ""),
            )
        )
        retention = {
            "policy": "complete verifier/proof-size Pareto frontier plus measurement and selected rows",
            "accepted_rows": len(accepted),
            "pareto_rows": len(frontier),
            "required_rows": len(required),
            "stored_rows": len(sorted_scores),
        }
    if max_report_rows is not None:
        sorted_scores = sorted_scores[:max_report_rows]

    return {
        "schema_version": 3,
        "provenance": dump.get("provenance"),
        "source_provenance": {
            "candidates": dump.get("provenance"),
            "calibration": calibration.get("provenance"),
            "measurements": measurements.get("provenance") if measurements else None,
        },
        "source_schema_version": dump.get("schema_version"),
        "num_variables": dump.get("num_variables"),
        "num_outer_rounds": dump.get("num_outer_rounds"),
        "constraint_work": dump.get("constraint_work"),
        "case_label": dump.get("case_label"),
        "workload_identity": dump.get("workload_identity"),
        "target_security_bits": dump.get("target_security_bits"),
        "target_merkle_security_bits": dump.get("target_merkle_security_bits"),
        "component_security_override_bits": dump.get(
            "component_security_override_bits"
        ),
        "component_merkle_security_override_bits": dump.get(
            "component_merkle_security_override_bits"
        ),
        "max_pow_bits": max_pow_bits,
        "extension_filter": dump.get("extension_filter", "all"),
        "round_log_inv_rate_offset_max": dump.get(
            "round_log_inv_rate_offset_max", 0
        ),
        "proof_mode": proof_mode,
        "model_validation": validation,
        "coefficients": coeffs,
        "verifier_coefficients": verifier_coeffs,
        "selected": selected,
        "selected_measured": selected_measured,
        "measurement_summary": measurement_summary,
        "measurement_shortlist": measurement_shortlist,
        "measurement_shortlist_meta": shortlist_meta,
        "candidate_retention": retention,
        "scores": sorted_scores,
    }


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in rows:
        key = measurement_shortlist_key(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def pareto_rows_3d(
    rows: list[dict[str, Any]],
    time_value,
    verifier_value,
    size_value,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(time_value(row)),
            float(verifier_value(row)),
            int(size_value(row)),
            str(row.get("label") or ""),
        ),
    )
    selected = []
    seen_configurations = set()
    verifier_values = sorted({float(verifier_value(row)) for row in ordered})
    verifier_indexes = {value: index + 1 for index, value in enumerate(verifier_values)}
    prefix_minima: list[tuple[int, float, float] | None] = [None] * (
        len(verifier_values) + 1
    )

    def prefix_min(index: int) -> tuple[int, float, float] | None:
        best = None
        while index > 0:
            candidate = prefix_minima[index]
            if candidate is not None and (best is None or candidate < best):
                best = candidate
            index -= index & -index
        return best

    def update(index: int, value: tuple[int, float, float]) -> None:
        while index < len(prefix_minima):
            current = prefix_minima[index]
            if current is None or value < current:
                prefix_minima[index] = value
            index += index & -index

    for row in ordered:
        identity = measurement_shortlist_key(row)
        if identity in seen_configurations:
            continue
        seen_configurations.add(identity)
        objective = (
            float(time_value(row)),
            float(verifier_value(row)),
            int(size_value(row)),
        )
        verifier_index = verifier_indexes[objective[1]]
        prior = prefix_min(verifier_index)
        dominated = prior is not None and prior[0] <= objective[2] and (
            prior[0] < objective[2]
            or prior[1] < objective[0]
            or prior[2] < objective[1]
        )
        if not dominated:
            selected.append(row)
        update(verifier_index, (objective[2], objective[0], objective[1]))
    return selected


def normalized_coefficients(calibration: dict[str, Any]) -> dict[str, Any]:
    source = calibration.get("coefficients")
    if not isinstance(source, dict):
        raise SystemExit("calibration JSON must contain a coefficients object")
    coeffs: dict[str, Any] = {"fixed_overhead": float(source.get("fixed_overhead", 0.0))}
    for name, _metric in COMPONENTS:
        if name == "spartan" and name not in source:
            coeffs[name] = 0.0
            continue
        if name == "merkle_path" and name not in source:
            coeffs[name] = 0.0
            continue
        if name not in source:
            raise SystemExit(f"calibration missing coefficient {name}")
        if name == "sumcheck" and isinstance(source[name], dict):
            coeffs[name] = {key: float(value) for key, value in source[name].items()}
        else:
            coeffs[name] = float(source[name])
    return coeffs


def normalized_verifier_coefficients(
    calibration: dict[str, Any], prover_coeffs: dict[str, Any]
) -> dict[str, Any]:
    source = calibration.get("verifier_coefficients")
    if source is None:
        return {
            "fixed_overhead": 0.0,
            "merkle_hash": float(prover_coeffs.get("merkle_path", 0.0)),
            "leaf_field_element": float(prover_coeffs.get("merkle_path", 0.0)),
            "row_field_element": float(prover_coeffs.get("row_opening", 0.0)),
            "extension_operation": prover_coeffs.get("sumcheck", {}),
            "pow_check": float(prover_coeffs.get("pow", 0.0)),
            "source": "legacy_prover_coefficients",
        }
    if not isinstance(source, dict):
        raise SystemExit("calibration verifier_coefficients must be an object")
    required = (
        "merkle_hash",
        "leaf_field_element",
        "row_field_element",
        "extension_operation",
        "pow_check",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise SystemExit(
            "calibration missing verifier coefficient " + ", ".join(missing)
        )
    extension = source["extension_operation"]
    if not isinstance(extension, dict):
        raise SystemExit("verifier extension_operation coefficient must be an object")
    return {
        "fixed_overhead": float(source.get("fixed_overhead", 0.0)),
        "merkle_hash": float(source["merkle_hash"]),
        "leaf_field_element": float(source["leaf_field_element"]),
        "row_field_element": float(source["row_field_element"]),
        "extension_operation": {
            key: float(value) for key, value in extension.items()
        },
        "pow_check": float(source["pow_check"]),
        "source": "verifier_component_calibration",
    }


def verifier_projected_seconds(
    candidate: dict[str, Any],
    coeffs: dict[str, Any],
    use_zk_metrics: bool = False,
) -> float:
    return sum(
        verifier_component_seconds(candidate, coeffs, name, metric, use_zk_metrics)
        for name, metric in VERIFIER_COMPONENTS
    ) + float(coeffs["fixed_overhead"])


def verifier_cost_breakdown(
    candidate: dict[str, Any],
    coeffs: dict[str, Any],
    use_zk_metrics: bool = False,
) -> dict[str, float]:
    out = {"fixed_overhead": float(coeffs["fixed_overhead"])}
    for name, metric in VERIFIER_COMPONENTS:
        out[name] = verifier_component_seconds(
            candidate, coeffs, name, metric, use_zk_metrics
        )
    return out


def verifier_component_seconds(
    candidate: dict[str, Any],
    coeffs: dict[str, Any],
    name: str,
    metric: str,
    use_zk_metrics: bool,
) -> float:
    metric_name = f"zk_{metric}" if use_zk_metrics else metric
    work = float(candidate.get(metric_name) or candidate.get(metric) or 0.0)
    coefficient = coeffs[name]
    if isinstance(coefficient, dict):
        extension = candidate.get("extension")
        if extension not in coefficient:
            raise SystemExit(
                f"missing verifier {name} coefficient for extension {extension}"
            )
        coefficient = coefficient[extension]
    return work * float(coefficient)


def projected_seconds(
    candidate: dict[str, Any], coeffs: dict[str, Any], use_zk_metrics: bool = False
) -> float:
    total = float(coeffs["fixed_overhead"])
    for name, metric in COMPONENTS:
        total += component_seconds(candidate, coeffs, name, metric, use_zk_metrics)
    return total


def cost_breakdown(
    candidate: dict[str, Any], coeffs: dict[str, Any], use_zk_metrics: bool = False
) -> dict[str, float]:
    out = {"fixed_overhead": float(coeffs["fixed_overhead"])}
    for name, metric in COMPONENTS:
        out[name] = component_seconds(candidate, coeffs, name, metric, use_zk_metrics)
    return out


def component_seconds(
    candidate: dict[str, Any],
    coeffs: dict[str, Any],
    name: str,
    metric: str,
    use_zk_metrics: bool = False,
) -> float:
    metric_name = zk_metric_name(metric) if use_zk_metrics else metric
    work = float(candidate.get(metric_name) or candidate.get(metric) or 0.0)
    if work == 0.0:
        return 0.0
    return coefficient_for(candidate, coeffs, name) * work


def coefficient_for(candidate: dict[str, Any], coeffs: dict[str, Any], name: str) -> float:
    coeff = coeffs[name]
    if isinstance(coeff, dict):
        extension = candidate.get("extension")
        if extension not in coeff:
            raise SystemExit(f"missing {name} coefficient for extension {extension}")
        return float(coeff[extension])
    return float(coeff)


def zk_metric_name(metric: str) -> str:
    return {
        "dft_work": "zk_dft_work",
        "merkle_work": "zk_merkle_work",
        "merkle_path_work": "zk_merkle_path_work",
        "row_work": "zk_row_work",
        "sumcheck_work": "zk_sumcheck_work",
    }.get(metric, metric)


def pow_tie_break_key(row: dict[str, Any]) -> tuple[int, int]:
    return (
        int(row.get("max_derived_pow_bits") or 0),
        int(row.get("pow_work_units") or 0),
    )


def build_measurement_shortlist(
    accepted: list[dict[str, Any]],
    validation: dict[str, Any],
    margin_seconds: float | None,
    margin_ratio: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not accepted:
        return [], {
            "margin_seconds": 0.0,
            "margin_source": "empty",
            "pre_dedup_count": 0,
            "dedup_count": 0,
        }
    if margin_seconds is not None and margin_seconds < 0:
        raise SystemExit("--measurement-shortlist-margin-seconds must be non-negative")
    if margin_ratio is not None and margin_ratio < 0:
        raise SystemExit("--measurement-shortlist-margin-ratio must be non-negative")

    best = float(accepted[0]["projected_seconds"])
    if margin_seconds is not None:
        margin = margin_seconds
        source = "cli_seconds"
    elif margin_ratio is not None:
        margin = best * margin_ratio
        source = "cli_ratio"
    else:
        diagnostic = validation.get("ordering_diagnostic") or {}
        derived = diagnostic.get("model_resolution_seconds")
        if derived is not None and float(derived) > 0.0:
            margin = float(derived)
            source = "heldout_residuals"
        else:
            margin = best * DEFAULT_SHORTLIST_MARGIN_RATIO
            source = "default_ratio"

    cutoff = best + margin
    candidates = [
        row
        for row in accepted
        if float(row["projected_seconds"]) <= cutoff
    ]
    deduped = dedup_measurement_shortlist(candidates)
    return deduped, {
        "best_projected_seconds": best,
        "cutoff_projected_seconds": cutoff,
        "margin_seconds": margin,
        "margin_source": source,
        "pre_dedup_count": len(candidates),
        "dedup_count": len(deduped),
        "deduplication": "collapse ell_zk variants by schedule and mask_log_inv_rate",
    }


def dedup_measurement_shortlist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = measurement_shortlist_key(row)
        old = selected.get(key)
        if old is None or shortlist_preferred(row) < shortlist_preferred(old):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            float(row["projected_seconds"]),
            pow_tie_break_key(row),
            int(row.get("zk_ell") or 0),
            str(row.get("label") or ""),
        ),
    )


def include_reference_rows(
    shortlist: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    reference_labels: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = list(dict.fromkeys(reference_labels))
    reference_rows = []
    for label in requested:
        matches = [row for row in accepted if row.get("label") == label]
        if not matches:
            raise SystemExit(
                f"reference label not found among accepted candidates: {label}"
            )
        reference_rows.extend(matches)
    combined = dedup_measurement_shortlist([*shortlist, *reference_rows])
    existing_keys = {measurement_shortlist_key(row) for row in shortlist}
    added = sum(
        measurement_shortlist_key(row) not in existing_keys
        for row in combined
    )
    return combined, {
        "reference_labels": requested,
        "reference_rows_added": added,
        "dedup_count": len(combined),
    }


def include_schedule_coverage_rows(
    shortlist: list[dict[str, Any]], accepted: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in accepted:
        group = (
            str(row.get("extension") or "unknown"),
            folding_schedule_family(row),
            int(row.get("round_log_inv_rate_offset") or 0),
        )
        current = by_group.get(group)
        if current is None or shortlist_preferred(row) < shortlist_preferred(current):
            by_group[group] = row
    coverage_rows = sorted(by_group.values(), key=shortlist_preferred)
    combined = dedup_measurement_shortlist([*shortlist, *coverage_rows])
    existing_keys = {measurement_shortlist_key(row) for row in shortlist}
    added = sum(
        measurement_shortlist_key(row) not in existing_keys for row in combined
    )
    coverage_labels = list(
        dict.fromkeys(str(row.get("label") or "") for row in coverage_rows)
    )
    return combined, {
        "coverage_grouping": [
            "extension",
            "folding_schedule_family",
            "round_log_inv_rate_offset",
        ],
        "coverage_group_count": len(by_group),
        "coverage_labels": coverage_labels,
        "coverage_rows_added": added,
        "dedup_count": len(combined),
    }


def folding_schedule_family(row: dict[str, Any]) -> str:
    params = row.get("whir_params")
    if not isinstance(params, dict):
        setup = row.get("setup_config") or {}
        params = setup.get("whir_params") if isinstance(setup, dict) else None
    if not isinstance(params, dict):
        return "unknown"
    schedule = params.get("folding_schedule")
    if schedule is None or (isinstance(schedule, dict) and "Constant" in schedule):
        return "constant"
    if isinstance(schedule, dict) and "ConstantFromSecondRound" in schedule:
        return "constant_from_second_round"
    if isinstance(schedule, dict) and "PerRound" in schedule:
        return "per_round"
    return "unknown"


def shortlist_preferred(row: dict[str, Any]) -> tuple[float, tuple[int, int], int, str]:
    return (
        float(row.get("projected_seconds") or 0.0),
        pow_tie_break_key(row),
        int(row.get("zk_ell") or 0),
        str(row.get("label") or ""),
    )


def measurement_shortlist_key(row: dict[str, Any]) -> str:
    setup = json.loads(json.dumps(row.get("setup_config") or {}, sort_keys=True))
    setup.pop("ell_zk", None)
    key = {
        "label": row.get("label"),
        "extension": row.get("extension"),
        "zk_mask_log_inv_rate": row.get("zk_mask_log_inv_rate"),
        "setup_without_ell_zk": setup,
    }
    return json.dumps(key, sort_keys=True)


def measured_selection(
    measurements: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if measurements is None:
        return None, None
    rows = [
        row
        for row in measurements.get("rows", [])
        if float(row.get("measured_seconds") or 0.0) > 0.0
    ]
    if not rows:
        return None, {
            "source_measurements": measurements.get("source_report"),
            "measured_rows": 0,
        }
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["measured_seconds"]),
            pow_tie_break_key(row),
            str(row.get("label") or ""),
        ),
    )
    fastest = ranked[0]
    ties = measured_ties_with_fastest(ranked)
    tied_rows = [
        row for row in ranked if not demonstrably_slower_than_fastest(row, fastest)
    ]
    selected_source = min(
        tied_rows,
        key=lambda row: (
            pow_tie_break_key(row),
            str(row.get("label") or ""),
        ),
    )
    selected = dict(selected_source)
    selected["measured_rank"] = ranked.index(selected_source) + 1
    return selected, {
        "source_measurements": measurements.get("source_report"),
        "measured_rows": len(rows),
        "selection": "one_percent_or_overlapping_median_ci_then_pow_then_label",
        "demonstrable_speed_threshold_relative": TIME_TIE_RELATIVE,
        "tied_with_best_count": len(ties),
        "tied_with_best": ties,
    }


def measured_ties_with_fastest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    fastest = rows[0]
    tied = []
    for rank, row in enumerate(rows, 1):
        if demonstrably_slower_than_fastest(row, fastest):
            continue
        tied.append(
            {
                "measured_rank": rank,
                "label": row.get("label"),
                "zk_ell": row.get("zk_ell"),
                "zk_mask_log_inv_rate": row.get("zk_mask_log_inv_rate"),
                "measured_seconds": row.get("measured_seconds"),
                "heldout_median_ci_seconds": row.get("heldout_median_ci_seconds"),
                "heldout_relative_median_difference": row.get(
                    "heldout_relative_median_difference"
                ),
                "heldout_paired_relative_median_ci": row.get(
                    "heldout_paired_relative_median_ci"
                ),
                "heldout_proof_size_median_bytes": row.get(
                    "heldout_proof_size_median_bytes"
                ),
                "max_derived_pow_bits": row.get("max_derived_pow_bits"),
                "pow_work_units": row.get("pow_work_units"),
            }
        )
    return tied


def demonstrably_slower_than_fastest(
    row: dict[str, Any], fastest: dict[str, Any]
) -> bool:
    fastest_seconds = float(fastest.get("measured_seconds") or 0.0)
    row_seconds = float(row.get("measured_seconds") or 0.0)
    relative = (
        (row_seconds - fastest_seconds) / fastest_seconds
        if fastest_seconds > 0.0
        else float("inf")
    )
    if float(relative) <= TIME_TIE_RELATIVE + TIME_TIE_EPSILON:
        return False
    row_ci = median_ci(row)
    fastest_ci = median_ci(fastest)
    return not (
        row_ci is not None
        and fastest_ci is not None
        and intervals_overlap(row_ci, fastest_ci)
    )


def median_ci(row: dict[str, Any]) -> tuple[float, float] | None:
    raw = row.get("heldout_median_ci_seconds")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return float(raw[0]), float(raw[1])


def intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def validate_model(
    calibration: dict[str, Any], coeffs: dict[str, Any], proof_mode: str
) -> dict[str, Any]:
    validation = calibration.get("validation") or {}
    tolerance = float(validation.get("max_relative_error", DEFAULT_VALIDATION_TOLERANCE))
    source_heldout = validation.get("heldout") or []
    if not isinstance(source_heldout, list):
        raise SystemExit("calibration validation heldout rows must be an array")
    heldout = []
    for index, row in enumerate(source_heldout):
        if not isinstance(row, dict) or row.get("proof_mode") not in ("no-zk", "full-zk"):
            raise SystemExit(
                f"calibration validation row {index} must declare proof_mode as no-zk or full-zk"
            )
        if row["proof_mode"] == proof_mode:
            heldout.append(row)
    rows = []
    trusted = bool(heldout)
    for row in heldout:
        measured = float(row.get("measured_seconds") or 0.0)
        projected = projected_seconds(row, coeffs, row_uses_zk_metrics(row))
        rel_error = abs(projected - measured) / measured if measured > 0 else float("inf")
        ok = rel_error <= tolerance
        trusted = trusted and ok
        rows.append(
            {
                "label": row.get("label"),
                "measured_seconds": measured,
                "projected_seconds": projected,
                "relative_error": rel_error,
                "ok": ok,
            }
        )
    ordering = ordering_diagnostic(rows)
    return {
        "trusted": trusted,
        "proof_mode": proof_mode,
        "source_heldout_rows": len(source_heldout),
        "mode_heldout_rows": len(heldout),
        "max_relative_error": tolerance,
        "heldout": rows,
        "ordering_diagnostic": ordering,
    }


def ordering_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if float(row.get("measured_seconds") or 0.0) > 0.0
        and float(row.get("projected_seconds") or 0.0) > 0.0
    ]
    if len(usable) < 2:
        resolution = (
            abs(float(usable[0]["projected_seconds"]) - float(usable[0]["measured_seconds"]))
            if usable
            else None
        )
        measured = float(usable[0]["measured_seconds"]) if usable else None
        return {
            "comparable_pairs": 0,
            "model_resolution_seconds": resolution,
            "model_resolution_relative": (resolution / measured if resolution and measured else None),
        }

    projected_order = {
        id(row): rank
        for rank, row in enumerate(
            sorted(usable, key=lambda row: float(row["projected_seconds"])),
            1,
        )
    }
    measured_order = {
        id(row): rank
        for rank, row in enumerate(
            sorted(usable, key=lambda row: float(row["measured_seconds"])),
            1,
        )
    }
    concordant = 0
    discordant = 0
    for left_index in range(len(usable)):
        for right_index in range(left_index + 1, len(usable)):
            left = usable[left_index]
            right = usable[right_index]
            projected_delta = projected_order[id(left)] - projected_order[id(right)]
            measured_delta = measured_order[id(left)] - measured_order[id(right)]
            product = projected_delta * measured_delta
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    comparable = concordant + discordant
    tau = (concordant - discordant) / comparable if comparable else None
    residuals = [
        abs(float(row["projected_seconds"]) - float(row["measured_seconds"]))
        for row in usable
    ]
    measured_values = sorted(float(row["measured_seconds"]) for row in usable)
    median_measured = measured_values[len(measured_values) // 2]
    resolution = max(residuals)
    return {
        "comparable_pairs": comparable,
        "kendall_tau": tau,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "projected_span_seconds": max(float(row["projected_seconds"]) for row in usable)
        - min(float(row["projected_seconds"]) for row in usable),
        "measured_span_seconds": max(float(row["measured_seconds"]) for row in usable)
        - min(float(row["measured_seconds"]) for row in usable),
        "model_resolution_seconds": resolution,
        "model_resolution_relative": resolution / median_measured if median_measured > 0 else None,
    }


def row_uses_zk_metrics(row: dict[str, Any]) -> bool:
    return row.get("proof_mode") == "full-zk"


def generate_candidates(
    cargo: str,
    num_variables: int,
    max_pow_bits: int,
    field: str,
    proof_mode: str,
    constraint_work: int | None,
    zk_ell_values: list[int] | None,
    zk_mask_log_inv_rate_values: list[int] | None,
    features: str = "parallel",
    security_bits: int = DEFAULT_SECURITY_BITS,
    merkle_security_bits: int | None = None,
    component_security_bits: int | None = None,
    component_merkle_security_bits: int | None = None,
    extension: str = "all",
    round_log_inv_rate_offset_max: int = 0,
) -> dict[str, Any]:
    num_outer_rounds = (
        max(0, constraint_work - 1).bit_length()
        if constraint_work is not None
        else num_variables
    )
    if proof_mode == "no-zk":
        return generate_candidate_dump(
            cargo,
            num_variables,
            max_pow_bits,
            field,
            proof_mode,
            num_outer_rounds,
            None,
            None,
            features,
            security_bits,
            merkle_security_bits,
            component_security_bits,
            component_merkle_security_bits,
            extension,
            round_log_inv_rate_offset_max,
        )
    ell_values = zk_ell_values or DEFAULT_ZK_ELL_SWEEP
    mask_rate_values = zk_mask_log_inv_rate_values or DEFAULT_ZK_MASK_LOG_INV_RATE_SWEEP
    dumps = []
    for zk_ell in ell_values:
        for zk_mask_log_inv_rate in mask_rate_values:
            dumps.append(
                generate_candidate_dump(
                    cargo,
                    num_variables,
                    max_pow_bits,
                    field,
                    proof_mode,
                    num_outer_rounds,
                    zk_ell,
                    zk_mask_log_inv_rate,
                    features,
                    security_bits,
                    merkle_security_bits,
                    component_security_bits,
                    component_merkle_security_bits,
                    extension,
                    round_log_inv_rate_offset_max,
                )
            )
    merged = dict(dumps[0])
    merged["candidates"] = []
    merged["zk_ell_sweep"] = [value for value in ell_values if value is not None]
    merged["zk_mask_log_inv_rate_sweep"] = [
        value for value in mask_rate_values if value is not None
    ]
    for dump in dumps:
        require_matching_code_provenance(
            merged, dump, "first candidate dump", "candidate dump"
        )
        for candidate in dump.get("candidates", []):
            row = dict(candidate)
            row["zk_ell"] = dump.get("zk_ell")
            row["zk_mask_log_inv_rate"] = dump.get("zk_mask_log_inv_rate")
            merged["candidates"].append(row)
    return merged


def generate_candidate_dump(
    cargo: str,
    num_variables: int,
    max_pow_bits: int,
    field: str,
    proof_mode: str,
    num_outer_rounds: int,
    zk_ell: int | None,
    zk_mask_log_inv_rate: int | None,
    features: str,
    security_bits: int,
    merkle_security_bits: int | None,
    component_security_bits: int | None,
    component_merkle_security_bits: int | None,
    extension: str = "all",
    round_log_inv_rate_offset_max: int = 0,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        cargo,
        "run",
        "-q",
        "--release",
        "--manifest-path",
        str(repo / "Cargo.toml"),
        "--features",
        features,
        "--bin",
        "poseidon-schedule-candidates",
        "--",
        "--num-variables",
        str(num_variables),
        "--field",
        field,
        "--extension",
        extension,
        "--max-pow-bits",
        str(max_pow_bits),
        "--round-log-inv-rate-offset-max",
        str(round_log_inv_rate_offset_max),
        "--security-bits",
        str(security_bits),
        "--merkle-security-bits",
        str(merkle_security_bits if merkle_security_bits is not None else security_bits),
        "--proof-mode",
        proof_mode,
        "--num-outer-rounds",
        str(num_outer_rounds),
    ]
    if (component_security_bits is None) != (component_merkle_security_bits is None):
        raise SystemExit(
            "component security and Merkle security targets must be supplied together"
        )
    if component_security_bits is not None:
        cmd.extend(
            [
                "--component-security-bits",
                str(component_security_bits),
                "--component-merkle-security-bits",
                str(component_merkle_security_bits),
            ]
        )
    if zk_ell is not None:
        cmd.extend(["--zk-ell", str(zk_ell)])
    if zk_mask_log_inv_rate is not None:
        cmd.extend(["--zk-mask-log-inv-rate", str(zk_mask_log_inv_rate)])
    output = subprocess.check_output(cmd, text=True)
    return json.loads(output)


def filter_candidate_extensions(
    dump: dict[str, Any], extension: str
) -> dict[str, Any]:
    if extension == "all":
        return dump
    filtered = dict(dump)
    filtered["extension_filter"] = extension
    filtered["candidates"] = [
        row for row in dump.get("candidates", []) if row.get("extension") == extension
    ]
    return filtered


def parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("integer list must not be empty")
    return values


def apply_case_metrics(
    dump: dict[str, Any],
    constraint_work: int | None,
    case_label: str | None,
    workload_identity: dict[str, Any] | None = None,
) -> None:
    if constraint_work is not None:
        dump["constraint_work"] = constraint_work
    if case_label is not None:
        dump["case_label"] = case_label
    if workload_identity is not None:
        dump["workload_identity"] = workload_identity
    for candidate in dump.get("candidates", []):
        if constraint_work is not None:
            candidate["constraint_work"] = constraint_work
        if case_label is not None:
            candidate["case_label"] = case_label
        if workload_identity is not None:
            candidate["workload_identity"] = workload_identity


def build_workload_identity(
    r1cs_path: str | None,
    constraint_work: int | None,
    case_label: str | None,
) -> dict[str, Any] | None:
    if r1cs_path is None:
        return None
    if not isinstance(constraint_work, int) or constraint_work <= 0:
        raise SystemExit("--workload-r1cs requires a positive --constraint-work")
    if not isinstance(case_label, str) or not case_label:
        raise SystemExit("--workload-r1cs requires --case-label")
    digest = hashlib.sha256()
    with Path(r1cs_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "label": case_label,
        "r1cs_sha256": digest.hexdigest(),
        "constraint_work": constraint_work,
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def require_matching_code_provenance(
    left: dict[str, Any],
    right: dict[str, Any],
    left_name: str,
    right_name: str,
) -> None:
    left_provenance = left.get("provenance")
    right_provenance = right.get("provenance")
    if left_provenance is None and right_provenance is None:
        raise SystemExit(
            f"cannot combine {left_name} and {right_name}: both artifacts are missing provenance"
        )
    if left_provenance is None or right_provenance is None:
        raise SystemExit(
            f"cannot combine {left_name} and {right_name}: one artifact is missing provenance"
        )
    if left_provenance != right_provenance:
        raise SystemExit(
            f"cannot combine {left_name} and {right_name}: provenance differs"
        )


def require_matching_workload_identity(
    left: dict[str, Any], right: dict[str, Any]
) -> None:
    left_identity = left.get("workload_identity")
    right_identity = right.get("workload_identity")
    if left_identity is None and right_identity is None:
        raise SystemExit(
            "candidate dump and heldout measurements are missing workload_identity"
        )
    if left_identity is None or right_identity is None:
        raise SystemExit("candidate dump and heldout measurements must both carry workload_identity")
    if left_identity != right_identity:
        raise SystemExit("candidate dump and heldout measurement workload_identity differ")


def require_consistent_workload_identity(
    artifact: dict[str, Any], artifact_name: str, rows_field: str
) -> None:
    identity = artifact.get("workload_identity")
    rows = artifact.get(rows_field, [])
    if not isinstance(rows, list):
        raise SystemExit(f"{artifact_name} {rows_field} must be an array")
    row_identities = [
        row.get("workload_identity") if isinstance(row, dict) else None for row in rows
    ]
    if identity is None:
        if any(row_identity is not None for row_identity in row_identities):
            raise SystemExit(
                f"{artifact_name} rows carry workload_identity but the artifact does not"
            )
        return
    if not isinstance(identity, dict) or set(identity) != {
        "label",
        "r1cs_sha256",
        "constraint_work",
    }:
        raise SystemExit(f"{artifact_name} has an invalid workload_identity")
    label = identity.get("label")
    if not isinstance(label, str) or not label:
        raise SystemExit(f"{artifact_name} workload_identity has an invalid label")
    constraint_work = identity.get("constraint_work")
    if (
        not isinstance(constraint_work, int)
        or isinstance(constraint_work, bool)
        or constraint_work <= 0
    ):
        raise SystemExit(
            f"{artifact_name} workload_identity has an invalid constraint_work"
        )
    if "case_label" in artifact and label != artifact.get("case_label"):
        raise SystemExit(f"{artifact_name} workload label differs from case_label")
    if (
        "constraint_work" in artifact
        and constraint_work != artifact.get("constraint_work")
    ):
        raise SystemExit(
            f"{artifact_name} workload constraint count differs from constraint_work"
        )
    digest = identity.get("r1cs_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SystemExit(f"{artifact_name} workload_identity has an invalid r1cs_sha256")
    for index, row_identity in enumerate(row_identities):
        if row_identity != identity:
            raise SystemExit(
                f"{artifact_name} row {index} workload_identity differs from the artifact"
            )


def require_row_proof_modes(rows: Any, proof_mode: str, source: str) -> None:
    if not isinstance(rows, list):
        raise SystemExit(f"{source} rows must be an array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("proof_mode") != proof_mode:
            raise SystemExit(
                f"{source} row {index} must use proof_mode={proof_mode}"
            )


def require_measurement_rows_match_candidates(
    measurements: dict[str, Any], accepted: list[dict[str, Any]]
) -> None:
    accepted_identities = {schedule_row_identity(row) for row in accepted}
    for index, row in enumerate(measurements.get("rows", [])):
        if schedule_row_identity(row) not in accepted_identities:
            raise SystemExit(
                f"heldout row {index} does not match an accepted candidate setup"
            )


def schedule_row_identity(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "label": row.get("label"),
            "extension": row.get("extension"),
            "proof_mode": row.get("proof_mode"),
            "setup_config": row.get("setup_config"),
            "workload_identity": row.get("workload_identity"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
