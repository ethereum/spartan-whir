#!/usr/bin/env python3
"""Compose and calibrate SPARK WHIR schedule candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FIXED_VALUE_COLUMN_BITS = 3
FIXED_AUDIT_COLUMN_BITS = 1
SPARK_COMPONENT_TARGETS = {(116, 116): (120, 123)}
PROVER_SELECTION_BAND_RELATIVE = 0.01
VERIFIER_SELECTION_BAND_RELATIVE = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--witness-report", required=True)
    parser.add_argument("--fixed-value-report", required=True)
    parser.add_argument("--fixed-audit-report", required=True)
    parser.add_argument("--read-report", action="append", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--extension", choices=("quintic", "octic"), required=True)
    parser.add_argument(
        "--proof-mode",
        choices=("no-zk", "full-zk"),
        default="full-zk",
    )
    parser.add_argument("--security-bits", type=int, required=True)
    parser.add_argument("--merkle-security-bits", type=int, required=True)
    parser.add_argument("--ell-zk", type=int)
    parser.add_argument("--mask-log-inv-rate", type=int)
    parser.add_argument("--top-per-component", type=int, default=8)
    parser.add_argument("--max-report-rows", type=int, default=100)
    parser.add_argument("--measurement-rows", type=int, default=10)
    parser.add_argument("--max-fixed-value-log-domain", type=int, required=True)
    parser.add_argument("--max-fixed-audit-log-domain", type=int, required=True)
    parser.add_argument("--fixed-audit-embedded", action="store_true")
    parser.add_argument("--workload-r1cs", required=True)
    parser.add_argument("--workload-label", required=True)
    parser.add_argument("--measurements")
    parser.add_argument(
        "--verifier-measurements",
        help=(
            "Heldout report used for verifier calibration and tie-breaking; "
            "run it with RAYON_NUM_THREADS=1"
        ),
    )
    parser.add_argument("--reference-witness-label")
    parser.add_argument("--reference-fixed-value-label")
    parser.add_argument("--reference-fixed-audit-label")
    parser.add_argument("--reference-read-label", action="append")
    args = parser.parse_args()

    reports = {
        "witness": read_json(args.witness_report),
        "fixed_value": read_json(args.fixed_value_report),
        "fixed_audit": read_json(args.fixed_audit_report),
        "read": [read_json(path) for path in args.read_report],
    }
    if args.proof_mode == "full-zk" and (
        args.ell_zk is None or args.mask_log_inv_rate is None
    ):
        raise SystemExit("full-zk composition requires --ell-zk and --mask-log-inv-rate")
    report = compose_report(
        reports,
        args.security_bits,
        args.merkle_security_bits,
        args.ell_zk,
        args.mask_log_inv_rate,
        args.top_per_component,
        args.max_report_rows,
        args.measurement_rows,
        read_json(args.measurements) if args.measurements else None,
        {
            "fixed_value": args.max_fixed_value_log_domain,
            "fixed_audit": args.max_fixed_audit_log_domain,
        },
        args.extension,
        {
            "witness": args.reference_witness_label,
            "fixed_value": args.reference_fixed_value_label,
            "fixed_audit": args.reference_fixed_audit_label,
            "read": args.reference_read_label,
        },
        proof_mode=args.proof_mode,
        fixed_audit_embedded=args.fixed_audit_embedded,
        workload_identity=workload_identity_from_r1cs(
            Path(args.workload_r1cs), args.workload_label
        ),
        verifier_measurements=(
            read_json(args.verifier_measurements)
            if args.verifier_measurements
            else None
        ),
    )
    write_json(args.out_report, report)
    selected = report.get("selected_measured") or report["selected"]
    print(
        "selected "
        f"label={selected['label']} "
        f"projected_seconds={selected['projected_seconds']:.9g} "
        f"verifier_projected_seconds={selected.get('verifier_projected_seconds', 0.0):.9g} "
        f"proof_size_bytes_estimate={selected['proof_size_bytes_estimate']}"
    )


def compose_report(
    reports: dict[str, Any],
    security_bits: int,
    merkle_security_bits: int,
    ell_zk: int | None,
    mask_log_inv_rate: int | None,
    top_per_component: int,
    max_report_rows: int,
    measurement_rows: int,
    measurements: dict[str, Any] | None,
    fixed_setup_log_domain_caps: dict[str, int],
    extension: str,
    reference_labels: dict[str, Any] | None = None,
    proof_mode: str = "full-zk",
    fixed_audit_embedded: bool = False,
    workload_identity: dict[str, Any] | None = None,
    verifier_measurements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if min(top_per_component, max_report_rows, measurement_rows) <= 0:
        raise SystemExit("row limits must be positive")
    if extension not in ("quintic", "octic"):
        raise SystemExit(f"unsupported extension: {extension}")
    reports = normalize_reports(reports)
    read_reports = reports["read"]
    provenance = require_matching_provenance(reports)
    require_matching_coefficients(reports)
    if proof_mode not in ("no-zk", "full-zk"):
        raise SystemExit(f"unsupported SPARK proof mode: {proof_mode}")
    witness_zk_parameters = None
    if proof_mode == "full-zk":
        if ell_zk is None or mask_log_inv_rate is None:
            raise SystemExit("full-zk composition requires ZK mask parameters")
        witness_zk_parameters = (ell_zk, mask_log_inv_rate)
    require_report_mode(reports["witness"], proof_mode, "witness")
    for component in ("fixed_value", "fixed_audit"):
        require_report_mode(reports[component], "no-zk", component)
    for index, report in enumerate(read_reports):
        require_report_mode(report, "no-zk", f"read[{index}]")
    require_component_dimensions(reports, extension, fixed_audit_embedded)
    component_security_bits, component_merkle_security_bits = require_security_targets(
        reports, security_bits, merkle_security_bits
    )
    workload_identity = require_witness_workload_identity(
        reports["witness"], workload_identity
    )
    if measurements is not None:
        require_matching_measurement_context(
            measurements, provenance, proof_mode, workload_identity
        )
    if verifier_measurements is not None:
        require_matching_measurement_context(
            verifier_measurements, provenance, proof_mode, workload_identity
        )
        if verifier_measurements.get("rayon_threads") != 1:
            raise SystemExit(
                "verifier measurements must record rayon_threads=1"
            )
    reference_labels = normalize_reference_labels(
        reference_labels or {}, len(read_reports)
    )

    all_candidates = {
        component: component_rows(
            reports[component],
            component,
            fixed_setup_log_domain_caps.get(component),
            extension,
            fixed_audit_embedded,
            witness_zk_parameters if component == "witness" else None,
            proof_mode if component == "witness" else "no-zk",
            component_security_bits,
            component_merkle_security_bits,
            workload_identity if component == "witness" else None,
        )
        for component in ("witness", "fixed_value", "fixed_audit")
    }
    all_candidates["read"] = shared_read_rows(
        read_reports,
        extension,
        component_security_bits,
        component_merkle_security_bits,
    )
    band_candidates, prover_band = prover_selection_band(
        all_candidates, fixed_audit_embedded
    )
    verifier_objective_available = all(
        isinstance(report.get("verifier_coefficients"), dict)
        for _, report in named_reports(reports)
    )
    component_frontiers = {
        component: component_pareto_rows(
            candidates, component, fixed_audit_embedded
        )
        for component, candidates in all_candidates.items()
    }
    if fixed_audit_embedded:
        component_frontiers["fixed_audit"] = [
            embedded_fixed_audit_candidate(
                reports["fixed_audit"],
                all_candidates["fixed_audit"],
                reference_labels.get("fixed_audit"),
            )
        ]
    ranked = {
        component: candidates[:top_per_component]
        for component, candidates in component_frontiers.items()
    }
    rows = composed_pareto_rows(
        component_frontiers,
        security_bits,
        merkle_security_bits,
        ell_zk,
        mask_log_inv_rate,
        proof_mode,
        extension,
        fixed_audit_embedded,
        None,
        PROVER_SELECTION_BAND_RELATIVE if verifier_objective_available else None,
    )
    if not rows:
        raise SystemExit(f"no valid {extension} schedule combinations")

    shortlist = stratified_shortlist(
        ranked,
        rows,
        security_bits,
        merkle_security_bits,
        ell_zk,
        mask_log_inv_rate,
        measurement_rows,
        all_candidates,
        reference_labels,
        proof_mode,
        extension,
        fixed_audit_embedded,
    )
    attach_workload_identity(rows, workload_identity)
    attach_workload_identity(shortlist, workload_identity)
    measurement_candidates = [*rows, *shortlist]
    measured_candidate_rows = []
    if measurements is not None:
        measured_candidate_rows = require_measurement_rows_match_candidates(
            measurements, measurement_candidates, workload_identity
        )
    verifier_candidate_rows = []
    if verifier_measurements is not None:
        verifier_candidate_rows = require_measurement_rows_match_candidates(
            verifier_measurements, measurement_candidates, workload_identity
        )

    calibration = calibrate(
        measurement_candidates,
        measurements,
        reference_labels,
        fixed_audit_embedded=fixed_audit_embedded,
    )
    if calibration is not None:
        rows = composed_pareto_rows(
            component_frontiers,
            security_bits,
            merkle_security_bits,
            ell_zk,
            mask_log_inv_rate,
            proof_mode,
            extension,
            fixed_audit_embedded,
            calibration,
            PROVER_SELECTION_BAND_RELATIVE if verifier_objective_available else None,
        )
        attach_workload_identity(rows, workload_identity)
        apply_calibration(shortlist, calibration, fixed_audit_embedded)
        apply_calibration(measured_candidate_rows, calibration, fixed_audit_embedded)
        apply_calibration(verifier_candidate_rows, calibration, fixed_audit_embedded)
    verifier_calibration = calibrate_verifier(
        measurement_candidates,
        verifier_measurements or measurements,
    )
    if verifier_calibration is not None:
        apply_verifier_calibration(rows, verifier_calibration)
        apply_verifier_calibration(shortlist, verifier_calibration)
        apply_verifier_calibration(measured_candidate_rows, verifier_calibration)
        apply_verifier_calibration(verifier_candidate_rows, verifier_calibration)
    composed_frontier_count = len(rows)
    rows = required_report_rows(
        rows, shortlist, [*measured_candidate_rows, *verifier_candidate_rows]
    )
    if len(rows) > max_report_rows:
        raise SystemExit(
            f"{len(rows)} Pareto, shortlist, and measured rows exceed "
            f"--max-report-rows {max_report_rows}"
        )
    rows.sort(key=ranking_key)

    selected_measured, measurement_summary = measured_selection(
        rows, measurements, verifier_measurements
    )
    return {
        "schema_version": 4,
        "provenance": provenance,
        "workload_identity": workload_identity,
        "matrix_closing": "Spark",
        "extension": extension,
        "proof_mode": proof_mode,
        "security_bits": security_bits,
        "merkle_security_bits": merkle_security_bits,
        "component_security_bits": component_security_bits,
        "component_merkle_security_bits": component_merkle_security_bits,
        "ell_zk": ell_zk,
        "mask_log_inv_rate": mask_log_inv_rate,
        "component_num_variables": {
            "witness": reports["witness"].get("num_variables"),
            "fixed_value": reports["fixed_value"].get("num_variables"),
            "fixed_audit": reports["fixed_audit"].get("num_variables"),
            "read": [report.get("num_variables") for report in read_reports],
        },
        "model": {
            "formula": "witness + fixed_value + fixed_audit + sum(read reports)",
            "fixed_value_excludes_setup_commitment": True,
            "fixed_audit_excludes_setup_commitment": True,
            "fixed_audit_embedded": fixed_audit_embedded,
            "read_report_count": len(read_reports),
            "fixed_setup_log_domain_caps": fixed_setup_log_domain_caps,
            "calibration": calibration,
            "verifier_calibration": verifier_calibration,
        },
        "candidate_retention": candidate_retention_metadata(
            all_candidates,
            band_candidates,
            component_frontiers,
            composed_frontier_count,
            len(rows),
            top_per_component,
            fixed_audit_embedded,
            prover_band,
        ),
        "selected": rows[0],
        "selected_measured": selected_measured,
        "measurement_summary": measurement_summary,
        "measurement_shortlist": shortlist,
        "measurement_shortlist_meta": {
            "reference_labels": composed_reference_labels(shortlist, reference_labels),
        },
        "scores": rows,
    }


def normalize_reports(reports: dict[str, Any]) -> dict[str, Any]:
    for component in ("witness", "fixed_value", "fixed_audit", "read"):
        if component not in reports:
            raise SystemExit(f"missing {component} report")
    read_reports = reports["read"]
    if isinstance(read_reports, dict):
        read_reports = [read_reports]
    if not isinstance(read_reports, list) or not read_reports:
        raise SystemExit("at least one read report is required")
    if any(not isinstance(report, dict) for report in read_reports):
        raise SystemExit("read reports must be JSON objects")
    return {
        "witness": reports["witness"],
        "fixed_value": reports["fixed_value"],
        "fixed_audit": reports["fixed_audit"],
        "read": read_reports,
    }


def named_reports(reports: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    named = [
        (component, reports[component])
        for component in ("witness", "fixed_value", "fixed_audit")
    ]
    named.extend(
        (f"read[{index}]", report)
        for index, report in enumerate(reports["read"])
    )
    return named


def require_matching_coefficients(reports: dict[str, Any]) -> None:
    coefficients = [report.get("coefficients") for _, report in named_reports(reports)]
    first = coefficients[0]
    if not isinstance(first, dict) or any(
        not isinstance(value, dict) for value in coefficients[1:]
    ):
        raise SystemExit("component reports must carry calibration coefficients")
    if any(value != first for value in coefficients[1:]):
        raise SystemExit("component reports use different calibration coefficients")
    verifier_coefficients = [
        report.get("verifier_coefficients") for _, report in named_reports(reports)
    ]
    verifier_first = verifier_coefficients[0]
    if any(value != verifier_first for value in verifier_coefficients[1:]):
        raise SystemExit("component reports use different verifier calibration coefficients")


def workload_identity_from_r1cs(path: Path, label: str) -> dict[str, Any]:
    if not label:
        raise SystemExit("workload label must not be empty")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SystemExit(f"failed to read workload R1CS {path}: {error}") from error
    return {"label": label, "r1cs_sha256": digest}


def require_witness_workload_identity(
    witness_report: dict[str, Any], expected: dict[str, Any] | None
) -> dict[str, Any]:
    actual = witness_report.get("workload_identity")
    if not isinstance(expected, dict):
        raise SystemExit("SPARK composition requires a workload identity")
    if not isinstance(actual, dict):
        raise SystemExit("witness report is missing workload_identity")
    constraint_work = actual.get("constraint_work")
    if not isinstance(constraint_work, int) or constraint_work <= 0:
        raise SystemExit("witness workload identity needs positive constraint_work")
    complete_expected = {**expected, "constraint_work": constraint_work}
    if actual != complete_expected:
        raise SystemExit("witness report workload identity differs from the requested workload")
    return complete_expected


def attach_workload_identity(
    rows: list[dict[str, Any]], workload_identity: dict[str, Any]
) -> None:
    for row in rows:
        row["workload_identity"] = json.loads(json.dumps(workload_identity))


def required_report_rows(
    frontier: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    measured: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_identity = {}
    for row in [*frontier, *shortlist, *measured]:
        by_identity.setdefault(composed_row_identity(row), row)
    return list(by_identity.values())


def candidate_retention_metadata(
    all_candidates: dict[str, list[dict[str, Any]]],
    band_candidates: dict[str, list[dict[str, Any]]],
    component_frontiers: dict[str, list[dict[str, Any]]],
    composed_frontier_count: int,
    required_report_row_count: int,
    top_per_component: int,
    fixed_audit_embedded: bool,
    prover_band: dict[str, Any],
) -> dict[str, Any]:
    components = {}
    for component in ("witness", "fixed_value", "fixed_audit", "read"):
        frontier = component_frontiers[component]
        components[component] = {
            "eligible_rows": len(all_candidates[component]),
            "rows_within_prover_selection_band": len(band_candidates[component]),
            "pareto_objective_points": len(
                {
                    (
                        component_score(row, component, fixed_audit_embedded),
                        component_verifier_score(
                            row, component, fixed_audit_embedded
                        ),
                        component_proof_size(row, component, fixed_audit_embedded),
                    )
                    for row in frontier
                }
            ),
            "pareto_rows": len(frontier),
        }
    return {
        "component_objectives": [
            "component_score",
            "component_verifier_score",
            "component_proof_size",
        ],
        "composed_objectives": [
            "projected_seconds",
            "verifier_projected_seconds",
            "proof_size_bytes_estimate",
        ],
        "top_per_component": top_per_component,
        "top_per_component_scope": "measurement_shortlist",
        "prover_selection_band": prover_band,
        "recursive_verifier_metric": "calibrated native verifier wall time",
        "recursive_verifier_used_for_ranking": "tie_break_only",
        "components": components,
        "composed_pareto_rows": composed_frontier_count,
        "required_report_rows": required_report_row_count,
    }


def prover_selection_band(
    candidates: dict[str, list[dict[str, Any]]], fixed_audit_embedded: bool
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    minima = {
        component: min(
            component_score(row, component, fixed_audit_embedded) for row in rows
        )
        for component, rows in candidates.items()
    }
    best_total = sum(minima.values())
    margin = best_total * PROVER_SELECTION_BAND_RELATIVE
    retained = {
        component: [
            row
            for row in rows
            if component_score(row, component, fixed_audit_embedded)
            <= minima[component] + margin + 1e-12
        ]
        for component, rows in candidates.items()
    }
    return retained, {
        "policy": "component excess bounded by one percent of the best composed prover projection",
        "relative_margin": PROVER_SELECTION_BAND_RELATIVE,
        "best_composed_projected_seconds": best_total,
        "margin_seconds": margin,
    }


def normalize_reference_labels(
    reference_labels: dict[str, Any], read_report_count: int
) -> dict[str, Any]:
    normalized = {
        component: reference_labels.get(component)
        for component in ("witness", "fixed_value", "fixed_audit")
    }
    read_labels = reference_labels.get("read")
    if isinstance(read_labels, str):
        read_labels = [read_labels]
    elif read_labels is not None:
        read_labels = list(read_labels)
    normalized["read"] = read_labels

    provided = [
        normalized["witness"] is not None,
        normalized["fixed_value"] is not None,
        normalized["fixed_audit"] is not None,
        bool(normalized["read"]),
    ]
    if not any(provided):
        return {}
    if not all(provided):
        raise SystemExit("all component reference labels are required together")
    if len(normalized["read"]) != read_report_count:
        raise SystemExit("one reference read label is required for each read report")
    return normalized


def composed_reference_labels(
    shortlist: list[dict[str, Any]], reference_labels: dict[str, Any]
) -> list[str]:
    if not reference_labels:
        return []
    components = ("witness", "fixed_value", "fixed_audit", "read")
    labels = [
        str(row["label"])
        for row in shortlist
        if all(
            row.get("component_labels", {}).get(component)
            == reference_labels[component]
            for component in components
        )
    ]
    if not labels:
        raise SystemExit("composed reference row is missing from measurement shortlist")
    return list(dict.fromkeys(labels))


def require_security_targets(
    reports: dict[str, Any],
    security_bits: int,
    merkle_security_bits: int,
) -> tuple[int, int]:
    expected_component_targets = SPARK_COMPONENT_TARGETS.get(
        (security_bits, merkle_security_bits)
    )
    if expected_component_targets is None:
        raise SystemExit(
            "the SPARK composer has no derived component targets for end-to-end "
            f"security {security_bits}/{merkle_security_bits}"
        )
    for component, report in named_reports(reports):
        report_security = report.get("target_security_bits")
        report_merkle = report.get("target_merkle_security_bits")
        if (report_security, report_merkle) != (security_bits, merkle_security_bits):
            raise SystemExit(
                f"{component} report targets end-to-end security "
                f"{report_security}/{report_merkle}, expected "
                f"{security_bits}/{merkle_security_bits}"
            )
        component_security = report.get("component_security_override_bits")
        component_merkle = report.get("component_merkle_security_override_bits")
        if not isinstance(component_security, int) or not isinstance(component_merkle, int):
            raise SystemExit(f"{component} report is missing explicit component security targets")
        if (component_security, component_merkle) != expected_component_targets:
            raise SystemExit(
                f"{component} report must use component security targets "
                f"{expected_component_targets[0]}/{expected_component_targets[1]}, got "
                f"{component_security}/{component_merkle}"
            )
    return expected_component_targets


def require_component_dimensions(
    reports: dict[str, Any], extension: str, fixed_audit_embedded: bool
) -> None:
    fixed_value_variables = positive_num_variables(reports["fixed_value"], "fixed_value")
    fixed_audit_variables = positive_num_variables(reports["fixed_audit"], "fixed_audit")
    read_variables = [
        positive_num_variables(report, f"read[{index}]")
        for index, report in enumerate(reports["read"])
    ]
    if fixed_value_variables <= FIXED_VALUE_COLUMN_BITS:
        raise SystemExit(
            "fixed_value num_variables is too small for the Spark fixed-value columns"
        )
    if fixed_audit_variables < FIXED_AUDIT_COLUMN_BITS:
        raise SystemExit(
            "fixed_audit num_variables is too small for the Spark audit columns"
        )

    if extension == "quintic":
        expected_read_variables = [fixed_value_variables, fixed_value_variables - 2]
    else:
        expected_read_variables = [fixed_value_variables + 1]
    if read_variables != expected_read_variables:
        raise SystemExit(
            f"{extension} Spark read reports must use num_variables "
            f"{expected_read_variables}, got {read_variables}"
        )

    value_variables = fixed_value_variables - FIXED_VALUE_COLUMN_BITS
    audit_memory_variables = fixed_audit_variables - FIXED_AUDIT_COLUMN_BITS
    dimensions_imply_embedded = audit_memory_variables + 1 <= value_variables
    if dimensions_imply_embedded and not fixed_audit_embedded:
        raise SystemExit(
            "fixed audit dimensions imply embedded audit tables; "
            "pass --fixed-audit-embedded"
        )
    if fixed_audit_embedded and not dimensions_imply_embedded:
        raise SystemExit(
            "fixed audit dimensions do not fit in the fixed value bundle; "
            "remove --fixed-audit-embedded"
        )


def positive_num_variables(report: dict[str, Any], component: str) -> int:
    num_variables = report.get("num_variables")
    if not isinstance(num_variables, int) or num_variables <= 0:
        raise SystemExit(f"{component} report is missing a positive num_variables")
    return num_variables


def stratified_shortlist(
    ranked: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    security_bits: int,
    merkle_security_bits: int,
    ell_zk: int | None,
    mask_log_inv_rate: int | None,
    limit: int,
    all_candidates: dict[str, list[dict[str, Any]]],
    reference_labels: dict[str, Any],
    proof_mode: str,
    extension: str,
    fixed_audit_embedded: bool,
) -> list[dict[str, Any]]:
    best = {component: candidates[0] for component, candidates in ranked.items()}
    selected = rows[: min(2, len(rows))]
    reference_row = None
    if any(reference_labels.values()):
        if not all(reference_labels.values()):
            raise SystemExit("all four reference labels are required together")
        reference = {
            component: find_candidate_row(
                all_candidates[component], label, component
            )
            for component, label in reference_labels.items()
        }
        reference_row = composed_row(
            reference["witness"],
            reference["fixed_value"],
            reference["fixed_audit"],
            reference["read"],
            security_bits,
            merkle_security_bits,
            ell_zk,
            mask_log_inv_rate,
            proof_mode,
            extension,
            fixed_audit_embedded,
        )
        selected = [rows[0], reference_row]
        for component in active_calibration_components(fixed_audit_embedded):
            alternatives = [
                candidate
                for candidate in ranked[component]
                if candidate_component_label(candidate, component)
                != reference_labels[component]
            ][:2]
            for alternative in alternatives:
                choice = dict(reference)
                choice[component] = alternative
                selected.append(
                    composed_row(
                        choice["witness"],
                        choice["fixed_value"],
                        choice["fixed_audit"],
                        choice["read"],
                        security_bits,
                        merkle_security_bits,
                        ell_zk,
                        mask_log_inv_rate,
                        proof_mode,
                        extension,
                        fixed_audit_embedded,
                    )
                )
    else:
        for component in active_calibration_components(fixed_audit_embedded):
            for alternative in ranked[component][1:3]:
                choice = dict(best)
                choice[component] = alternative
                selected.append(
                    composed_row(
                        choice["witness"],
                        choice["fixed_value"],
                        choice["fixed_audit"],
                        choice["read"],
                        security_bits,
                        merkle_security_bits,
                        ell_zk,
                        mask_log_inv_rate,
                        proof_mode,
                        extension,
                        fixed_audit_embedded,
                    )
                )
    selected = deduplicate_rows(selected)
    limited = selected[:limit]
    if reference_row is not None and all(
        row["label"] != reference_row["label"] for row in limited
    ):
        reference_setup = json.dumps(reference_row["setup_config"], sort_keys=True)
        limited = [
            row
            for row in limited
            if json.dumps(row["setup_config"], sort_keys=True) != reference_setup
        ]
        limited.append(reference_row)
    return limited


def find_candidate_row(
    candidates: list[dict[str, Any]], label: Any, component: str
) -> dict[str, Any]:
    for row in candidates:
        if candidate_component_label(row, component) == label:
            return row
    raise SystemExit(f"{component} reference label not found: {label}")


def component_rows(
    report: dict[str, Any],
    component: str,
    max_setup_log_domain: int | None,
    extension: str,
    fixed_audit_embedded: bool = False,
    zk_parameters: tuple[int, int] | None = None,
    expected_proof_mode: str = "no-zk",
    component_security_bits: int = 120,
    component_merkle_security_bits: int = 123,
    workload_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source_rows = report.get("scores")
    if not isinstance(source_rows, list):
        raise SystemExit(f"{component} report scores must be an array")
    rows = []
    for index, source in enumerate(source_rows):
        if not isinstance(source, dict):
            raise SystemExit(f"{component} score row {index} is not an object")
        if source.get("accepted_for_ranking") is not True:
            continue
        require_component_row_context(
            source,
            component,
            index,
            expected_proof_mode,
            component_security_bits,
            component_merkle_security_bits,
            workload_identity,
        )
        if source.get("extension") != extension:
            continue
        if zk_parameters is not None and (
            source.get("zk_ell") != zk_parameters[0]
            or source.get("zk_mask_log_inv_rate") != zk_parameters[1]
        ):
            continue
        row = {**source, "_component_num_variables": report.get("num_variables")}
        if within_setup_domain_cap(report, row, max_setup_log_domain):
            rows.append(row)
    require_unique_component_labels(rows, component)
    rows.sort(
        key=lambda row: (
            component_score(row, component, fixed_audit_embedded),
            component_proof_size(row, component, fixed_audit_embedded),
            row["label"],
        )
    )
    if not rows:
        if zk_parameters is not None:
            raise SystemExit(
                f"{component} report has no valid {extension} rows for "
                f"ell_zk={zk_parameters[0]} and "
                f"mask_log_inv_rate={zk_parameters[1]}"
            )
        raise SystemExit(f"{component} report has no valid {extension} rows")
    return rows


def require_component_row_context(
    row: dict[str, Any],
    component: str,
    index: int,
    expected_proof_mode: str,
    component_security_bits: int,
    component_merkle_security_bits: int,
    workload_identity: dict[str, Any] | None,
) -> None:
    prefix = f"{component} score row {index}"
    label = row.get("label")
    if not isinstance(label, str) or not label:
        raise SystemExit(f"{prefix} is missing a nonempty label")
    if row.get("valid") is not True:
        raise SystemExit(f"{prefix} is accepted for ranking but not valid")
    if row.get("proof_mode") != expected_proof_mode:
        raise SystemExit(f"{prefix} must use proof_mode={expected_proof_mode}")
    if row.get("base_field") != "koalabear":
        raise SystemExit(f"{prefix} must use base_field=koalabear")
    if row.get("whir_component_security_bits") != component_security_bits:
        raise SystemExit(
            f"{prefix} must use WHIR component security {component_security_bits}"
        )
    if row.get("merkle_component_security_bits") != component_merkle_security_bits:
        raise SystemExit(
            f"{prefix} must use Merkle component security "
            f"{component_merkle_security_bits}"
        )
    achieved = row.get("security_bits_achieved")
    if (
        not isinstance(achieved, (int, float))
        or not math.isfinite(float(achieved))
        or float(achieved) < component_security_bits
    ):
        raise SystemExit(
            f"{prefix} must achieve at least {component_security_bits} bits"
        )
    if not isinstance(row.get("whir_params"), dict):
        raise SystemExit(f"{prefix} is missing WhirParams")
    if workload_identity is not None and row.get("workload_identity") != workload_identity:
        raise SystemExit(f"{prefix} workload identity differs from the witness report")


def require_unique_component_labels(
    rows: list[dict[str, Any]], component: str
) -> None:
    seen = set()
    for row in rows:
        label = str(row["label"])
        if label in seen:
            raise SystemExit(f"{component} report has duplicate eligible label {label}")
        seen.add(label)


def component_pareto_rows(
    rows: list[dict[str, Any]],
    component: str,
    fixed_audit_embedded: bool,
) -> list[dict[str, Any]]:
    return pareto_rows_3d(
        rows,
        lambda row: component_score(row, component, fixed_audit_embedded),
        lambda row: component_verifier_score(row, component, fixed_audit_embedded),
        lambda row: component_proof_size(row, component, fixed_audit_embedded),
        lambda row: json.dumps(row["whir_params"], sort_keys=True, separators=(",", ":")),
    )


def pareto_rows(
    rows: list[dict[str, Any]],
    time_value,
    size_value,
    configuration_identity,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(time_value(row)),
            int(size_value(row)),
            str(row.get("label") or ""),
        ),
    )
    selected = []
    seen_configurations = set()
    best_size = None
    best_time_for_size = None
    for row in ordered:
        identity = configuration_identity(row)
        if identity in seen_configurations:
            continue
        seen_configurations.add(identity)
        time = float(time_value(row))
        size = int(size_value(row))
        if best_size is None or size < best_size:
            selected.append(row)
            best_size = size
            best_time_for_size = time
        elif size == best_size and time == best_time_for_size:
            selected.append(row)
    return selected


def pareto_rows_3d(
    rows: list[dict[str, Any]],
    time_value,
    verifier_value,
    size_value,
    configuration_identity,
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
    # Fenwick prefix minima over verifier work. Each entry stores
    # (proof size, prover time, verifier work) for the best prior point.
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
        identity = configuration_identity(row)
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


def embedded_fixed_audit_candidate(
    report: dict[str, Any],
    candidates: list[dict[str, Any]],
    reference_label: Any,
) -> dict[str, Any]:
    if reference_label is not None:
        return find_candidate_row(candidates, reference_label, "fixed_audit")
    selected = report.get("selected")
    if not isinstance(selected, dict):
        raise SystemExit(
            "embedded fixed audit requires a reference label or report.selected"
        )
    identity = component_candidate_identity(selected)
    matches = [row for row in candidates if component_candidate_identity(row) == identity]
    if len(matches) != 1:
        raise SystemExit(
            "embedded fixed-audit report.selected does not identify one eligible row"
        )
    return matches[0]


def component_candidate_identity(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "label": row.get("label"),
            "extension": row.get("extension"),
            "whir_params": row.get("whir_params"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def composed_pareto_rows(
    components: dict[str, list[dict[str, Any]]],
    security_bits: int,
    merkle_security_bits: int,
    ell_zk: int | None,
    mask_log_inv_rate: int | None,
    proof_mode: str,
    extension: str,
    fixed_audit_embedded: bool,
    calibration: dict[str, Any] | None,
    prover_band_relative: float | None = None,
) -> list[dict[str, Any]]:
    component_order = ("witness", "fixed_value", "fixed_audit", "read")
    component_minima = {
        component: min(
            calibrated_component_score(
                candidate, component, fixed_audit_embedded, calibration
            )
            for candidate in components[component]
        )
        for component in component_order
    }
    best_total = sum(component_minima.values())
    cutoff = (
        best_total * (1.0 + prover_band_relative)
        if prover_band_relative is not None
        else float("inf")
    )
    partials = [{"choices": {}, "time": 0.0, "verifier": 0.0, "size": 0}]
    for component_index, component in enumerate(component_order):
        remaining_minimum = sum(
            component_minima[name] for name in component_order[component_index + 1 :]
        )
        expanded = []
        for partial in partials:
            for candidate in components[component]:
                expanded.append(
                    {
                        "choices": {**partial["choices"], component: candidate},
                        "time": float(partial["time"])
                        + calibrated_component_score(
                            candidate,
                            component,
                            fixed_audit_embedded,
                            calibration,
                        ),
                        "verifier": float(partial["verifier"])
                        + component_verifier_score(
                            candidate, component, fixed_audit_embedded
                        ),
                        "size": int(partial["size"])
                        + component_proof_size(
                            candidate, component, fixed_audit_embedded
                        ),
                    }
                )
        expanded = [
            partial
            for partial in expanded
            if float(partial["time"]) + remaining_minimum <= cutoff + 1e-12
        ]
        partials = pareto_rows_3d(
            expanded,
            lambda partial: partial["time"],
            lambda partial: partial["verifier"],
            lambda partial: partial["size"],
            partial_configuration_identity,
        )
    rows = []
    for partial in partials:
        choices = partial["choices"]
        rows.append(
            composed_row(
                choices["witness"],
                choices["fixed_value"],
                choices["fixed_audit"],
                choices["read"],
                security_bits,
                merkle_security_bits,
                ell_zk,
                mask_log_inv_rate,
                proof_mode,
                extension,
                fixed_audit_embedded,
            )
        )
    rows = deduplicate_rows(rows)
    if calibration is not None:
        apply_calibration(rows, calibration, fixed_audit_embedded)
    return pareto_rows_3d(
        rows,
        lambda row: row["projected_seconds"],
        lambda row: row["verifier_projected_seconds"],
        proof_size,
        lambda row: json.dumps(row["setup_config"], sort_keys=True, separators=(",", ":")),
    )


def calibrated_component_score(
    row: dict[str, Any],
    component: str,
    fixed_audit_embedded: bool,
    calibration: dict[str, Any] | None,
) -> float:
    score = component_score(row, component, fixed_audit_embedded)
    if calibration is None:
        return score
    component_scales = calibration.get("component_scales")
    if isinstance(component_scales, dict):
        return float(component_scales.get(component, 0.0)) * score
    return float(calibration.get("scale", 1.0)) * score


def partial_configuration_identity(partial: dict[str, Any]) -> str:
    return json.dumps(
        {
            component: candidate.get("whir_params")
            for component, candidate in partial["choices"].items()
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def shared_read_rows(
    reports: list[dict[str, Any]],
    extension: str,
    component_security_bits: int,
    component_merkle_security_bits: int,
) -> list[dict[str, Any]]:
    rows_by_report = [
        component_rows(
            report,
            "read",
            None,
            extension,
            expected_proof_mode="no-zk",
            component_security_bits=component_security_bits,
            component_merkle_security_bits=component_merkle_security_bits,
        )
        for report in reports
    ]
    multiple_reports = len(reports) > 1
    indexed_rows = []
    for report_index, (report, rows) in enumerate(zip(reports, rows_by_report)):
        by_params: dict[str, dict[str, Any]] = {}
        for row in rows:
            params = row["whir_params"]
            if multiple_reports:
                require_derived_or_empty_round_rates(
                    params,
                    report.get("num_variables"),
                    report_index,
                    str(row["label"]),
                )
            normalized_params = normalized_shared_read_params(params, multiple_reports)
            key = json.dumps(normalized_params, sort_keys=True, separators=(",", ":"))
            candidate = {**row, "_normalized_whir_params": normalized_params}
            current = by_params.get(key)
            if current is None or (
                component_score(candidate, "read"),
                component_verifier_score(candidate, "read"),
                proof_size(candidate),
                str(candidate["label"]),
            ) < (
                component_score(current, "read"),
                component_verifier_score(current, "read"),
                proof_size(current),
                str(current["label"]),
            ):
                by_params[key] = candidate
        indexed_rows.append(by_params)

    shared_keys = set(indexed_rows[0])
    for by_params in indexed_rows[1:]:
        shared_keys.intersection_update(by_params)
    if not shared_keys:
        raise SystemExit(
            f"read reports have no shared {extension} WhirParams tuple"
        )

    shared = []
    for key in shared_keys:
        source_rows = [by_params[key] for by_params in indexed_rows]
        labels = [str(row["label"]) for row in source_rows]
        shared.append(
            {
                "label": "__".join(labels),
                "extension": extension,
                "valid": True,
                "proof_size_bytes_estimate": sum(proof_size(row) for row in source_rows),
                "whir_params": source_rows[0]["_normalized_whir_params"],
                "_component_score": sum(
                    component_score(row, "read") for row in source_rows
                ),
                "_component_verifier_score": sum(
                    component_verifier_score(row, "read") for row in source_rows
                ),
                "_source_labels": labels,
                "_component_num_variables": [
                    report.get("num_variables") for report in reports
                ],
            }
        )
    shared.sort(
        key=lambda row: (
            component_score(row, "read"),
            component_verifier_score(row, "read"),
            proof_size(row),
            str(row["label"]),
        )
    )
    return shared


def normalized_shared_read_params(
    params: dict[str, Any], clear_round_rates: bool
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(params))
    if clear_round_rates:
        normalized["round_log_inv_rates"] = []
    return normalized


def require_derived_or_empty_round_rates(
    params: dict[str, Any],
    num_variables: Any,
    report_index: int,
    label: str,
) -> None:
    rates = params.get("round_log_inv_rates", [])
    if not isinstance(rates, list) or any(
        not isinstance(rate, int) or rate < 0 for rate in rates
    ):
        raise SystemExit(
            f"read[{report_index}] row {label} has invalid round_log_inv_rates"
        )
    derived = derived_round_log_inv_rates(num_variables, params)
    if rates and rates != derived:
        raise SystemExit(
            f"read[{report_index}] row {label} must use derived or empty "
            "round_log_inv_rates"
        )


def derived_round_log_inv_rates(
    num_variables: Any, params: dict[str, Any]
) -> list[int]:
    if not isinstance(num_variables, int) or num_variables <= 0:
        raise SystemExit("read report is missing a positive num_variables")
    schedule = params.get("folding_schedule")
    if schedule is None:
        schedule_kind = "constant"
        schedule_value: Any = params.get("folding_factor")
    elif isinstance(schedule, dict) and set(schedule) == {"Constant"}:
        schedule_kind = "constant"
        schedule_value = schedule["Constant"]
    elif isinstance(schedule, dict) and set(schedule) == {"ConstantFromSecondRound"}:
        schedule_kind = "constant_from_second_round"
        schedule_value = schedule["ConstantFromSecondRound"]
    elif isinstance(schedule, dict) and set(schedule) == {"PerRound"}:
        schedule_kind = "per_round"
        schedule_value = schedule["PerRound"]
    else:
        raise SystemExit("read WhirParams contain an invalid folding_schedule")

    factors = folding_factors(num_variables, schedule_kind, schedule_value)
    starting_rate = params.get("starting_log_inv_rate")
    initial_reduction = params.get("rs_domain_initial_reduction_factor")
    if not isinstance(starting_rate, int) or starting_rate < 0:
        raise SystemExit("read WhirParams contain an invalid starting_log_inv_rate")
    if not isinstance(initial_reduction, int) or initial_reduction < 0:
        raise SystemExit(
            "read WhirParams contain an invalid rs_domain_initial_reduction_factor"
        )

    rate = starting_rate
    rates = []
    for round_index, factor in enumerate(factors[:-1]):
        reduction = initial_reduction if round_index == 0 else 1
        rate += factor - reduction
        if rate < 0:
            raise SystemExit("read WhirParams derive a negative round log inverse rate")
        rates.append(rate)
    return rates


def folding_factors(
    num_variables: int, schedule_kind: str, schedule_value: Any
) -> list[int]:
    remaining = num_variables
    factors = []
    if schedule_kind == "constant":
        if not isinstance(schedule_value, int) or not 0 < schedule_value <= num_variables:
            raise SystemExit("read WhirParams contain an invalid constant folding schedule")
        while True:
            factor = min(schedule_value, remaining)
            factors.append(factor)
            remaining -= factor
            if remaining <= 6:
                return factors

    if schedule_kind == "constant_from_second_round":
        if not isinstance(schedule_value, dict):
            raise SystemExit("read WhirParams contain an invalid folding schedule")
        first = schedule_value.get("first")
        rest = schedule_value.get("rest")
        if (
            not isinstance(first, int)
            or not isinstance(rest, int)
            or not 0 < first <= num_variables
            or not 0 < rest <= num_variables
        ):
            raise SystemExit("read WhirParams contain an invalid folding schedule")
        factors.append(first)
        remaining -= first
        while remaining > 6:
            factor = min(rest, remaining)
            factors.append(factor)
            remaining -= factor
        return factors

    if not isinstance(schedule_value, list) or not schedule_value:
        raise SystemExit("read WhirParams contain an invalid per-round folding schedule")
    for factor in schedule_value:
        if not isinstance(factor, int) or not 0 < factor <= remaining:
            raise SystemExit("read WhirParams contain an invalid per-round folding schedule")
        factors.append(factor)
        remaining -= factor
        if remaining <= 6:
            return factors
    raise SystemExit("read WhirParams folding schedule does not reach the final sumcheck")


def within_setup_domain_cap(
    report: dict[str, Any], row: dict[str, Any], max_setup_log_domain: int | None
) -> bool:
    if max_setup_log_domain is None:
        return True
    num_variables = report.get("num_variables")
    starting_log_inv_rate = (row.get("whir_params") or {}).get("starting_log_inv_rate")
    if not isinstance(num_variables, int) or not isinstance(starting_log_inv_rate, int):
        return False
    return num_variables + starting_log_inv_rate <= max_setup_log_domain


def component_score(
    row: dict[str, Any], component: str, fixed_audit_embedded: bool = False
) -> float:
    if component == "fixed_audit" and fixed_audit_embedded:
        return 0.0
    if component == "read" and row.get("_component_score") is not None:
        return float(row["_component_score"])
    costs = row.get("cost_breakdown") or {}
    if component == "witness":
        names = ("dft", "merkle", "merkle_path", "row_opening", "sumcheck", "pow")
        multiplier = 1.0
    elif component in ("fixed_value", "fixed_audit"):
        names = ("merkle_path", "row_opening", "sumcheck", "pow")
        multiplier = 1.0
    elif component == "read":
        names = ("dft", "merkle", "merkle_path", "row_opening", "sumcheck", "pow")
        multiplier = 1.0
    else:
        raise AssertionError(component)
    score = multiplier * sum(float(costs.get(name) or 0.0) for name in names)
    if component in ("fixed_value", "fixed_audit"):
        score += post_setup_commitment_cost(row)
    return score


def component_verifier_score(
    row: dict[str, Any], component: str, fixed_audit_embedded: bool = False
) -> float:
    if component == "fixed_audit" and fixed_audit_embedded:
        return 0.0
    if component == "read" and row.get("_component_verifier_score") is not None:
        return float(row["_component_verifier_score"])
    return float(row.get("verifier_projected_seconds") or 0.0)


def component_proof_size(
    row: dict[str, Any], component: str, fixed_audit_embedded: bool = False
) -> int:
    if component == "fixed_audit" and fixed_audit_embedded:
        return 0
    return proof_size(row)


def post_setup_commitment_cost(row: dict[str, Any]) -> float:
    num_variables = row.get("_component_num_variables")
    params = row.get("whir_params") or {}
    starting_log_inv_rate = params.get("starting_log_inv_rate")
    if not isinstance(num_variables, int) or not isinstance(starting_log_inv_rate, int):
        return 0.0
    initial_work = 1 << (num_variables + starting_log_inv_rate)
    costs = row.get("cost_breakdown") or {}
    total = 0.0
    for cost_name, work_name in (("dft", "dft_work"), ("merkle", "merkle_work")):
        work = int(row.get(work_name) or 0)
        if work <= 0:
            continue
        post_setup_work = max(0, work - initial_work)
        total += float(costs.get(cost_name) or 0.0) * post_setup_work / work
    return total


def composed_row(
    witness: dict[str, Any],
    fixed_value: dict[str, Any],
    fixed_audit: dict[str, Any],
    read: dict[str, Any],
    security_bits: int,
    merkle_security_bits: int,
    ell_zk: int | None,
    mask_log_inv_rate: int | None,
    proof_mode: str,
    extension: str,
    fixed_audit_embedded: bool,
) -> dict[str, Any]:
    components = {
        "witness": witness,
        "fixed_value": fixed_value,
        "fixed_audit": fixed_audit,
        "read": read,
    }
    scores = {
        component: component_score(row, component, fixed_audit_embedded)
        for component, row in components.items()
    }
    verifier_scores = {
        component: component_verifier_score(row, component, fixed_audit_embedded)
        for component, row in components.items()
    }
    projected = sum(scores.values())
    verifier_projected = sum(verifier_scores.values())
    size = sum(
        component_proof_size(row, component, fixed_audit_embedded)
        for component, row in components.items()
    )
    labels: dict[str, Any] = {
        component: str(row["label"])
        for component, row in components.items()
        if component != "read"
    }
    labels["read"] = list(read.get("_source_labels") or [str(read["label"])])
    label = "spark_" + "__".join(
        f"{short}={component_label_text(labels[component])}"
        for component, short in (
            ("witness", "w"),
            ("fixed_value", "fv"),
            ("fixed_audit", "fa"),
            ("read", "r"),
        )
    )
    setup_config = {
        "matrix_closing": "Spark",
        "security": {
            "security_level_bits": security_bits,
            "merkle_security_bits": merkle_security_bits,
            "soundness_assumption": "JohnsonBound",
        },
        "whir_params": witness["whir_params"],
        "spark_whir_params": {
            "fixed_value": fixed_value["whir_params"],
            "fixed_audit": fixed_audit["whir_params"],
            "read": read["whir_params"],
        },
    }
    if proof_mode == "full-zk":
        if ell_zk is None or mask_log_inv_rate is None:
            raise SystemExit("full-zk composition requires ZK mask parameters")
        setup_config["ell_zk"] = ell_zk
        setup_config["mask_log_inv_rate"] = mask_log_inv_rate
    return {
        "label": label,
        "extension": extension,
        "valid": True,
        "proof_mode": proof_mode,
        "matrix_closing": "Spark",
        "projected_schedule_seconds": projected,
        "projected_seconds": projected,
        "verifier_projected_seconds": verifier_projected,
        "proof_size_bytes_estimate": size,
        "component_labels": labels,
        "component_scores": scores,
        "component_verifier_scores": verifier_scores,
        "fixed_audit_embedded": fixed_audit_embedded,
        "setup_config": setup_config,
    }


def component_label_text(label: Any) -> str:
    if isinstance(label, list):
        return "__".join(str(value) for value in label)
    return str(label)


def candidate_component_label(row: dict[str, Any], component: str) -> Any:
    if component == "read":
        return list(row.get("_source_labels") or [str(row["label"])])
    return str(row["label"])


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = json.dumps(row["setup_config"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def calibrate(
    rows: list[dict[str, Any]],
    measurements: dict[str, Any] | None,
    reference_labels: dict[str, Any] | None = None,
    fixed_audit_embedded: bool = False,
) -> dict[str, Any] | None:
    if measurements is None:
        return None
    measured = measurement_map(measurements)
    pairs = [
        (float(row["projected_schedule_seconds"]), measured[row["label"]])
        for row in rows
        if row["label"] in measured
    ]
    if len(pairs) < 3:
        raise SystemExit("SPARK calibration requires at least three measured rows")
    if reference_labels and all(reference_labels.values()):
        component_calibration = calibrate_components(
            rows,
            measured,
            reference_labels,
            active_calibration_components(fixed_audit_embedded),
        )
        if component_calibration is not None:
            return component_calibration
    calibration_pairs = [pair for index, pair in enumerate(pairs) if index % 3 != 2]
    validation_pairs = [pair for index, pair in enumerate(pairs) if index % 3 == 2]
    if not validation_pairs:
        validation_pairs = [calibration_pairs.pop()]
    scale = 1.0
    intercept = sum(y - x for x, y in calibration_pairs) / len(calibration_pairs)
    errors = [relative_error(intercept + scale * x, y) for x, y in validation_pairs]
    return {
        "method": "unit-scale offset fit with every third measured row held out",
        "intercept_seconds": intercept,
        "scale": scale,
        "calibration_rows": len(calibration_pairs),
        "validation_rows": len(validation_pairs),
        "validation_max_relative_error": max(errors),
        "validation_within_ten_percent": max(errors) <= 0.10,
    }


def calibrate_components(
    rows: list[dict[str, Any]],
    measured: dict[str, float],
    reference_labels: dict[str, Any],
    active_components: tuple[str, ...] = (
        "witness",
        "fixed_value",
        "fixed_audit",
        "read",
    ),
) -> dict[str, Any] | None:
    components = ("witness", "fixed_value", "fixed_audit", "read")
    reference = next(
        (
            row
            for row in rows
            if row["label"] in measured
            and all(row["component_labels"][name] == reference_labels[name] for name in components)
        ),
        None,
    )
    if reference is None:
        return None

    reference_time = measured[reference["label"]]
    scales = {}
    fallback_components = []
    calibration_labels = {reference["label"]}
    for component in active_components:
        slopes = []
        for row in rows:
            if row["label"] not in measured:
                continue
            if any(
                row["component_labels"][name] != reference_labels[name]
                for name in components
                if name != component
            ):
                continue
            delta_score = float(row["component_scores"][component]) - float(
                reference["component_scores"][component]
            )
            if delta_score == 0.0:
                continue
            slopes.append((measured[row["label"]] - reference_time) / delta_score)
            calibration_labels.add(row["label"])
        if not slopes:
            return None
        slopes.sort()
        fitted_scale = slopes[len(slopes) // 2]
        if fitted_scale <= 0.0:
            fitted_scale = 1.0
            fallback_components.append(component)
        scales[component] = fitted_scale

    intercept = reference_time - sum(
        scales[component] * float(reference["component_scores"][component])
        for component in active_components
    )
    validation_rows = [
        row
        for row in rows
        if row["label"] in measured and row["label"] not in calibration_labels
    ]
    errors = [
        relative_error(
            intercept
            + sum(
                scales[component] * float(row["component_scores"][component])
                for component in active_components
            ),
            measured[row["label"]],
        )
        for row in validation_rows
    ]
    return {
        "method": "per-component slopes from configured-reference perturbations",
        "intercept_seconds": intercept,
        "component_scales": scales,
        "unit_scale_fallback_components": fallback_components,
        "calibration_rows": len(calibration_labels),
        "validation_rows": len(validation_rows),
        "validation_max_relative_error": max(errors) if errors else None,
        "validation_within_ten_percent": bool(errors) and max(errors) <= 0.10,
    }


def active_calibration_components(fixed_audit_embedded: bool) -> tuple[str, ...]:
    if fixed_audit_embedded:
        return ("witness", "fixed_value", "read")
    return ("witness", "fixed_value", "fixed_audit", "read")


def fit_affine(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    variance = sum((x - x_mean) ** 2 for x, _ in pairs)
    if variance == 0.0:
        return y_mean - x_mean, 1.0
    scale = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / variance
    return y_mean - scale * x_mean, scale


def apply_calibration(
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    fixed_audit_embedded: bool = False,
) -> None:
    intercept = float(calibration["intercept_seconds"])
    component_scales = calibration.get("component_scales")
    if component_scales is not None:
        for row in rows:
            row["projected_seconds"] = intercept + sum(
                float(component_scales[component]) * float(row["component_scores"][component])
                for component in active_calibration_components(fixed_audit_embedded)
            )
        return
    scale = float(calibration["scale"])
    for row in rows:
        row["projected_seconds"] = intercept + scale * float(row["projected_schedule_seconds"])


def calibrate_verifier(
    rows: list[dict[str, Any]], measurements: dict[str, Any] | None
) -> dict[str, Any] | None:
    if measurements is None:
        return None
    measured = {
        str(row["label"]): float(row["heldout_verify_seconds"])
        for row in measurements.get("rows", [])
        if row.get("heldout_verify_seconds") is not None
    }
    rows_by_label = {str(row["label"]): row for row in rows}
    pairs = []
    for measured_row in measurements.get("rows", []):
        label = str(measured_row.get("label") or "")
        row = rows_by_label.get(label)
        if row is None or label not in measured:
            continue
        projected = float(row.get("verifier_projected_seconds") or 0.0)
        if projected > 0.0:
            pairs.append((projected, measured[label]))
    if not pairs:
        return None
    if len(pairs) >= 4:
        calibration_pairs = pairs[::2]
        validation_pairs = pairs[1::2]
    else:
        calibration_pairs = pairs
        validation_pairs = []
    if len(calibration_pairs) == 1:
        intercept, scale = (
            0.0,
            calibration_pairs[0][1] / calibration_pairs[0][0],
        )
        method = "single-point scale"
    else:
        intercept, scale = fit_affine(calibration_pairs)
        method = "affine fit over measured verifier rows"
        if scale <= 0.0:
            ratios = sorted(
                measured_value / projected
                for projected, measured_value in calibration_pairs
            )
            intercept, scale = 0.0, ratios[len(ratios) // 2]
            method = "median scale because affine slope was non-positive"
    calibration_errors = [
        relative_error(intercept + scale * projected, measured_value)
        for projected, measured_value in calibration_pairs
    ]
    validation_errors = [
        relative_error(intercept + scale * projected, measured_value)
        for projected, measured_value in validation_pairs
    ]
    return {
        "method": method,
        "measurement": "native wall time",
        "rayon_threads": measurements.get("rayon_threads"),
        "source_measurements": measurements.get("source_report"),
        "rows": len(pairs),
        "calibration_rows": len(calibration_pairs),
        "validation_rows": len(validation_pairs),
        "intercept_seconds": intercept,
        "scale": scale,
        "calibration_max_relative_error": max(calibration_errors),
        "validation_max_relative_error": (
            max(validation_errors) if validation_errors else None
        ),
        "validation_within_twenty_percent": bool(validation_errors)
        and max(validation_errors) <= 0.20,
    }


def apply_verifier_calibration(
    rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> None:
    intercept = float(calibration["intercept_seconds"])
    scale = float(calibration["scale"])
    for row in rows:
        component_projection = float(row.get("verifier_projected_seconds") or 0.0)
        row["verifier_component_projection_seconds"] = component_projection
        row["verifier_projected_seconds"] = intercept + scale * component_projection


def measured_selection(
    rows: list[dict[str, Any]],
    measurements: dict[str, Any] | None,
    verifier_measurements: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if measurements is None:
        return None, None
    measured_rows = {
        row["label"]: row
        for row in measurements.get("rows", [])
        if row.get("measured_seconds") is not None
    }
    verifier_source = verifier_measurements or measurements
    verifier_rows = {
        row["label"]: row
        for row in (verifier_source or {}).get("rows", [])
        if row.get("heldout_verify_seconds") is not None
    }
    candidates = [row for row in rows if row["label"] in measured_rows]
    if not candidates:
        return None, {
            "source_measurements": measurements.get("source_report"),
            "measured_rows": 0,
            "tied_with_best_count": 0,
            "tied_with_best": [],
        }
    candidates.sort(
        key=lambda row: (
            float(measured_rows[row["label"]]["measured_seconds"]),
            str(row["label"]),
        )
    )
    fastest = candidates[0]
    fastest_measurement = measured_rows[fastest["label"]]
    tied = [
        row
        for row in candidates
        if measurement_is_tied(measured_rows[row["label"]], fastest_measurement)
    ]
    fastest_verifier = min(tied, key=lambda row: verifier_seconds(row, verifier_rows))
    verifier_tied = [
        row
        for row in tied
        if verifier_measurement_is_tied(
            row,
            fastest_verifier,
            verifier_rows,
        )
    ]
    selected = min(
        verifier_tied,
        key=lambda row: (
            measured_size(measured_rows[row["label"]]),
            row["label"],
        ),
    )
    out = dict(selected)
    out.update(
        {
            "measured_seconds": measured_rows[selected["label"]]["measured_seconds"],
            "heldout_median_ci_seconds": measured_rows[selected["label"]].get(
                "heldout_median_ci_seconds"
            ),
            "heldout_proof_size_median_bytes": measured_size(
                measured_rows[selected["label"]]
            ),
            "heldout_verify_seconds": verifier_rows.get(selected["label"], {}).get(
                "heldout_verify_seconds"
            ),
        }
    )
    tied_details = [
        measured_candidate_summary(
            row,
            measured_rows[row["label"]],
            fastest_measurement,
            candidates.index(row) + 1,
            verifier_rows.get(row["label"]),
        )
        for row in tied
    ]
    verifier_tied_labels = {row["label"] for row in verifier_tied}
    return out, {
        "source_measurements": measurements.get("source_report"),
        "measured_rows": len(candidates),
        "selection": "one_percent_prover_band_then_one_percent_or_overlapping_verifier_median_ci_then_proof_size_then_label",
        "prover_selection_band_relative": PROVER_SELECTION_BAND_RELATIVE,
        "verifier_selection_band_relative": VERIFIER_SELECTION_BAND_RELATIVE,
        "paired_confidence_interval_role": "diagnostic_only",
        "prover_median_confidence_interval_role": "diagnostic_only",
        "source_verifier_measurements": (
            verifier_source.get("source_report") if verifier_source else None
        ),
        "verifier_rayon_threads": (
            verifier_source.get("rayon_threads") if verifier_source else None
        ),
        "tied_with_best_count": len(tied_details),
        "tied_with_best": tied_details,
        "verifier_tied_count": len(verifier_tied_labels),
        "verifier_tied": [
            row for row in tied_details if row["label"] in verifier_tied_labels
        ],
    }


def select_measured(
    rows: list[dict[str, Any]], measurements: dict[str, Any] | None
) -> dict[str, Any] | None:
    selected, _summary = measured_selection(rows, measurements)
    return selected


def measured_candidate_summary(
    candidate: dict[str, Any],
    measurement: dict[str, Any],
    fastest: dict[str, Any],
    measured_rank: int,
    verifier_measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fastest_seconds = float(fastest["measured_seconds"])
    measured_seconds = float(measurement["measured_seconds"])
    relative_slowdown = (
        (measured_seconds - fastest_seconds) / fastest_seconds
        if fastest_seconds > 0.0
        else float("inf")
    )
    configured_pow_bits = component_pow_bits(candidate)
    per_proof_pow_bits = dict(configured_pow_bits)
    configuration_only_pow_bits = {}
    if candidate.get("fixed_audit_embedded"):
        configuration_only_pow_bits["fixed_audit"] = per_proof_pow_bits.pop(
            "fixed_audit", None
        )
    numeric_pow_bits = [
        value for value in per_proof_pow_bits.values() if isinstance(value, int)
    ]
    return {
        "measured_rank": measured_rank,
        "label": candidate["label"],
        "extension": candidate.get("extension"),
        "proof_mode": candidate.get("proof_mode"),
        "setup_config": candidate.get("setup_config"),
        "component_labels": candidate.get("component_labels"),
        "measured_seconds": measured_seconds,
        "relative_slowdown_from_fastest": relative_slowdown,
        "heldout_median_ci_seconds": measurement.get("heldout_median_ci_seconds"),
        "heldout_relative_median_difference": measurement.get(
            "heldout_relative_median_difference"
        ),
        "heldout_paired_relative_median_ci": measurement.get(
            "heldout_paired_relative_median_ci"
        ),
        "proof_size_bytes_estimate": proof_size(candidate),
        "heldout_proof_size_median_bytes": measurement.get(
            "heldout_proof_size_median_bytes"
        ),
        "heldout_verify_seconds": (
            verifier_measurement.get("heldout_verify_seconds")
            if verifier_measurement is not None
            else None
        ),
        "heldout_verify_median_ci_seconds": (
            verifier_measurement.get("heldout_verify_median_ci_seconds")
            if verifier_measurement is not None
            else None
        ),
        "selection_proof_size_bytes": measured_size(measurement),
        "configured_component_pow_bits": configured_pow_bits,
        "per_proof_component_pow_bits": per_proof_pow_bits,
        "configuration_only_pow_bits": configuration_only_pow_bits,
        "max_per_proof_component_pow_bits": (
            max(numeric_pow_bits) if numeric_pow_bits else None
        ),
    }


def component_pow_bits(row: dict[str, Any]) -> dict[str, Any]:
    setup = row.get("setup_config") or {}
    spark = setup.get("spark_whir_params") or {}
    return {
        "witness": (setup.get("whir_params") or {}).get("pow_bits"),
        "fixed_value": (spark.get("fixed_value") or {}).get("pow_bits"),
        "fixed_audit": (spark.get("fixed_audit") or {}).get("pow_bits"),
        "read": (spark.get("read") or {}).get("pow_bits"),
    }


def measurement_is_tied(row: dict[str, Any], fastest: dict[str, Any]) -> bool:
    fastest_seconds = float(fastest["measured_seconds"])
    row_seconds = float(row["measured_seconds"])
    relative = (
        (row_seconds - fastest_seconds) / fastest_seconds
        if fastest_seconds > 0.0
        else float("inf")
    )
    return relative <= PROVER_SELECTION_BAND_RELATIVE + 1e-12


def verifier_seconds(
    candidate: dict[str, Any], verifier_rows: dict[str, dict[str, Any]]
) -> float:
    measurement = verifier_rows.get(str(candidate["label"]))
    if measurement is not None and measurement.get("heldout_verify_seconds") is not None:
        return float(measurement["heldout_verify_seconds"])
    projected = candidate.get("verifier_projected_seconds")
    return float(projected) if projected is not None else float("inf")


def verifier_measurement_is_tied(
    candidate: dict[str, Any],
    fastest: dict[str, Any],
    verifier_rows: dict[str, dict[str, Any]],
) -> bool:
    candidate_seconds = verifier_seconds(candidate, verifier_rows)
    fastest_seconds = verifier_seconds(fastest, verifier_rows)
    if candidate_seconds == float("inf"):
        return fastest_seconds == float("inf")
    if candidate_seconds == fastest_seconds:
        return True
    relative = (
        (candidate_seconds - fastest_seconds) / fastest_seconds
        if fastest_seconds > 0.0
        else float("inf")
    )
    if relative <= VERIFIER_SELECTION_BAND_RELATIVE + 1e-12:
        return True
    candidate_measurement = verifier_rows.get(str(candidate["label"]))
    fastest_measurement = verifier_rows.get(str(fastest["label"]))
    return (
        candidate_measurement is not None
        and fastest_measurement is not None
        and intervals_overlap(
            verifier_median_interval(candidate_measurement),
            verifier_median_interval(fastest_measurement),
        )
    )


def measurement_map(measurements: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["label"]): float(row["measured_seconds"])
        for row in measurements.get("rows", [])
        if row.get("measured_seconds") is not None
    }


def median_interval(row: dict[str, Any]) -> tuple[float, float]:
    value = float(row["measured_seconds"])
    interval = row.get("heldout_median_ci_seconds")
    if not isinstance(interval, list) or len(interval) != 2:
        return value, value
    return float(interval[0]), float(interval[1])


def verifier_median_interval(row: dict[str, Any]) -> tuple[float, float]:
    value = float(row["heldout_verify_seconds"])
    interval = row.get("heldout_verify_median_ci_seconds")
    if not isinstance(interval, list) or len(interval) != 2:
        return value, value
    return float(interval[0]), float(interval[1])


def intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1]) + 1e-12


def measured_size(row: dict[str, Any]) -> int:
    return int(row.get("heldout_proof_size_median_bytes") or proof_size(row))


def proof_size(row: dict[str, Any]) -> int:
    return int(row.get("proof_size_bytes_estimate") or 0)


def ranking_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    return (
        float(row["projected_seconds"]),
        float(row.get("verifier_projected_seconds") or 0.0),
        proof_size(row),
        str(row["label"]),
    )


def relative_error(projected: float, measured: float) -> float:
    return abs(projected - measured) / measured if measured > 0.0 else float("inf")


def require_report_mode(report: dict[str, Any], mode: str, name: str) -> None:
    if report.get("proof_mode") != mode:
        raise SystemExit(f"{name} report must use proof_mode={mode}")


def require_matching_provenance(reports: dict[str, Any]) -> Any:
    values = [report.get("provenance") for _, report in named_reports(reports)]
    first = values[0]
    if first is None or any(value is None for value in values[1:]):
        raise SystemExit("component reports must carry non-null provenance")
    if any(value != first for value in values[1:]):
        raise SystemExit("component reports have different provenance")
    return first


def require_matching_measurement_context(
    measurements: dict[str, Any],
    provenance: Any,
    proof_mode: str,
    workload_identity: dict[str, Any],
) -> None:
    measurement_provenance = measurements.get("provenance")
    if measurement_provenance is None:
        raise SystemExit("heldout measurements must carry non-null provenance")
    if measurement_provenance != provenance:
        raise SystemExit("heldout measurement provenance differs from component reports")
    measurement_mode = measurements.get("proof_mode")
    if measurement_mode != proof_mode:
        raise SystemExit(
            f"heldout measurements use proof_mode={measurement_mode}, expected {proof_mode}"
        )
    if measurements.get("workload_identity") != workload_identity:
        raise SystemExit(
            "heldout measurement workload identity differs from the composed workload"
        )


def require_measurement_rows_match_candidates(
    measurements: dict[str, Any],
    candidates: list[dict[str, Any]],
    workload_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = measurements.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("heldout measurement rows must be an array")
    candidates_by_identity = {
        composed_row_identity(candidate): candidate for candidate in candidates
    }
    seen_labels = set()
    matched = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"heldout measurement row {index} is not an object")
        label = row.get("label")
        if not isinstance(label, str) or not label:
            raise SystemExit(f"heldout measurement row {index} is missing a label")
        if label in seen_labels:
            raise SystemExit(f"heldout measurements contain duplicate label {label}")
        seen_labels.add(label)
        candidate = candidates_by_identity.get(composed_row_identity(row))
        if candidate is None:
            raise SystemExit(
                f"heldout measurement row {index} does not match a composed candidate setup"
            )
        if workload_identity is not None:
            if row.get("workload_identity") != workload_identity:
                raise SystemExit(
                    f"heldout measurement row {index} has a different workload identity"
                )
            if row.get("constraint_work") != workload_identity["constraint_work"]:
                raise SystemExit(
                    f"heldout measurement row {index} has a different constraint count"
                )
            if row.get("heldout_case_label") != workload_identity["label"]:
                raise SystemExit(
                    f"heldout measurement row {index} has a different case label"
                )
        matched.append(candidate)
    return matched


def composed_row_identity(row: dict[str, Any]) -> str:
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


def read_json(path: str) -> dict[str, Any]:
    with Path(path).open() as file:
        return json.load(file)


def write_json(path: str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
