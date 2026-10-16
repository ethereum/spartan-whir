import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "poseidon_schedule_scorer.py"
spec = importlib.util.spec_from_file_location("poseidon_schedule_scorer", SCRIPT)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

ADD_HELDOUT_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "poseidon_schedule_add_heldout.py"
)
add_heldout_spec = importlib.util.spec_from_file_location(
    "poseidon_schedule_add_heldout", ADD_HELDOUT_SCRIPT
)
add_heldout = importlib.util.module_from_spec(add_heldout_spec)
add_heldout_spec.loader.exec_module(add_heldout)

TEST_PROVENANCE = {"code": {"test": "same"}, "build": {"test": "same"}}
TEST_WORKLOAD_IDENTITY = {
    "label": "case",
    "r1cs_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "constraint_work": 123,
}


def candidate(
    label,
    *,
    extension="octic",
    valid=True,
    max_pow=10,
    dft=0,
    merkle=0,
    merkle_path=0,
    row=0,
    sumcheck=0,
    pow_work=0,
    proof_size_bytes=None,
    legacy_proof_size=None,
    zk=False,
    zk_dft=None,
    zk_merkle=None,
    zk_merkle_path=None,
    zk_row=None,
    zk_sumcheck=None,
    zk_mask_queries=None,
    zk_ell=None,
    zk_mask_log_inv_rate=None,
    measured_seconds=None,
    measured_proof_size=None,
    median_ci=None,
    relative_difference=None,
    paired_ci=None,
    folding_family="constant",
    round_rate_offset=0,
):
    folding_schedule = {
        "constant": None,
        "constant_from_second_round": {
            "ConstantFromSecondRound": {"first": 8, "rest": 6}
        },
        "per_round": {"PerRound": [8, 6]},
    }[folding_family]
    out = {
        "label": label,
        "proof_mode": "full-zk" if zk else "no-zk",
        "extension": extension,
        "valid": valid,
        "rejection_reason": None if valid else "invalid",
        "security_bits_achieved": 128.0,
        "max_derived_pow_bits": max_pow,
        "round_log_inv_rate_offset": round_rate_offset,
        "constraint_work": 0,
        "dft_work": dft,
        "merkle_work": merkle,
        "merkle_path_work": merkle_path,
        "row_work": row,
        "sumcheck_work": sumcheck,
        "pow_work_units": pow_work,
        "proof_size_bytes_estimate": proof_size_bytes,
        "whir_params": {
            "pow_bits": max_pow,
            "folding_factor": 8,
            "folding_schedule": folding_schedule,
            "starting_log_inv_rate": 1,
            "rs_domain_initial_reduction_factor": 8,
            "round_log_inv_rates": [],
        },
        "setup_config": {
            "matrix_closing": "DirectSparse",
            "label": label,
            **({"ell_zk": zk_ell, "mask_log_inv_rate": zk_mask_log_inv_rate} if zk else {}),
        }
        if valid
        else None,
    }
    if zk_dft is not None:
        out["zk_dft_work"] = zk_dft
    if zk_merkle is not None:
        out["zk_merkle_work"] = zk_merkle
    if zk_merkle_path is not None:
        out["zk_merkle_path_work"] = zk_merkle_path
    if zk_row is not None:
        out["zk_row_work"] = zk_row
    if zk_sumcheck is not None:
        out["zk_sumcheck_work"] = zk_sumcheck
    if zk_mask_queries is not None:
        out["zk_mask_queries"] = zk_mask_queries
    if zk_ell is not None:
        out["zk_ell"] = zk_ell
    if zk_mask_log_inv_rate is not None:
        out["zk_mask_log_inv_rate"] = zk_mask_log_inv_rate
    if legacy_proof_size is not None:
        out["proof_size_score"] = legacy_proof_size
    if measured_seconds is not None:
        out["measured_seconds"] = measured_seconds
    if measured_proof_size is not None:
        out["heldout_proof_size_median_bytes"] = measured_proof_size
    if median_ci is not None:
        out["heldout_median_ci_seconds"] = list(median_ci)
    if relative_difference is not None:
        out["heldout_relative_median_difference"] = relative_difference
    if paired_ci is not None:
        out["heldout_paired_relative_median_ci"] = list(paired_ci)
    return out


class PoseidonScheduleScorerTests(unittest.TestCase):
    def score_dump(self, dump, calibration, **kwargs):
        dump.setdefault("provenance", TEST_PROVENANCE)
        measurements = kwargs.get("measurements")
        if measurements is not None:
            measurements.setdefault("provenance", TEST_PROVENANCE)
            dump.setdefault("workload_identity", TEST_WORKLOAD_IDENTITY)
            dump.setdefault("case_label", TEST_WORKLOAD_IDENTITY["label"])
            dump.setdefault(
                "constraint_work", TEST_WORKLOAD_IDENTITY["constraint_work"]
            )
            for row in dump.get("candidates", []):
                row.setdefault("workload_identity", dump["workload_identity"])
            measurements.setdefault("workload_identity", dump["workload_identity"])
            for row in measurements.get("rows", []):
                row.setdefault("workload_identity", measurements["workload_identity"])
        return scorer.score_dump(dump, calibration, **kwargs)

    def calibration(self, heldout=None):
        validation_rows = (
            [
                {
                    "label": "heldout",
                    "proof_mode": "no-zk",
                    "measured_seconds": 10.0,
                    "dft_work": 10,
                }
            ]
            if heldout is None
            else heldout
        )
        return {
            "provenance": TEST_PROVENANCE,
            "coefficients": {
                "fixed_overhead": 0.0,
                "dft": 1.0,
                "merkle": 1.0,
                "merkle_path": 1.0,
                "row_opening": 1.0,
                "sumcheck": 1.0,
                "pow": 1.0,
            },
            "verifier_coefficients": {
                "fixed_overhead": 0.0,
                "merkle_hash": 2.0,
                "leaf_field_element": 0.0,
                "row_field_element": 3.0,
                "extension_operation": {
                    "quartic": 4.0,
                    "quintic": 5.0,
                    "octic": 6.0,
                },
                "pow_check": 7.0,
            },
            "validation": {
                "max_relative_error": 0.10,
                "heldout": validation_rows,
            },
        }

    def per_extension_calibration(self):
        calibration = self.calibration()
        calibration["coefficients"]["sumcheck"] = {
            "quartic": 0.25,
            "quintic": 0.4,
            "octic": 1.0,
        }
        return calibration

    def test_rejects_over_cap_pow(self):
        dump = {"schema_version": 1, "proof_mode": "no-zk", "candidates": [candidate("bad", max_pow=23, dft=1)]}
        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)
        self.assertIsNone(report["selected"])
        self.assertIn("exceeds max", report["scores"][0]["rejection_reasons"][-1])

    def test_selects_lowest_projected_time(self):
        dump = {
            "schema_version": 1, "proof_mode": "no-zk",
            "candidates": [
                candidate("slow", dft=10),
                candidate("fast", dft=2),
            ],
        }
        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)
        self.assertEqual(report["selected"]["label"], "fast")
        self.assertIn("cost_breakdown", report["selected"])

    def test_cost_breakdown_includes_merkle_path(self):
        dump = {"schema_version": 1, "proof_mode": "no-zk", "candidates": [candidate("a", merkle_path=7)]}
        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)
        self.assertEqual(report["selected"]["cost_breakdown"]["merkle_path"], 7.0)

    def test_verifier_projection_uses_independent_work_counters(self):
        row = candidate("a", extension="quintic")
        row.update(
            {
                "verifier_merkle_hashes": 2,
                "verifier_row_field_elements": 3,
                "verifier_extension_operations": 4,
                "verifier_pow_checks": 5,
            }
        )
        dump = {"schema_version": 2, "proof_mode": "no-zk", "candidates": [row]}

        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)

        self.assertEqual(report["selected"]["verifier_projected_seconds"], 68.0)
        self.assertEqual(
            report["selected"]["verifier_cost_breakdown"]["merkle_hash"], 4.0
        )

    def test_legacy_calibration_defaults_merkle_path_to_zero(self):
        calibration = self.calibration()
        del calibration["coefficients"]["merkle_path"]
        dump = {"schema_version": 1, "proof_mode": "no-zk", "candidates": [candidate("a", merkle_path=7)]}
        report = self.score_dump(dump, calibration, max_pow_bits=22)
        self.assertEqual(report["selected"]["cost_breakdown"]["merkle_path"], 0.0)

    def test_uses_extension_specific_sumcheck_coefficients(self):
        dump = {
            "schema_version": 1, "proof_mode": "no-zk",
            "candidates": [
                candidate("octic", extension="octic", sumcheck=10),
                candidate("quartic", extension="quartic", sumcheck=10),
            ],
        }
        report = self.score_dump(dump, self.per_extension_calibration(), max_pow_bits=22)
        self.assertEqual(report["selected"]["label"], "quartic")
        self.assertEqual(report["selected"]["cost_breakdown"]["sumcheck"], 2.5)

    def test_model_tie_ignores_proof_size_and_uses_stable_label(self):
        dump = {
            "schema_version": 1, "proof_mode": "no-zk",
            "candidates": [
                candidate("large", dft=1, proof_size_bytes=20),
                candidate("small", dft=1, proof_size_bytes=5),
            ],
        }
        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)
        self.assertEqual(report["selected"]["label"], "large")
        self.assertEqual(report["scores"][0]["label"], "large")

    def test_model_tie_ignores_legacy_proof_size_score(self):
        dump = {
            "schema_version": 1, "proof_mode": "no-zk",
            "candidates": [
                candidate("large", dft=1, legacy_proof_size=20),
                candidate("small", dft=1, legacy_proof_size=5),
            ],
        }
        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)
        self.assertEqual(report["selected"]["label"], "large")

    def test_model_tie_breaks_by_pow_before_label(self):
        dump = {
            "schema_version": 1, "proof_mode": "no-zk",
            "candidates": [
                candidate("a", dft=1, proof_size_bytes=5, max_pow=4, pow_work=16),
                candidate("b", dft=1, proof_size_bytes=50, max_pow=0, pow_work=0),
            ],
        }
        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)
        self.assertEqual(report["selected"]["label"], "b")

    def test_component_report_cap_preserves_three_axis_frontier(self):
        dump = {
            "schema_version": 1,
            "proof_mode": "no-zk",
            "component_security_override_bits": 120,
            "candidates": [
                candidate("fast-large", dft=1, proof_size_bytes=30),
                candidate("verify", dft=2, proof_size_bytes=20),
                candidate("small", dft=3, proof_size_bytes=10),
                candidate("dominated", dft=4, proof_size_bytes=40),
            ],
        }
        dump["candidates"][0]["verifier_merkle_hashes"] = 3
        dump["candidates"][1]["verifier_merkle_hashes"] = 1
        dump["candidates"][2]["verifier_merkle_hashes"] = 2
        dump["candidates"][3]["verifier_merkle_hashes"] = 4

        report = self.score_dump(
            dump,
            self.calibration(),
            max_pow_bits=22,
            max_report_rows=3,
        )

        self.assertEqual(
            {row["label"] for row in report["scores"]},
            {"fast-large", "verify", "small"},
        )
        self.assertEqual(report["candidate_retention"]["pareto_rows"], 3)

    def test_apply_case_metrics_updates_candidates(self):
        dump = {"schema_version": 1, "proof_mode": "no-zk", "candidates": [candidate("a")]}
        scorer.apply_case_metrics(
            dump,
            constraint_work=123,
            case_label="case",
            workload_identity=TEST_WORKLOAD_IDENTITY,
        )
        self.assertEqual(dump["constraint_work"], 123)
        self.assertEqual(dump["case_label"], "case")
        self.assertEqual(dump["workload_identity"], TEST_WORKLOAD_IDENTITY)
        self.assertEqual(dump["candidates"][0]["constraint_work"], 123)
        self.assertEqual(dump["candidates"][0]["case_label"], "case")
        self.assertEqual(
            dump["candidates"][0]["workload_identity"], TEST_WORKLOAD_IDENTITY
        )

    def test_build_workload_identity_hashes_r1cs_and_requires_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            r1cs = Path(directory) / "case.r1cs"
            r1cs.write_bytes(b"abc")

            identity = scorer.build_workload_identity(str(r1cs), 123, "case")

            self.assertEqual(identity, TEST_WORKLOAD_IDENTITY)
            with self.assertRaisesRegex(SystemExit, "positive --constraint-work"):
                scorer.build_workload_identity(str(r1cs), None, "case")
            with self.assertRaisesRegex(SystemExit, "requires --case-label"):
                scorer.build_workload_identity(str(r1cs), 123, None)

    def test_score_report_propagates_workload_metadata_to_ranked_rows(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "num_outer_rounds": 17,
            "candidates": [candidate("a", dft=1)],
        }
        scorer.apply_case_metrics(dump, 123, "case", TEST_WORKLOAD_IDENTITY)

        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)

        self.assertEqual(report["num_outer_rounds"], 17)
        self.assertEqual(report["constraint_work"], 123)
        self.assertEqual(report["case_label"], "case")
        self.assertEqual(report["workload_identity"], TEST_WORKLOAD_IDENTITY)
        self.assertEqual(report["selected"]["workload_identity"], TEST_WORKLOAD_IDENTITY)
        self.assertTrue(report["measurement_shortlist"])
        for row in [*report["scores"], *report["measurement_shortlist"]]:
            self.assertEqual(row["workload_identity"], TEST_WORKLOAD_IDENTITY)

    def test_measurements_require_matching_workload_identity(self):
        dump = {
            "schema_version": 5,
            "provenance": TEST_PROVENANCE,
            "proof_mode": "no-zk",
            "candidates": [candidate("a", dft=1)],
        }
        measurements = {
            "provenance": TEST_PROVENANCE,
            "proof_mode": "no-zk",
            "rows": [candidate("a", dft=1, measured_seconds=1.0)],
        }

        with self.assertRaisesRegex(SystemExit, "missing workload_identity"):
            scorer.score_dump(
                dump,
                self.calibration(),
                max_pow_bits=22,
                measurements=measurements,
            )

    def test_rejects_candidate_row_with_foreign_workload_identity(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "constraint_work": 123,
            "case_label": "case",
            "workload_identity": TEST_WORKLOAD_IDENTITY,
            "candidates": [candidate("a", dft=1)],
        }
        dump["candidates"][0]["workload_identity"] = {
            **TEST_WORKLOAD_IDENTITY,
            "label": "other",
        }

        with self.assertRaisesRegex(SystemExit, "row 0 workload_identity differs"):
            self.score_dump(dump, self.calibration(), max_pow_bits=22)

    def test_marks_untrusted_when_validation_fails(self):
        heldout = [
            {
                "label": "bad",
                "proof_mode": "no-zk",
                "measured_seconds": 10.0,
                "dft_work": 1,
            }
        ]
        dump = {"schema_version": 1, "proof_mode": "no-zk", "candidates": [candidate("ok", dft=1)]}
        report = self.score_dump(dump, self.calibration(heldout), max_pow_bits=22)
        self.assertFalse(report["model_validation"]["trusted"])
        self.assertEqual(report["selected"]["label"], "ok")

    def test_validation_uses_zk_metrics_for_zk_heldout(self):
        heldout = [candidate("zk", dft=1, zk=True, zk_dft=10)]
        heldout[0]["measured_seconds"] = 10.0
        dump = {
            "schema_version": 1,
            "proof_mode": "full-zk",
            "candidates": [candidate("ok", dft=1, zk=True)],
        }
        report = self.score_dump(dump, self.calibration(heldout), max_pow_bits=22)
        self.assertTrue(report["model_validation"]["trusted"])
        self.assertEqual(report["model_validation"]["heldout"][0]["projected_seconds"], 10.0)

    def test_validation_uses_only_rows_for_the_scored_proof_mode(self):
        heldout = [candidate("zk", dft=1, zk=True, zk_dft=10)]
        heldout[0]["measured_seconds"] = 10.0
        dump = {
            "schema_version": 1,
            "proof_mode": "no-zk",
            "candidates": [candidate("plain", dft=1)],
        }

        report = self.score_dump(dump, self.calibration(heldout), max_pow_bits=22)

        self.assertFalse(report["model_validation"]["trusted"])
        self.assertEqual(report["model_validation"]["source_heldout_rows"], 1)
        self.assertEqual(report["model_validation"]["mode_heldout_rows"], 0)
        self.assertEqual(report["model_validation"]["heldout"], [])

    def test_validation_rejects_rows_without_a_proof_mode(self):
        heldout = [{"label": "ambiguous", "measured_seconds": 1.0, "dft_work": 1}]
        dump = {
            "schema_version": 1,
            "proof_mode": "no-zk",
            "candidates": [candidate("plain", dft=1)],
        }

        with self.assertRaisesRegex(SystemExit, "must declare proof_mode"):
            self.score_dump(dump, self.calibration(heldout), max_pow_bits=22)

    def test_validation_reports_ordering_diagnostic(self):
        heldout = [
            {"label": "a", "proof_mode": "no-zk", "measured_seconds": 1.0, "dft_work": 1},
            {"label": "b", "proof_mode": "no-zk", "measured_seconds": 3.0, "dft_work": 2},
            {"label": "c", "proof_mode": "no-zk", "measured_seconds": 2.0, "dft_work": 3},
        ]
        dump = {"schema_version": 1, "proof_mode": "no-zk", "candidates": [candidate("ok", dft=1)]}
        report = self.score_dump(dump, self.calibration(heldout), max_pow_bits=22)
        diagnostic = report["model_validation"]["ordering_diagnostic"]
        self.assertEqual(diagnostic["comparable_pairs"], 3)
        self.assertIn("kendall_tau", diagnostic)
        self.assertGreater(diagnostic["model_resolution_seconds"], 0)

    def test_measurement_shortlist_keeps_fastest_equivalent_ell(self):
        common = dict(
            dft=1,
            zk=True,
            zk_dft=1,
            zk_merkle=2,
            zk_merkle_path=3,
            zk_row=4,
            zk_sumcheck=5,
            zk_mask_queries=90,
            zk_mask_log_inv_rate=3,
        )
        dump = {
            "schema_version": 1, "proof_mode": "full-zk",
            "candidates": [
                candidate("same", proof_size_bytes=20, zk_ell=8, **common),
                candidate("same", proof_size_bytes=10, zk_ell=3, **common),
                candidate("outside", dft=100, proof_size_bytes=1, zk=True),
            ],
        }
        dump["candidates"][1]["zk_dft_work"] = 2
        report = self.score_dump(
            dump,
            self.calibration(),
            max_pow_bits=22,
            measurement_shortlist_margin_seconds=1.1,
        )
        labels = [row["label"] for row in report["measurement_shortlist"]]
        self.assertEqual(labels, ["same"])
        self.assertEqual(report["measurement_shortlist"][0]["zk_ell"], 8)
        self.assertEqual(report["measurement_shortlist_meta"]["pre_dedup_count"], 2)
        self.assertEqual(report["measurement_shortlist_meta"]["dedup_count"], 1)

    def test_reference_label_adds_accepted_row_outside_model_margin(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [
                candidate("model", dft=1),
                candidate("reference", dft=100),
            ],
        }

        report = self.score_dump(
            dump,
            self.calibration(),
            max_pow_bits=22,
            measurement_shortlist_margin_seconds=0.0,
            reference_labels=["reference"],
        )

        self.assertEqual(
            [row["label"] for row in report["measurement_shortlist"]],
            ["model", "reference"],
        )
        self.assertEqual(
            report["measurement_shortlist_meta"]["reference_labels"],
            ["reference"],
        )
        self.assertEqual(
            report["measurement_shortlist_meta"]["reference_rows_added"], 1
        )

    def test_reference_label_must_name_an_accepted_candidate(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [candidate("rejected", valid=False, dft=1)],
        }

        with self.assertRaisesRegex(
            SystemExit, "reference label not found among accepted candidates"
        ):
            self.score_dump(
                dump,
                self.calibration(),
                max_pow_bits=22,
                reference_labels=["rejected"],
            )

    def test_missing_reference_label_is_rejected(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [candidate("accepted", dft=1)],
        }

        with self.assertRaisesRegex(
            SystemExit, "reference label not found among accepted candidates"
        ):
            self.score_dump(
                dump,
                self.calibration(),
                max_pow_bits=22,
                reference_labels=["missing"],
            )

    def test_measurement_shortlist_covers_extension_folding_and_rate_groups(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [
                candidate("quintic-constant-derived", extension="quintic", dft=1),
                candidate(
                    "quintic-cfsr-derived",
                    extension="quintic",
                    dft=50,
                    folding_family="constant_from_second_round",
                ),
                candidate(
                    "quintic-cfsr-offset",
                    extension="quintic",
                    dft=60,
                    folding_family="constant_from_second_round",
                    round_rate_offset=1,
                ),
                candidate("quartic-constant-derived", extension="quartic", dft=70),
                candidate("octic-constant-derived", extension="octic", dft=80),
            ],
        }

        report = self.score_dump(
            dump,
            self.calibration(),
            max_pow_bits=22,
            measurement_shortlist_margin_seconds=0.0,
        )

        labels = {row["label"] for row in report["measurement_shortlist"]}
        expected = {
            "quintic-constant-derived",
            "quintic-cfsr-derived",
            "quintic-cfsr-offset",
            "quartic-constant-derived",
            "octic-constant-derived",
        }
        self.assertEqual(labels, expected)
        self.assertEqual(
            set(report["measurement_shortlist_meta"]["required_labels"]),
            expected,
        )
        self.assertEqual(
            report["measurement_shortlist_meta"]["coverage_group_count"], 5
        )

    def test_selected_measured_is_separate_from_model_selected(self):
        dump = {
            "schema_version": 1, "proof_mode": "no-zk",
            "candidates": [candidate("model", dft=1), candidate("measured", dft=2)],
        }
        measurements = {
            "proof_mode": "no-zk",
            "source_report": "report.json",
            "rows": [
                {**candidate("model", dft=1), "measured_seconds": 3.0},
                {
                    **candidate("measured", dft=2),
                    "measured_seconds": 1.0,
                    "heldout_median_ci_seconds": [0.9, 1.1],
                },
            ],
        }
        report = self.score_dump(
            dump,
            self.calibration(),
            max_pow_bits=22,
            measurements=measurements,
        )
        self.assertEqual(report["selected"]["label"], "model")
        self.assertEqual(report["selected_measured"]["label"], "measured")
        self.assertEqual(
            report["measurement_summary"]["selection"],
            "one_percent_or_overlapping_median_ci_then_pow_then_label",
        )

    def test_output_config_prefers_measured_selection(self):
        report = {
            "selected": candidate("model"),
            "selected_measured": candidate("measured"),
        }

        self.assertEqual(
            scorer.selected_setup_config(report),
            candidate("measured")["setup_config"],
        )

    def test_output_config_falls_back_to_model_selection(self):
        report = {"selected": candidate("model"), "selected_measured": None}

        self.assertEqual(
            scorer.selected_setup_config(report),
            candidate("model")["setup_config"],
        )

    def test_significant_sub_one_percent_improvement_uses_lower_pow(self):
        measurements = {
            "proof_mode": "full-zk",
            "rows": [
                candidate(
                    "fast",
                    zk=True,
                    measured_seconds=0.994,
                    measured_proof_size=200,
                    relative_difference=0.0,
                    paired_ci=(0.0, 0.0),
                ),
                candidate(
                    "small",
                    zk=True,
                    measured_seconds=1.0,
                    measured_proof_size=100,
                    max_pow=4,
                    relative_difference=0.006,
                    paired_ci=(0.004, 0.008),
                ),
            ],
        }
        selected, _ = scorer.measured_selection(measurements)
        self.assertEqual(selected["label"], "small")

    def test_more_than_one_percent_directional_improvement_wins(self):
        measurements = {
            "proof_mode": "full-zk",
            "rows": [
                candidate(
                    "fast",
                    zk=True,
                    measured_seconds=0.98,
                    measured_proof_size=200,
                    relative_difference=0.0,
                    paired_ci=(0.0, 0.0),
                ),
                candidate(
                    "small",
                    zk=True,
                    measured_seconds=1.0,
                    measured_proof_size=100,
                    relative_difference=0.0204,
                    paired_ci=(0.012, 0.03),
                ),
            ],
        }
        selected, _ = scorer.measured_selection(measurements)
        self.assertEqual(selected["label"], "fast")

    def test_more_than_one_percent_with_overlapping_median_ci_uses_lower_pow(self):
        measurements = {
            "proof_mode": "full-zk",
            "rows": [
                candidate(
                    "fast",
                    zk=True,
                    measured_seconds=0.98,
                    measured_proof_size=200,
                    median_ci=(0.95, 1.01),
                    relative_difference=0.0,
                    paired_ci=(0.0, 0.0),
                ),
                candidate(
                    "small",
                    zk=True,
                    measured_seconds=1.0,
                    measured_proof_size=100,
                    median_ci=(0.97, 1.03),
                    max_pow=4,
                    relative_difference=0.0204,
                    paired_ci=(-0.002, 0.03),
                ),
            ],
        }
        selected, _ = scorer.measured_selection(measurements)
        self.assertEqual(selected["label"], "small")

    def test_no_zk_nonoverlapping_two_x_slowdown_cannot_win_on_pow(self):
        measurements = {
            "proof_mode": "no-zk",
            "rows": [
                candidate(
                    "fast",
                    measured_seconds=1.0,
                    median_ci=(0.99, 1.01),
                    max_pow=10,
                ),
                candidate(
                    "slow-low-pow",
                    measured_seconds=2.0,
                    median_ci=(1.99, 2.01),
                    max_pow=0,
                    relative_difference=0.0,
                ),
            ],
        }

        selected, _ = scorer.measured_selection(measurements)

        self.assertEqual(selected["label"], "fast")

    def test_exactly_one_percent_slower_is_a_time_tie(self):
        measurements = {
            "proof_mode": "no-zk",
            "rows": [
                candidate(
                    "fast-high-pow",
                    measured_seconds=1.0,
                    median_ci=(0.999, 1.001),
                    max_pow=10,
                ),
                candidate(
                    "one-percent-low-pow",
                    measured_seconds=1.01,
                    median_ci=(1.009, 1.011),
                    max_pow=0,
                ),
            ],
        }

        selected, _ = scorer.measured_selection(measurements)

        self.assertEqual(selected["label"], "one-percent-low-pow")

    def test_measured_full_tie_ignores_proof_size_and_uses_stable_label(self):
        measurements = {
            "proof_mode": "full-zk",
            "rows": [
                candidate(
                    "large",
                    zk=True,
                    measured_seconds=1.0,
                    measured_proof_size=200,
                    relative_difference=0.0,
                    paired_ci=(0.0, 0.0),
                ),
                candidate(
                    "small",
                    zk=True,
                    measured_seconds=1.0,
                    measured_proof_size=100,
                    relative_difference=0.0,
                    paired_ci=(-0.001, 0.001),
                ),
            ],
        }
        selected, _ = scorer.measured_selection(measurements)
        self.assertEqual(selected["label"], "large")

    def test_recalibrate_updates_extension_specific_sumcheck(self):
        calibration = self.per_extension_calibration()
        row = candidate("heldout", extension="quartic", sumcheck=10)
        row["measured_seconds"] = 10.0
        add_heldout.recalibrate(calibration, [row], prior_weight=0.01)
        self.assertIn("quartic", calibration["coefficients"]["sumcheck"])
        self.assertNotEqual(calibration["coefficients"]["sumcheck"]["quartic"], 0.25)

    def test_recalibrate_uses_zk_metrics_for_zk_heldout(self):
        calibration = self.calibration()
        row = candidate("zk", dft=1, zk=True, zk_dft=10)
        row["measured_seconds"] = 10.0
        add_heldout.recalibrate(calibration, [row], prior_weight=0.0)
        self.assertAlmostEqual(calibration["coefficients"]["dft"], 1.0)

    def test_recalibrate_can_learn_spartan_from_zero_coefficient(self):
        calibration = self.calibration()
        calibration["coefficients"]["spartan"] = 0.0
        row = candidate("spartan")
        row["constraint_work"] = 100
        row["measured_seconds"] = 1.0

        add_heldout.recalibrate(calibration, [row], prior_weight=0.0)

        self.assertGreater(calibration["coefficients"]["spartan"], 0.0)

    def test_zk_candidate_generation_sweeps_default_zk_knobs(self):
        calls = []

        def fake_generate_candidate_dump(
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
        ):
            calls.append((zk_ell, zk_mask_log_inv_rate))
            self.assertEqual(features, "parallel")
            self.assertEqual(security_bits, 116)
            self.assertIsNone(merkle_security_bits)
            self.assertIsNone(component_security_bits)
            self.assertIsNone(component_merkle_security_bits)
            self.assertEqual(extension, "all")
            self.assertEqual(round_log_inv_rate_offset_max, 0)
            return {
                "schema_version": 2,
                "provenance": TEST_PROVENANCE,
                "proof_mode": proof_mode,
                "num_outer_rounds": num_outer_rounds,
                "zk_ell": zk_ell,
                "zk_mask_log_inv_rate": zk_mask_log_inv_rate,
                "candidates": [{"label": f"ell{zk_ell}_mask{zk_mask_log_inv_rate}"}],
            }

        old_generate_candidate_dump = scorer.generate_candidate_dump
        try:
            scorer.generate_candidate_dump = fake_generate_candidate_dump
            dump = scorer.generate_candidates(
                "cargo",
                20,
                22,
                "koalabear",
                "full-zk",
                1 << 19,
                None,
                None,
            )
        finally:
            scorer.generate_candidate_dump = old_generate_candidate_dump

        expected = [
            (ell, mask)
            for ell in scorer.DEFAULT_ZK_ELL_SWEEP
            for mask in scorer.DEFAULT_ZK_MASK_LOG_INV_RATE_SWEEP
        ]
        self.assertEqual(calls, expected)
        self.assertEqual(dump["zk_ell_sweep"], scorer.DEFAULT_ZK_ELL_SWEEP)
        self.assertEqual(
            dump["zk_mask_log_inv_rate_sweep"],
            scorer.DEFAULT_ZK_MASK_LOG_INV_RATE_SWEEP,
        )
        self.assertEqual(dump["num_outer_rounds"], 19)

    def test_candidate_command_propagates_extension_and_round_rate_offset(self):
        calls = []

        def fake_check_output(cmd, text):
            calls.append(cmd)
            self.assertTrue(text)
            return '{"schema_version": 5, "proof_mode": "no-zk", "candidates": []}'

        old_check_output = scorer.subprocess.check_output
        try:
            scorer.subprocess.check_output = fake_check_output
            scorer.generate_candidate_dump(
                "cargo",
                20,
                22,
                "koalabear",
                "no-zk",
                19,
                None,
                None,
                "parallel",
                116,
                116,
                120,
                123,
                extension="quintic",
                round_log_inv_rate_offset_max=2,
            )
        finally:
            scorer.subprocess.check_output = old_check_output

        cmd = calls[0]
        self.assertEqual(cmd[cmd.index("--extension") + 1], "quintic")
        self.assertEqual(
            cmd[cmd.index("--round-log-inv-rate-offset-max") + 1], "2"
        )

    def test_extension_filter_retains_only_requested_rows(self):
        dump = {
            "schema_version": 5,
            "extension_filter": "all",
            "candidates": [
                candidate("q", extension="quartic"),
                candidate("k", extension="quintic"),
                candidate("o", extension="octic"),
            ],
        }

        filtered = scorer.filter_candidate_extensions(dump, "quintic")

        self.assertEqual(filtered["extension_filter"], "quintic")
        self.assertEqual(
            [row["extension"] for row in filtered["candidates"]], ["quintic"]
        )

    def test_component_search_accepts_rows_without_standalone_setup_config(self):
        row = candidate("component", dft=1)
        row["setup_config"] = None
        dump = {
            "schema_version": 4,
            "proof_mode": "no-zk",
            "component_security_override_bits": 120,
            "component_merkle_security_override_bits": 123,
            "candidates": [row],
        }

        report = self.score_dump(dump, self.calibration(), max_pow_bits=22)

        self.assertEqual(report["selected"]["label"], "component")

    def test_rejects_mismatched_code_provenance(self):
        dump = {
            "schema_version": 3,
            "proof_mode": "no-zk",
            "provenance": {"code": {"spartan_whir": {"head": "a"}}},
            "candidates": [candidate("a", dft=1)],
        }
        calibration = self.calibration()
        calibration["provenance"] = {"code": {"spartan_whir": {"head": "b"}}}
        with self.assertRaises(SystemExit):
            scorer.score_dump(dump, calibration, max_pow_bits=22)

    def test_rejects_mismatched_build_provenance(self):
        provenance = {
            "code": {"spartan_whir": {"head": "same"}},
            "build": {"profile": "release", "features": "parallel"},
        }
        dump = {
            "schema_version": 3,
            "proof_mode": "no-zk",
            "provenance": provenance,
            "candidates": [candidate("a", dft=1)],
        }
        calibration = self.calibration()
        calibration["provenance"] = {
            **provenance,
            "build": {"profile": "debug", "features": "parallel"},
        }
        with self.assertRaises(SystemExit):
            scorer.score_dump(dump, calibration, max_pow_bits=22)

    def test_rejects_artifacts_when_both_provenances_are_missing(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [candidate("a", dft=1)],
        }
        calibration = self.calibration()
        del calibration["provenance"]

        with self.assertRaisesRegex(SystemExit, "both artifacts are missing provenance"):
            scorer.score_dump(dump, calibration, max_pow_bits=22)

    def test_rejects_candidate_row_with_mismatched_proof_mode(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [candidate("zk-row", dft=1, zk=True)],
        }

        with self.assertRaisesRegex(SystemExit, "candidate row 0 must use proof_mode=no-zk"):
            self.score_dump(dump, self.calibration(), max_pow_bits=22)

    def test_rejects_heldout_row_with_mismatched_proof_mode(self):
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [candidate("a", dft=1)],
        }
        measurements = {
            "proof_mode": "no-zk",
            "rows": [candidate("a", dft=1, zk=True)],
        }

        with self.assertRaisesRegex(SystemExit, "heldout row 0 must use proof_mode=no-zk"):
            self.score_dump(
                dump,
                self.calibration(),
                max_pow_bits=22,
                measurements=measurements,
            )

    def test_rejects_heldout_row_with_foreign_setup_config(self):
        accepted = candidate("a", dft=1)
        foreign = candidate("a", dft=1)
        foreign["setup_config"] = {
            **foreign["setup_config"],
            "unexpected": "different",
        }
        dump = {
            "schema_version": 5,
            "proof_mode": "no-zk",
            "candidates": [accepted],
        }
        measurements = {
            "proof_mode": "no-zk",
            "rows": [foreign],
        }

        with self.assertRaisesRegex(SystemExit, "does not match an accepted candidate setup"):
            self.score_dump(
                dump,
                self.calibration(),
                max_pow_bits=22,
                measurements=measurements,
            )

    def test_recalibration_uses_fresh_model_resolution_floor(self):
        calibration = self.calibration(heldout=[])
        row = candidate("heldout", dft=10)
        row["measured_seconds"] = 10.0

        add_heldout.refresh_model_resolution(calibration, [row])

        self.assertEqual(calibration["validation"]["model_resolution_relative"], 0.01)
        self.assertEqual(calibration["validation"]["max_relative_error"], 0.01)

    def test_add_heldout_rejects_mismatched_code_provenance(self):
        calibration = {"provenance": {"code": {"head": "a"}}}
        heldout = {"provenance": {"code": {"head": "b"}}}

        with self.assertRaises(SystemExit):
            add_heldout.require_matching_code_provenance(calibration, heldout)


if __name__ == "__main__":
    unittest.main()
