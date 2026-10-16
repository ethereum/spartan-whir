import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "poseidon_spark_schedule_scorer.py"
SPEC = importlib.util.spec_from_file_location("poseidon_spark_schedule_scorer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ADD_HELDOUT_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "poseidon_schedule_add_heldout.py"
)
ADD_HELDOUT_SPEC = importlib.util.spec_from_file_location(
    "poseidon_schedule_add_heldout", ADD_HELDOUT_SCRIPT
)
ADD_HELDOUT_MODULE = importlib.util.module_from_spec(ADD_HELDOUT_SPEC)
assert ADD_HELDOUT_SPEC.loader is not None
ADD_HELDOUT_SPEC.loader.exec_module(ADD_HELDOUT_MODULE)

WORKLOAD_IDENTITY = {
    "label": "test-circuit",
    "r1cs_sha256": "00" * 32,
    "constraint_work": 123,
}
TEST_COEFFICIENTS = {"fixture": 1.0}


def candidate(
    label,
    projected,
    proof_bytes,
    *,
    extension="quintic",
    starting_log_inv_rate=1,
    pow_bits=0,
    folding_factor=6,
    round_log_inv_rates=None,
    zk_ell=None,
    zk_mask_log_inv_rate=None,
):
    row = {
        "label": label,
        "extension": extension,
        "proof_mode": "full-zk" if zk_ell is not None else "no-zk",
        "valid": True,
        "accepted_for_ranking": True,
        "base_field": "koalabear",
        "whir_component_security_bits": 120,
        "merkle_component_security_bits": 123,
        "security_bits_achieved": 120.0,
        "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
        "proof_size_bytes_estimate": proof_bytes,
        "whir_params": {
            "pow_bits": pow_bits,
            "folding_factor": folding_factor,
            "starting_log_inv_rate": starting_log_inv_rate,
            "rs_domain_initial_reduction_factor": 1,
            "folding_schedule": {"Constant": folding_factor},
            "round_log_inv_rates": list(round_log_inv_rates or []),
        },
        "cost_breakdown": {
            "fixed_overhead": 10.0,
            "spartan": 20.0,
            "dft": projected,
            "merkle": projected,
            "merkle_path": projected,
            "row_opening": projected,
            "sumcheck": projected,
            "pow": projected,
        },
        "dft_work": 1,
        "merkle_work": 1,
    }
    if zk_ell is not None:
        row["zk_ell"] = zk_ell
    if zk_mask_log_inv_rate is not None:
        row["zk_mask_log_inv_rate"] = zk_mask_log_inv_rate
    return row


def report(mode, variables, rows):
    for row in rows:
        row["proof_mode"] = mode
        row["workload_identity"] = copy.deepcopy(WORKLOAD_IDENTITY)
    return {
        "provenance": {"git_commit": "test"},
        "coefficients": copy.deepcopy(TEST_COEFFICIENTS),
        "proof_mode": mode,
        "num_variables": variables,
        "target_security_bits": 116,
        "target_merkle_security_bits": 116,
        "component_security_override_bits": 120,
        "component_merkle_security_override_bits": 123,
        "constraint_work": WORKLOAD_IDENTITY["constraint_work"],
        "case_label": WORKLOAD_IDENTITY["label"],
        "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
        "selected": copy.deepcopy(rows[0]) if rows else None,
        "scores": rows,
    }


class SparkScheduleScorerTests(unittest.TestCase):
    def reports(self):
        return {
            "witness": report(
                "full-zk",
                20,
                [
                    candidate(
                        "w0",
                        1.0,
                        100,
                        zk_ell=3,
                        zk_mask_log_inv_rate=3,
                    )
                ],
            ),
            "fixed_value": report("no-zk", 25, [candidate("fv0", 2.0, 200)]),
            "fixed_audit": report("no-zk", 22, [candidate("fa0", 3.0, 300)]),
            "read": [
                report(
                    "no-zk",
                    25,
                    [candidate("r25", 4.0, 400, round_log_inv_rates=[6, 11, 16])],
                ),
                report(
                    "no-zk",
                    23,
                    [candidate("r23", 5.0, 500, round_log_inv_rates=[6, 11])],
                ),
            ],
        }

    def compose(
        self,
        reports=None,
        *,
        extension="quintic",
        proof_mode="full-zk",
        fixed_audit_embedded=True,
        reference_labels=None,
        measurements=None,
        measurement_rows=2,
    ):
        return MODULE.compose_report(
            reports or self.reports(),
            116,
            116,
            3 if proof_mode == "full-zk" else None,
            3 if proof_mode == "full-zk" else None,
            measurement_rows,
            10,
            2,
            measurements,
            {"fixed_value": 26, "fixed_audit": 24},
            extension,
            reference_labels,
            proof_mode=proof_mode,
            fixed_audit_embedded=fixed_audit_embedded,
            workload_identity=copy.deepcopy(WORKLOAD_IDENTITY),
        )

    def test_composes_quintic_two_read_groups_and_sums_estimates(self):
        result = self.compose()
        row = result["selected"]

        self.assertEqual(result["schema_version"], 4)
        self.assertEqual(result["extension"], "quintic")
        self.assertEqual(result["component_num_variables"]["read"], [25, 23])
        self.assertEqual(row["component_labels"]["read"], ["r25", "r23"])
        self.assertEqual(row["component_scores"]["witness"], 6.0)
        self.assertEqual(row["component_scores"]["fixed_value"], 8.0)
        self.assertEqual(row["component_scores"]["fixed_audit"], 0.0)
        self.assertEqual(row["component_scores"]["read"], 54.0)
        self.assertEqual(row["projected_schedule_seconds"], 68.0)
        self.assertEqual(row["proof_size_bytes_estimate"], 1200)
        self.assertEqual(
            result["candidate_retention"]["recursive_verifier_metric"],
            "calibrated native verifier wall time",
        )
        self.assertEqual(
            result["candidate_retention"]["recursive_verifier_used_for_ranking"],
            "tie_break_only",
        )

    def test_component_frontier_survives_top_per_component_limit(self):
        reports = self.reports()
        reports["witness"]["scores"] = [
            candidate("fast-large", 1.0, 400, pow_bits=1, zk_ell=3, zk_mask_log_inv_rate=3),
            candidate("middle", 2.0, 300, pow_bits=2, zk_ell=3, zk_mask_log_inv_rate=3),
            candidate("slow-small", 3.0, 200, pow_bits=3, zk_ell=3, zk_mask_log_inv_rate=3),
            candidate("slow-large", 4.0, 350, pow_bits=4, zk_ell=3, zk_mask_log_inv_rate=3),
        ]

        result = self.compose(reports, measurement_rows=1)
        labels = {row["component_labels"]["witness"] for row in result["scores"]}

        self.assertEqual(labels, {"fast-large", "middle", "slow-small"})
        self.assertEqual(
            result["candidate_retention"]["components"]["witness"]["pareto_rows"],
            3,
        )
        self.assertEqual(
            result["candidate_retention"]["top_per_component_scope"],
            "measurement_shortlist",
        )

    def test_component_frontier_keeps_equal_objective_configurations(self):
        reports = self.reports()
        reports["witness"]["scores"] = [
            candidate("same-a", 1.0, 100, pow_bits=1, zk_ell=3, zk_mask_log_inv_rate=3),
            candidate("same-b", 1.0, 100, pow_bits=2, zk_ell=3, zk_mask_log_inv_rate=3),
        ]

        result = self.compose(reports, measurement_rows=1)

        self.assertEqual(
            {row["component_labels"]["witness"] for row in result["scores"]},
            {"same-a", "same-b"},
        )

    def test_report_cap_must_fit_the_composed_frontier(self):
        reports = self.reports()
        reports["witness"]["scores"] = [
            candidate(
                f"w{index}",
                float(index + 1),
                400 - index * 100,
                pow_bits=index + 1,
                zk_ell=3,
                zk_mask_log_inv_rate=3,
            )
            for index in range(4)
        ]

        with self.assertRaisesRegex(SystemExit, "exceed --max-report-rows 2"):
            MODULE.compose_report(
                reports,
                116,
                116,
                3,
                3,
                1,
                2,
                1,
                None,
                {"fixed_value": 26, "fixed_audit": 24},
                "quintic",
                proof_mode="full-zk",
                fixed_audit_embedded=True,
                workload_identity=copy.deepcopy(WORKLOAD_IDENTITY),
            )

    def test_different_source_derived_rates_are_normalized_to_empty(self):
        result = self.compose()
        read_params = result["selected"]["setup_config"]["spark_whir_params"]["read"]

        self.assertEqual(read_params["round_log_inv_rates"], [])
        self.assertEqual(read_params["folding_schedule"], {"Constant": 6})

    def test_constant_read_schedule_allows_partial_final_fold(self):
        reports = self.reports()
        for read_report in reports["read"]:
            params = read_report["scores"][0]["whir_params"]
            params["folding_factor"] = 8
            params["folding_schedule"] = {"Constant": 8}
            params["round_log_inv_rates"] = []

        result = self.compose(reports)

        read_params = result["selected"]["setup_config"]["spark_whir_params"]["read"]
        self.assertEqual(read_params["folding_schedule"], {"Constant": 8})
        self.assertEqual(read_params["round_log_inv_rates"], [])

    def test_setup_config_has_exact_rust_json_shape(self):
        reports = self.reports()
        result = self.compose(reports)
        expected_read = copy.deepcopy(reports["read"][0]["scores"][0]["whir_params"])
        expected_read["round_log_inv_rates"] = []

        self.assertEqual(
            result["selected"]["setup_config"],
            {
                "matrix_closing": "Spark",
                "security": {
                    "security_level_bits": 116,
                    "merkle_security_bits": 116,
                    "soundness_assumption": "JohnsonBound",
                },
                "whir_params": reports["witness"]["scores"][0]["whir_params"],
                "spark_whir_params": {
                    "fixed_value": reports["fixed_value"]["scores"][0]["whir_params"],
                    "fixed_audit": reports["fixed_audit"]["scores"][0]["whir_params"],
                    "read": expected_read,
                },
                "ell_zk": 3,
                "mask_log_inv_rate": 3,
            },
        )

    def test_rejects_read_reports_without_a_complete_shared_tuple(self):
        reports = self.reports()
        reports["read"][1]["scores"][0]["whir_params"]["pow_bits"] = 1

        with self.assertRaisesRegex(SystemExit, "no shared quintic WhirParams tuple"):
            self.compose(reports)

    def test_rejects_non_derived_shared_read_rates(self):
        reports = self.reports()
        reports["read"][1]["scores"][0]["whir_params"]["round_log_inv_rates"] = [7, 12]

        with self.assertRaisesRegex(SystemExit, "must use derived or empty"):
            self.compose(reports)

    def test_rejects_requested_extension_missing_from_a_component_report(self):
        reports = self.reports()
        reports["witness"]["scores"][0]["extension"] = "octic"

        with self.assertRaisesRegex(SystemExit, "witness report has no valid quintic rows"):
            self.compose(reports)

    def test_composes_octic_when_explicitly_selected(self):
        reports = self.reports()
        reports["read"] = [
            report(
                "no-zk",
                26,
                [candidate("r26", 4.0, 400, extension="octic")],
            )
        ]
        for component in ("witness", "fixed_value", "fixed_audit"):
            component_report = reports[component]
            for row in component_report["scores"]:
                row["extension"] = "octic"
            component_report["selected"] = copy.deepcopy(component_report["scores"][0])

        result = self.compose(reports, extension="octic")

        self.assertEqual(result["extension"], "octic")
        self.assertEqual(result["selected"]["extension"], "octic")
        self.assertEqual(result["component_num_variables"]["read"], [26])

    def test_embedded_fixed_audit_has_no_per_proof_time_or_size(self):
        reports = self.reports()
        result = self.compose(reports, fixed_audit_embedded=True)
        row = result["selected"]

        self.assertEqual(row["component_scores"]["fixed_audit"], 0.0)
        self.assertEqual(row["projected_schedule_seconds"], 68.0)
        self.assertEqual(row["proof_size_bytes_estimate"], 1200)
        self.assertTrue(row["fixed_audit_embedded"])
        self.assertTrue(result["model"]["fixed_audit_embedded"])
        self.assertEqual(
            row["setup_config"]["spark_whir_params"]["fixed_audit"],
            reports["fixed_audit"]["scores"][0]["whir_params"],
        )

    def test_embedded_fixed_audit_uses_the_explicit_reference_only(self):
        reports = self.reports()
        reports["fixed_audit"]["scores"] = [
            candidate("fa-fast", 0.1, 100, pow_bits=10),
            candidate("fa-reference", 2.0, 300, pow_bits=6),
        ]
        reference_labels = {
            "witness": "w0",
            "fixed_value": "fv0",
            "fixed_audit": "fa-reference",
            "read": ["r25", "r23"],
        }

        result = self.compose(reports, reference_labels=reference_labels)

        self.assertEqual(
            {row["component_labels"]["fixed_audit"] for row in result["scores"]},
            {"fa-reference"},
        )
        self.assertEqual(
            result["candidate_retention"]["components"]["fixed_audit"]["pareto_rows"],
            1,
        )

    def test_embedded_audit_pow_is_configuration_only(self):
        row = self.compose()["selected"]
        measurement = {
            "measured_seconds": 1.0,
            "heldout_proof_size_median_bytes": 10,
        }

        summary = MODULE.measured_candidate_summary(row, measurement, measurement, 1)

        self.assertNotIn("fixed_audit", summary["per_proof_component_pow_bits"])
        self.assertEqual(
            summary["configuration_only_pow_bits"],
            {"fixed_audit": 0},
        )
        self.assertEqual(summary["max_per_proof_component_pow_bits"], 0)

    def test_rejects_omitted_embedded_audit_flag(self):
        with self.assertRaisesRegex(SystemExit, "imply embedded audit tables"):
            self.compose(fixed_audit_embedded=False)

    def test_rejects_embedded_audit_flag_when_dimensions_do_not_fit(self):
        reports = self.reports()
        reports["fixed_audit"]["num_variables"] = 23

        with self.assertRaisesRegex(SystemExit, "do not fit in the fixed value bundle"):
            self.compose(reports, fixed_audit_embedded=True)

    def test_accepts_separate_audit_when_dimensions_do_not_fit(self):
        reports = self.reports()
        reports["fixed_audit"]["num_variables"] = 23

        result = self.compose(reports, fixed_audit_embedded=False)

        self.assertFalse(result["model"]["fixed_audit_embedded"])
        self.assertEqual(result["selected"]["component_scores"]["fixed_audit"], 12.0)
        self.assertEqual(result["selected"]["proof_size_bytes_estimate"], 1500)

    def test_rejects_invalid_fixed_audit_dimension(self):
        reports = self.reports()
        reports["fixed_audit"]["num_variables"] = 0

        with self.assertRaisesRegex(SystemExit, "fixed_audit.*positive num_variables"):
            self.compose(reports)

    def test_rejects_wrong_quintic_read_dimensions(self):
        reports = self.reports()
        reports["read"][1]["num_variables"] = 24

        with self.assertRaisesRegex(SystemExit, r"must use num_variables \[25, 23\]"):
            self.compose(reports)

    def test_rejects_wrong_quintic_read_report_count(self):
        reports = self.reports()
        reports["read"] = reports["read"][:1]

        with self.assertRaisesRegex(SystemExit, r"must use num_variables \[25, 23\]"):
            self.compose(reports)

    def test_rejects_wrong_octic_read_dimensions(self):
        reports = self.reports()
        for _, component_report in MODULE.named_reports(MODULE.normalize_reports(reports)):
            for row in component_report["scores"]:
                row["extension"] = "octic"
            component_report["selected"] = copy.deepcopy(component_report["scores"][0])

        with self.assertRaisesRegex(SystemExit, r"must use num_variables \[26\]"):
            self.compose(reports, extension="octic")

    def test_composes_full_zk_spark_security_config(self):
        result = self.compose()
        row = result["selected"]
        self.assertEqual(row["setup_config"]["matrix_closing"], "Spark")
        self.assertEqual(row["setup_config"]["security"]["security_level_bits"], 116)
        self.assertEqual(result["component_security_bits"], 120)
        self.assertEqual(result["component_merkle_security_bits"], 123)
    def test_full_zk_witness_grid_uses_matching_zk_parameters(self):
        reports = self.reports()
        reports["witness"]["scores"] = [
            candidate(
                "wrong-zk",
                0.1,
                10,
                zk_ell=4,
                zk_mask_log_inv_rate=5,
            ),
            candidate(
                "matching-zk",
                1.0,
                100,
                zk_ell=3,
                zk_mask_log_inv_rate=3,
            ),
        ]

        result = self.compose(reports)

        self.assertEqual(
            result["selected"]["component_labels"]["witness"], "matching-zk"
        )

    def test_rejects_full_zk_witness_without_matching_zk_parameters(self):
        reports = self.reports()
        reports["witness"]["scores"] = [
            candidate(
                "wrong-zk",
                0.1,
                10,
                zk_ell=4,
                zk_mask_log_inv_rate=5,
            )
        ]

        with self.assertRaisesRegex(
            SystemExit,
            "no valid quintic rows for ell_zk=3 and mask_log_inv_rate=3",
        ):
            self.compose(reports)

    def test_composes_no_zk_spark_config_without_zk_parameters(self):
        reports = self.reports()
        reports["witness"]["proof_mode"] = "no-zk"
        reports["witness"]["scores"][0]["proof_mode"] = "no-zk"
        reports["witness"]["scores"][0].pop("zk_ell")
        reports["witness"]["scores"][0].pop("zk_mask_log_inv_rate")

        result = self.compose(reports, proof_mode="no-zk")

        row = result["selected"]
        self.assertEqual(result["proof_mode"], "no-zk")
        self.assertEqual(row["proof_mode"], "no-zk")
        self.assertNotIn("ell_zk", row["setup_config"])
        self.assertNotIn("mask_log_inv_rate", row["setup_config"])

    def test_fixed_table_setup_domain_caps_filter_impractical_rates(self):
        reports = self.reports()
        reports["fixed_value"]["scores"] = [
            candidate("huge", 0.1, 10, starting_log_inv_rate=8),
            candidate("bounded", 1.0, 20, starting_log_inv_rate=1),
        ]

        result = self.compose(reports)

        self.assertEqual(result["selected"]["component_labels"]["fixed_value"], "bounded")

    def test_rejects_component_report_with_end_to_end_target_in_component_field(self):
        reports = self.reports()
        reports["witness"]["target_security_bits"] = 120
        reports["witness"]["component_security_override_bits"] = None

        with self.assertRaisesRegex(SystemExit, "targets end-to-end security"):
            self.compose(reports)

    def test_rejects_mismatched_component_targets(self):
        reports = self.reports()
        reports["read"][1]["component_security_override_bits"] = 121

        with self.assertRaisesRegex(SystemExit, "must use component security targets 120/123"):
            self.compose(reports)

    def test_rejects_uniformly_low_component_targets(self):
        reports = self.reports()
        for _, component_report in MODULE.named_reports(MODULE.normalize_reports(reports)):
            component_report["component_security_override_bits"] = 116
            component_report["component_merkle_security_override_bits"] = 116

        with self.assertRaisesRegex(SystemExit, "must use component security targets 120/123"):
            self.compose(reports)

    def test_rejected_component_row_is_excluded(self):
        reports = self.reports()
        rejected = candidate(
            "rejected",
            0.01,
            1,
            zk_ell=3,
            zk_mask_log_inv_rate=3,
        )
        rejected["accepted_for_ranking"] = False
        reports["witness"]["scores"].insert(0, rejected)

        result = self.compose(reports)

        self.assertEqual(result["selected"]["component_labels"]["witness"], "w0")

    def test_rejects_invalid_accepted_component_row_context(self):
        mutations = (
            ("proof mode", lambda row: row.__setitem__("proof_mode", "no-zk"), "proof_mode"),
            ("base field", lambda row: row.__setitem__("base_field", "babybear"), "base_field"),
            (
                "WHIR security",
                lambda row: row.__setitem__("whir_component_security_bits", 119),
                "WHIR component security",
            ),
            (
                "Merkle security",
                lambda row: row.__setitem__("merkle_component_security_bits", 122),
                "Merkle component security",
            ),
            (
                "achieved security",
                lambda row: row.__setitem__("security_bits_achieved", 119.9),
                "achieve at least",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                reports = self.reports()
                mutate(reports["witness"]["scores"][0])
                with self.assertRaisesRegex(SystemExit, message):
                    self.compose(reports)

    def test_rejects_duplicate_eligible_component_labels(self):
        reports = self.reports()
        reports["witness"]["scores"].append(
            candidate(
                "w0",
                2.0,
                90,
                pow_bits=1,
                zk_ell=3,
                zk_mask_log_inv_rate=3,
            )
        )

        with self.assertRaisesRegex(SystemExit, "duplicate eligible label w0"):
            self.compose(reports)

    def test_rejects_different_component_coefficients(self):
        reports = self.reports()
        reports["read"][1]["coefficients"] = {"fixture": 2.0}

        with self.assertRaisesRegex(SystemExit, "different calibration coefficients"):
            self.compose(reports)

    def test_rejects_witness_workload_mismatch(self):
        reports = self.reports()
        reports["witness"]["workload_identity"]["r1cs_sha256"] = "11" * 32

        with self.assertRaisesRegex(SystemExit, "differs from the requested workload"):
            self.compose(reports)

    def test_rejects_witness_row_workload_mismatch(self):
        reports = self.reports()
        reports["witness"]["scores"][0]["workload_identity"]["label"] = "other"

        with self.assertRaisesRegex(SystemExit, "differs from the witness report"):
            self.compose(reports)

    def test_workload_identity_hashes_the_r1cs_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.r1cs"
            path.write_bytes(b"r1cs fixture")

            identity = MODULE.workload_identity_from_r1cs(path, "fixture")

        self.assertEqual(identity["label"], "fixture")
        self.assertEqual(
            identity["r1cs_sha256"],
            "45d1cdd645f2aa64ad0e447afeb77a226a7e8382746b1be43ce5b2ea3256832f",
        )

    def test_rejects_mismatched_read_provenance(self):
        reports = self.reports()
        reports["read"][1]["provenance"] = {"git_commit": "other"}

        with self.assertRaisesRegex(SystemExit, "different provenance"):
            self.compose(reports)

    def test_rejects_null_provenance(self):
        reports = self.reports()
        reports["read"][1]["provenance"] = None

        with self.assertRaisesRegex(SystemExit, "non-null provenance"):
            self.compose(reports)

    def test_rejects_heldout_measurements_without_provenance(self):
        measurements = {"proof_mode": "full-zk", "rows": []}

        with self.assertRaisesRegex(
            SystemExit, "heldout measurements must carry non-null provenance"
        ):
            self.compose(measurements=measurements)

    def test_rejects_heldout_measurements_with_different_provenance(self):
        measurements = {
            "provenance": {"git_commit": "other"},
            "proof_mode": "full-zk",
            "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
            "rows": [],
        }

        with self.assertRaisesRegex(
            SystemExit, "heldout measurement provenance differs"
        ):
            self.compose(measurements=measurements)

    def test_rejects_heldout_measurements_with_wrong_proof_mode(self):
        measurements = {
            "provenance": {"git_commit": "test"},
            "proof_mode": "no-zk",
            "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
            "rows": [],
        }

        with self.assertRaisesRegex(
            SystemExit, "heldout measurements use proof_mode=no-zk, expected full-zk"
        ):
            self.compose(measurements=measurements)

    def test_accepts_matching_heldout_measurement_context(self):
        MODULE.require_matching_measurement_context(
            {
                "provenance": {"git_commit": "test"},
                "proof_mode": "full-zk",
                "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
            },
            {"git_commit": "test"},
            "full-zk",
            WORKLOAD_IDENTITY,
        )

    def test_composed_reference_label_is_preserved_in_shortlist_metadata(self):
        reports = self.reports()
        reports["witness"]["scores"].append(
            candidate(
                "w-fast",
                0.1,
                50,
                pow_bits=1,
                zk_ell=3,
                zk_mask_log_inv_rate=3,
            )
        )
        reference_labels = {
            "witness": "w0",
            "fixed_value": "fv0",
            "fixed_audit": "fa0",
            "read": ["r25", "r23"],
        }

        result = self.compose(
            reports,
            reference_labels=reference_labels,
            measurement_rows=1,
        )

        reference_rows = [
            row
            for row in result["measurement_shortlist"]
            if row["component_labels"] == reference_labels
        ]
        self.assertEqual(len(reference_rows), 1)
        self.assertEqual(
            result["measurement_shortlist_meta"]["reference_labels"],
            [reference_rows[0]["label"]],
        )

    def test_measurement_row_must_exactly_match_a_composed_candidate(self):
        composed = self.compose()["measurement_shortlist"][0]

        def changed_label(row):
            row["label"] += "-changed"

        def changed_extension(row):
            row["extension"] = "octic"

        def changed_proof_mode(row):
            row["proof_mode"] = "no-zk"

        def changed_ell(row):
            row["setup_config"]["ell_zk"] = 4

        def changed_mask(row):
            row["setup_config"]["mask_log_inv_rate"] = 4

        def changed_schedule(row):
            row["setup_config"]["whir_params"]["pow_bits"] += 1

        for name, mutate in (
            ("label", changed_label),
            ("extension", changed_extension),
            ("proof mode", changed_proof_mode),
            ("ell", changed_ell),
            ("mask", changed_mask),
            ("schedule", changed_schedule),
        ):
            with self.subTest(name=name):
                measured = copy.deepcopy(composed)
                measured["measured_seconds"] = 1.0
                mutate(measured)
                measurements = {
                    "provenance": {"git_commit": "test"},
                    "proof_mode": "full-zk",
                    "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
                    "rows": [measured],
                }

                with self.assertRaisesRegex(
                    SystemExit, "does not match a composed candidate setup"
                ):
                    self.compose(measurements=measurements)

    def test_exact_composed_measurement_identity_is_accepted(self):
        composed = self.compose()["measurement_shortlist"][0]

        matched = MODULE.require_measurement_rows_match_candidates(
            {"rows": [copy.deepcopy(composed)]},
            [composed],
        )
        self.assertEqual(matched, [composed])

    def test_rejects_measured_constraint_or_case_mismatch(self):
        composed = self.compose()["measurement_shortlist"][0]
        base = copy.deepcopy(composed)
        base["constraint_work"] = WORKLOAD_IDENTITY["constraint_work"]
        base["heldout_case_label"] = WORKLOAD_IDENTITY["label"]

        for field, value, message in (
            ("constraint_work", 1, "different constraint count"),
            ("heldout_case_label", "other", "different case label"),
        ):
            with self.subTest(field=field):
                measured = copy.deepcopy(base)
                measured[field] = value
                with self.assertRaisesRegex(SystemExit, message):
                    MODULE.require_measurement_rows_match_candidates(
                        {"rows": [measured]},
                        [composed],
                        WORKLOAD_IDENTITY,
                    )

    def test_measured_candidates_survive_report_truncation(self):
        reports = self.reports()
        reports["witness"]["scores"] = [
            candidate(
                f"w{index}",
                float(index + 1),
                100 - index,
                pow_bits=index,
                zk_ell=3,
                zk_mask_log_inv_rate=3,
            )
            for index in range(4)
        ]
        initial = MODULE.compose_report(
            reports,
            116,
            116,
            3,
            3,
            4,
            10,
            4,
            None,
            {"fixed_value": 26, "fixed_audit": 24},
            "quintic",
            proof_mode="full-zk",
            fixed_audit_embedded=True,
            workload_identity=copy.deepcopy(WORKLOAD_IDENTITY),
        )
        measured = []
        for index, row in enumerate(initial["scores"]):
            measured_row = copy.deepcopy(row)
            measured_row["measured_seconds"] = 1.0 + index * 0.1
            measured_row["constraint_work"] = WORKLOAD_IDENTITY["constraint_work"]
            measured_row["heldout_case_label"] = WORKLOAD_IDENTITY["label"]
            measured.append(measured_row)

        result = MODULE.compose_report(
            reports,
            116,
            116,
            3,
            3,
            4,
            10,
            4,
            {
                "provenance": {"git_commit": "test"},
                "proof_mode": "full-zk",
                "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
                "rows": measured,
            },
            {"fixed_value": 26, "fixed_audit": 24},
            "quintic",
            proof_mode="full-zk",
            fixed_audit_embedded=True,
            workload_identity=copy.deepcopy(WORKLOAD_IDENTITY),
        )

        self.assertEqual(result["measurement_summary"]["measured_rows"], 4)
        self.assertEqual(
            {row["label"] for row in measured},
            {
                row["label"]
                for row in result["scores"]
                if row["label"] in {measured_row["label"] for measured_row in measured}
            },
        )

    def test_calibration_is_applied_to_every_required_report_row(self):
        reports = self.reports()
        reports["witness"]["scores"] = [
            candidate(
                f"w{index}",
                float(index + 1),
                400 - index * 100,
                pow_bits=index + 1,
                zk_ell=3,
                zk_mask_log_inv_rate=3,
            )
            for index in range(3)
        ]
        initial = MODULE.compose_report(
            reports,
            116,
            116,
            3,
            3,
            3,
            10,
            3,
            None,
            {"fixed_value": 26, "fixed_audit": 24},
            "quintic",
            proof_mode="full-zk",
            fixed_audit_embedded=True,
            workload_identity=copy.deepcopy(WORKLOAD_IDENTITY),
        )
        measured = []
        for row in initial["scores"]:
            measured_row = copy.deepcopy(row)
            measured_row["measured_seconds"] = row["projected_seconds"] + 100.0
            measured_row["constraint_work"] = WORKLOAD_IDENTITY["constraint_work"]
            measured_row["heldout_case_label"] = WORKLOAD_IDENTITY["label"]
            measured.append(measured_row)

        result = MODULE.compose_report(
            reports,
            116,
            116,
            3,
            3,
            3,
            10,
            3,
            {
                "provenance": {"git_commit": "test"},
                "proof_mode": "full-zk",
                "workload_identity": copy.deepcopy(WORKLOAD_IDENTITY),
                "rows": measured,
            },
            {"fixed_value": 26, "fixed_audit": 24},
            "quintic",
            proof_mode="full-zk",
            fixed_audit_embedded=True,
            workload_identity=copy.deepcopy(WORKLOAD_IDENTITY),
        )

        self.assertTrue(all(row["projected_seconds"] > 100.0 for row in result["scores"]))
        self.assertEqual(
            result["selected"]["projected_seconds"],
            min(row["projected_seconds"] for row in result["scores"]),
        )

    def test_rejects_duplicate_measurement_labels(self):
        composed = self.compose()["measurement_shortlist"][0]

        with self.assertRaisesRegex(SystemExit, "duplicate label"):
            MODULE.require_measurement_rows_match_candidates(
                {"rows": [copy.deepcopy(composed), copy.deepcopy(composed)]},
                [composed],
            )

    def test_add_heldout_rejects_both_missing_provenances(self):
        with self.assertRaisesRegex(
            SystemExit, "both artifacts are missing provenance"
        ):
            ADD_HELDOUT_MODULE.require_matching_code_provenance({}, {})

    def test_add_heldout_rejects_row_mode_mismatch(self):
        with self.assertRaisesRegex(
            SystemExit, "heldout row 0 must use proof_mode=full-zk"
        ):
            ADD_HELDOUT_MODULE.require_matching_proof_modes(
                {},
                {"proof_mode": "full-zk"},
                [{"proof_mode": "no-zk"}],
            )

    def test_add_heldout_rejects_top_level_mode_mismatch(self):
        with self.assertRaisesRegex(SystemExit, "proof_mode differs"):
            ADD_HELDOUT_MODULE.require_matching_proof_modes(
                {"proof_mode": "no-zk"},
                {"proof_mode": "full-zk"},
                [],
            )

    def test_add_heldout_accepts_mode_agnostic_calibration(self):
        ADD_HELDOUT_MODULE.require_matching_proof_modes(
            {},
            {"proof_mode": "full-zk"},
            [{"proof_mode": "full-zk"}],
        )

    def test_rejects_read_report_with_wrong_end_to_end_target(self):
        reports = self.reports()
        reports["read"][1]["target_security_bits"] = 120

        with self.assertRaisesRegex(SystemExit, r"read\[1\] report targets end-to-end"):
            self.compose(reports)

    def test_affine_calibration_holds_out_every_third_row(self):
        pairs = [(1.0, 3.0), (2.0, 5.0), (4.0, 9.0), (5.0, 11.0)]
        intercept, scale = MODULE.fit_affine(pairs)
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(scale, 2.0)

    def test_calibration_preserves_model_ordering(self):
        rows = [
            {"label": f"r{i}", "projected_schedule_seconds": float(i)}
            for i in range(1, 5)
        ]
        measurements = {
            "rows": [
                {"label": f"r{i}", "measured_seconds": 10.0 + float(i)}
                for i in range(1, 5)
            ]
        }
        result = MODULE.calibrate(rows, measurements)
        self.assertEqual(result["scale"], 1.0)
        self.assertTrue(result["validation_within_ten_percent"])

    def test_component_calibration_uses_reference_perturbations(self):
        components = ("witness", "fixed_value", "fixed_audit", "read")
        reference_labels = {component: f"{component}0" for component in components}

        def row(label, scores, labels):
            return {
                "label": label,
                "projected_schedule_seconds": sum(scores.values()),
                "component_scores": scores,
                "component_labels": labels,
            }

        reference_scores = {component: 1.0 for component in components}
        rows = [row("reference", reference_scores, reference_labels)]
        measurements = [{"label": "reference", "measured_seconds": 25.0}]
        expected_scales = {
            "witness": 2.0,
            "fixed_value": 3.0,
            "fixed_audit": 4.0,
            "read": 5.0,
        }
        for component in components:
            scores = dict(reference_scores)
            scores[component] = 2.0
            labels = dict(reference_labels)
            labels[component] = f"{component}1"
            rows.append(row(component, scores, labels))
            measurements.append(
                {"label": component, "measured_seconds": 25.0 + expected_scales[component]}
            )
        validation_scores = {component: 2.0 for component in components}
        validation_labels = {component: f"{component}1" for component in components}
        rows.append(row("validation", validation_scores, validation_labels))
        measurements.append({"label": "validation", "measured_seconds": 39.0})

        result = MODULE.calibrate(
            rows,
            {"rows": measurements},
            reference_labels,
        )

        self.assertEqual(result["component_scales"], expected_scales)
        self.assertEqual(result["unit_scale_fallback_components"], [])
        self.assertEqual(result["validation_rows"], 1)
        self.assertTrue(result["validation_within_ten_percent"])

    def test_embedded_audit_calibration_uses_only_active_components(self):
        components = ("witness", "fixed_value", "fixed_audit", "read")
        reference_labels = {component: f"{component}0" for component in components}

        def row(label, scores, labels):
            return {
                "label": label,
                "projected_schedule_seconds": sum(scores.values()),
                "projected_seconds": sum(scores.values()),
                "component_scores": scores,
                "component_labels": labels,
            }

        reference_scores = {
            "witness": 1.0,
            "fixed_value": 1.0,
            "fixed_audit": 0.0,
            "read": 1.0,
        }
        rows = [row("reference", reference_scores, reference_labels)]
        measurements = [{"label": "reference", "measured_seconds": 13.0}]
        expected_scales = {"witness": 2.0, "fixed_value": 3.0, "read": 5.0}
        for component, scale in expected_scales.items():
            scores = dict(reference_scores)
            scores[component] += 1.0
            labels = dict(reference_labels)
            labels[component] = f"{component}1"
            rows.append(row(component, scores, labels))
            measurements.append({"label": component, "measured_seconds": 13.0 + scale})
        validation_scores = dict(reference_scores)
        validation_labels = dict(reference_labels)
        for component in expected_scales:
            validation_scores[component] += 1.0
            validation_labels[component] = f"{component}1"
        rows.append(row("validation", validation_scores, validation_labels))
        measurements.append({"label": "validation", "measured_seconds": 23.0})

        calibration = MODULE.calibrate(
            rows,
            {"rows": measurements},
            reference_labels,
            fixed_audit_embedded=True,
        )
        MODULE.apply_calibration(rows, calibration, fixed_audit_embedded=True)

        self.assertEqual(calibration["component_scales"], expected_scales)
        self.assertNotIn("fixed_audit", calibration["component_scales"])
        self.assertEqual(rows[-1]["projected_seconds"], 23.0)

    def test_fixed_score_excludes_only_the_initial_setup_commitment(self):
        row = candidate("fixed", 2.0, 10, starting_log_inv_rate=1)
        row["_component_num_variables"] = 2
        row["dft_work"] = 16
        row["merkle_work"] = 24

        score = MODULE.component_score(row, "fixed_value")

        # Four opening terms cost 8. The initial setup work is 2^(2 + 1) = 8,
        # leaving half the DFT cost and two thirds of the Merkle cost.
        self.assertAlmostEqual(score, 8.0 + 1.0 + 4.0 / 3.0)

    def test_prover_confidence_interval_does_not_expand_one_percent_band(self):
        def row(label, proof_size, pow_bits):
            params = {"pow_bits": pow_bits}
            return {
                "label": label,
                "extension": "quintic",
                "proof_mode": "full-zk",
                "proof_size_bytes_estimate": proof_size,
                "setup_config": {
                    "whir_params": params,
                    "spark_whir_params": {
                        "fixed_value": params,
                        "fixed_audit": params,
                        "read": params,
                    },
                },
            }

        rows = [
            row("fast", 20, 4),
            row("slow-small", 10, 2),
        ]
        measurements = {
            "rows": [
                {
                    "label": "fast",
                    "measured_seconds": 1.0,
                    "heldout_median_ci_seconds": [0.9, 1.2],
                    "heldout_paired_relative_median_ci": [0.0, 0.0],
                    "heldout_proof_size_median_bytes": 20,
                },
                {
                    "label": "slow-small",
                    "measured_seconds": 1.1,
                    "heldout_median_ci_seconds": [0.95, 1.2],
                    "heldout_paired_relative_median_ci": [0.05, 0.15],
                    "heldout_proof_size_median_bytes": 10,
                },
            ]
        }

        selected, summary = MODULE.measured_selection(rows, measurements)

        self.assertEqual(selected["label"], "fast")
        self.assertEqual(summary["paired_confidence_interval_role"], "diagnostic_only")
        self.assertEqual(
            [row["label"] for row in summary["tied_with_best"]],
            ["fast"],
        )
        self.assertEqual(
            summary["tied_with_best"][0]["per_proof_component_pow_bits"],
            {"witness": 4, "fixed_value": 4, "fixed_audit": 4, "read": 4},
        )
        self.assertEqual(
            summary["tied_with_best"][0]["heldout_proof_size_median_bytes"], 20
        )

    def test_paired_confidence_interval_does_not_override_marginal_intervals(self):
        rows = [
            {"label": "fast", "proof_size_bytes_estimate": 20},
            {"label": "slow-small", "proof_size_bytes_estimate": 10},
        ]
        measurements = {
            "rows": [
                {
                    "label": "fast",
                    "measured_seconds": 1.0,
                    "heldout_median_ci_seconds": [0.99, 1.01],
                    "heldout_paired_relative_median_ci": [0.0, 0.0],
                    "heldout_proof_size_median_bytes": 20,
                },
                {
                    "label": "slow-small",
                    "measured_seconds": 1.1,
                    "heldout_median_ci_seconds": [1.09, 1.11],
                    "heldout_paired_relative_median_ci": [-0.01, 0.01],
                    "heldout_proof_size_median_bytes": 10,
                },
            ]
        }

        selected, summary = MODULE.measured_selection(rows, measurements)

        self.assertEqual(selected["label"], "fast")
        self.assertEqual(
            [row["label"] for row in summary["tied_with_best"]], ["fast"]
        )

    def test_exactly_one_percent_slowdown_is_tied(self):
        self.assertTrue(
            MODULE.measurement_is_tied(
                {
                    "measured_seconds": 1.01,
                    "heldout_median_ci_seconds": [1.01, 1.01],
                },
                {
                    "measured_seconds": 1.0,
                    "heldout_median_ci_seconds": [1.0, 1.0],
                },
            )
        )

    def test_nearly_touching_prover_intervals_do_not_expand_band(self):
        self.assertFalse(
            MODULE.measurement_is_tied(
                {
                    "measured_seconds": 1.1,
                    "heldout_median_ci_seconds": [1.0000000000005, 1.2],
                },
                {
                    "measured_seconds": 1.0,
                    "heldout_median_ci_seconds": [0.9, 1.0],
                },
            )
        )

    def test_measured_selection_uses_proof_size_below_one_percent(self):
        rows = [
            {"label": "fast", "proof_size_bytes_estimate": 20},
            {"label": "small", "proof_size_bytes_estimate": 10},
        ]
        measurements = {
            "rows": [
                {
                    "label": "fast",
                    "measured_seconds": 0.994,
                    "heldout_median_ci_seconds": [0.992, 0.996],
                    "heldout_relative_median_difference": 0.0,
                    "heldout_paired_relative_median_ci": [0.0, 0.0],
                    "heldout_proof_size_median_bytes": 20,
                },
                {
                    "label": "small",
                    "measured_seconds": 1.0,
                    "heldout_median_ci_seconds": [0.998, 1.002],
                    "heldout_relative_median_difference": 0.006,
                    "heldout_paired_relative_median_ci": [0.004, 0.008],
                    "heldout_proof_size_median_bytes": 10,
                },
            ]
        }

        selected = MODULE.select_measured(rows, measurements)

        self.assertEqual(selected["label"], "small")

    def test_measured_selection_uses_single_thread_verifier_before_size(self):
        rows = [
            {"label": "small", "proof_size_bytes_estimate": 10},
            {"label": "fast-verifier", "proof_size_bytes_estimate": 20},
        ]
        measurements = {
            "rows": [
                {
                    "label": "small",
                    "measured_seconds": 1.0,
                    "heldout_median_ci_seconds": [0.99, 1.01],
                    "heldout_proof_size_median_bytes": 10,
                },
                {
                    "label": "fast-verifier",
                    "measured_seconds": 1.005,
                    "heldout_median_ci_seconds": [0.995, 1.015],
                    "heldout_proof_size_median_bytes": 20,
                },
            ]
        }
        verifier_measurements = {
            "rayon_threads": 1,
            "rows": [
                {"label": "small", "heldout_verify_seconds": 0.03},
                {"label": "fast-verifier", "heldout_verify_seconds": 0.01},
            ],
        }

        selected, summary = MODULE.measured_selection(
            rows, measurements, verifier_measurements
        )

        self.assertEqual(selected["label"], "fast-verifier")
        self.assertEqual(selected["heldout_verify_seconds"], 0.01)
        self.assertEqual(summary["verifier_rayon_threads"], 1)

    def test_measured_selection_uses_size_when_verifier_intervals_overlap(self):
        rows = [
            {"label": "small", "proof_size_bytes_estimate": 10},
            {"label": "slightly-faster-verifier", "proof_size_bytes_estimate": 20},
        ]
        measurements = {
            "rows": [
                {
                    "label": "small",
                    "measured_seconds": 1.0,
                    "heldout_proof_size_median_bytes": 10,
                },
                {
                    "label": "slightly-faster-verifier",
                    "measured_seconds": 1.005,
                    "heldout_proof_size_median_bytes": 20,
                },
            ]
        }
        verifier_measurements = {
            "rayon_threads": 1,
            "rows": [
                {
                    "label": "small",
                    "heldout_verify_seconds": 0.0101,
                    "heldout_verify_median_ci_seconds": [0.0098, 0.0104],
                },
                {
                    "label": "slightly-faster-verifier",
                    "heldout_verify_seconds": 0.01,
                    "heldout_verify_median_ci_seconds": [0.0097, 0.0103],
                },
            ],
        }

        selected, summary = MODULE.measured_selection(
            rows, measurements, verifier_measurements
        )

        self.assertEqual(selected["label"], "small")
        self.assertEqual(summary["verifier_tied_count"], 2)

    def test_three_axis_frontier_retains_lower_verifier_work(self):
        rows = [
            {"label": "small", "time": 1.0, "verifier": 3.0, "size": 10},
            {"label": "verify", "time": 1.0, "verifier": 1.0, "size": 20},
            {"label": "dominated", "time": 2.0, "verifier": 4.0, "size": 30},
        ]

        frontier = MODULE.pareto_rows_3d(
            rows,
            lambda row: row["time"],
            lambda row: row["verifier"],
            lambda row: row["size"],
            lambda row: row["label"],
        )

        self.assertEqual({row["label"] for row in frontier}, {"small", "verify"})


if __name__ == "__main__":
    unittest.main()
