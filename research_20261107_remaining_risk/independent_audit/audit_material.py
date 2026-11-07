"""Saved-output remaining512 audit. Never imports producer/core or fits models.

Source/template preparation alone grants no execution permission. A root-bound
plan and exact one-time approval are checked before reading any trial artifacts.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

UNIT = "source bulk GPa attribution, aggregation qualified"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("Duplicate JSON key")
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def timepoint(text):
    value = datetime.fromisoformat(text)
    if value.utcoffset() is None:
        raise AssertionError("Timezone-aware actual UTC required")
    return value


class Checks:
    def __init__(self):
        self.counts = {}
        self.errors = {}

    def check(self, kind, condition):
        if not condition:
            raise AssertionError(kind)
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def exact(self, kind, actual, expected):
        self.check(kind, actual == expected)

    def number(self, kind, actual, expected, atol=2e-13, rtol=2e-14):
        error = abs(actual - expected)
        self.check(kind, math.isfinite(actual) and math.isfinite(expected)
                   and error <= atol + rtol * abs(expected))
        self.errors[kind] = max(self.errors.get(kind, 0), error)


def descriptor_pair(identity):
    amounts = {key: F(str(value)) for key, value in identity["composition_amounts"].items()
               if not key.startswith("@") and F(str(value)) > 0}
    total = sum(amounts.values(), F(0))
    if total <= 0:
        raise AssertionError("Positive independent composition")
    fractions = {key: float(value / total) for key, value in amounts.items()}
    group = lambda names: math.fsum(fractions.get(key, 0.0) for key in names)
    raw = [float("O" in fractions), group(("O",)), group(("H",)), group(("C", "N")),
           group(("F", "Cl", "Br", "I")), group(("S", "Se", "Te")), float(len(fractions)),
           -math.fsum(value * math.log(value) for value in fractions.values())]
    return raw, raw[:6] + [raw[6] / 118, raw[7] / math.log(118)]


def leaf(tree, row):
    left, right = tree["children_left"], tree["children_right"]
    n = len(left)
    if not n or any(len(tree[key]) != n for key in ("children_right", "feature", "threshold")):
        raise AssertionError("Consistent nonempty tree arrays")
    node, visited = 0, set()
    while True:
        if node in visited or not 0 <= node < n:
            raise AssertionError("Acyclic valid tree index")
        visited.add(node)
        if left[node] == -1:
            if right[node] != -1:
                raise AssertionError("Matched leaf sentinel")
            return node
        feature = tree["feature"][node]
        if type(feature) is not int or not 0 <= feature < len(row):
            raise AssertionError("Valid split feature")
        node = left[node] if row[feature] <= tree["threshold"][node] else right[node]


def verify_forecasts(plan, science, helper, checks):
    root = Path(plan["forecast_root"])
    freeze = read(root / "FORECAST_FREEZE.json")
    checks.exact("forecast_freeze_status", freeze["status"], "ALL512_F_G_LAWS_FROZEN_BEFORE_ANY_TARGET")
    checks.exact("forecast_plan_binding", freeze["plan_sha256"], plan["pins"]["science_plan"]["sha256"])
    checks.exact("forecast_source_binding", freeze["source_sha256"], science["source_sha256"])
    checks.exact("forecast_identity_binding", freeze["identity_roles_sha256"], science["inputs"]["identity_roles"]["sha256"])
    for name, digest in freeze["files_sha256"].items():
        checks.exact("frozen_forecast_payload_hashes", sha(root / name), digest)
    for name, expected in (("law_pairs", 512), ("new_targets", 0), ("new_fits", 0),
                           ("new_calibration_optimizations", 0), ("reference_train_rows", 296)):
        checks.exact("forecast_scope_" + name, freeze[name], expected)
    roles = read(science["inputs"]["identity_roles"]["file"])
    identities = [row["representative"] for row in roles["selected_groups"]]
    ids = [row["component_id"] for row in roles["selected_groups"]]
    checks.check("512_unique_frozen_original_groups", len(ids) == len(set(ids)) == 512)
    contexts = read(root / "CONTEXTS512.json")["records"]
    forecasts = read(root / "FORECASTS512.json")["records"]
    geometry = read(root / "GEOMETRY512.json")["records"]
    routes = read(root / "RF_ROUTES512.json")["records"]
    for label, rows in (("contexts", contexts), ("forecasts", forecasts),
                        ("geometry", geometry), ("routes", routes)):
        checks.exact(label + "_fixed_identity_order", [row["component_id"] for row in rows], ids)
    state = read(science["inputs"]["frozen_rf"]["file"])
    transform = state["fixed_feature_transform"]
    checks.exact("unchanged_RF_training_size", state["train_rows"], 296)
    checks.exact("unchanged_RF_tree_count", len(state["trees"]), 256)
    fit_x = state["X_train_standardized_then_float32"]
    y = state["y_train_native"]
    checks.check("RF296_saved_training_shapes", len(fit_x) == len(y) == 296
                 and all(len(row) == 10 for row in fit_x))
    checks.check("RF296_finite_training_source", all(math.isfinite(value) for value in y)
                 and all(math.isfinite(value) and helper.float32(value) == value for row in fit_x for value in row))
    checks.check("positive_fixed_transform", len(transform["center"]) == len(transform["sd"]) == 10
                 and all(math.isfinite(value) for value in transform["center"])
                 and all(math.isfinite(value) and value > 0 for value in transform["sd"]))
    auto = read(science["inputs"]["frozen_auto"]["file"])["views"]["auto"]
    selection = read(science["inputs"]["descriptor_selection"]["file"])
    checks.exact("saved_g5_selection", selection["auto_index"], 5)
    checks.exact("saved_g5_auto_kind", auto["kind"], "context")
    checks.exact("saved_g5_auto_status", auto["status"], "CONVERGED_KKT")
    checks.exact("saved_g5_auto_index", auto["descriptor_index"], 5)
    query_x = []
    for identity, context, forecast, route in zip(identities, contexts, forecasts, routes):
        checks.exact("original_context_identity", context["identity"], identity)
        checks.exact("original_forecast_identity", forecast["identity"], identity)
        raw, bounded = descriptor_pair(identity)
        for name, actual, expected in (("raw_composition_descriptor", context["raw_descriptors8"], raw),
                                       ("bounded_composition_descriptor", context["bounded_descriptors8"], bounded)):
            checks.exact(name + "_length", len(actual), 8)
            for a, b in zip(actual, expected):
                checks.number(name, a, b)
        checks.exact("known_proxy_forecast_context_binding", forecast["native_proxies"], context["native_proxies"])
        checks.exact("known_descriptor_forecast_context_binding", forecast["bounded_descriptors8"], context["bounded_descriptors8"])
        features = context["native_proxies"] + raw
        checks.exact("fixed_native_feature_count", len(features), 10)
        row = [helper.float32((value - center) / sd)
               for value, center, sd in zip(features, transform["center"], transform["sd"])]
        checks.exact("independently_reconstructed_query_float32", route["query_standardized_float32"], row)
        query_x.append(row)
    expected_weights = [[F(0) for _ in y] for _ in ids]
    for tree_index, tree in enumerate(state["trees"]):
        training_leaves = [leaf(tree, row) for row in fit_x]
        checks.exact("all296_original_training_routes", tree["original_train_leaf_ids"], training_leaves)
        membership = defaultdict(list)
        for index, value in enumerate(training_leaves):
            membership[value].append(index)
        for row_index, row in enumerate(query_x):
            reached = leaf(tree, row)
            checks.exact("all512_saved_RF_query_routes", routes[row_index]["leaf_ids"][tree_index], reached)
            members = membership[reached]
            mass = F(1, 256 * len(members))
            for index in members:
                expected_weights[row_index][index] += mass
    output = {}
    for index, (forecast, geom) in enumerate(zip(forecasts, geometry)):
        checks.exact("native_reference_unit", forecast["F_law"]["unit"], UNIT)
        checks.exact("native_candidate_unit", forecast["G_law"]["unit"], UNIT)
        expected = {}
        for value, weight in zip(y, expected_weights[index]):
            if weight:
                point = helper.binary(value)
                expected[point] = expected.get(point, F(0)) + weight
        f = helper.atoms(forecast["F_law"])
        g = helper.atoms(forecast["G_law"])
        checks.exact("all512_exact_equal_tree_original_row_CDF", f, tuple(sorted(expected.items())))
        own_g = helper.transformed_atoms(f, auto["theta0"], auto["theta1"], forecast["bounded_descriptors8"][5])
        checks.exact("all512_frozen_auto_corrected_CDF", g, own_g)
        own = helper.all_real_geometry(f, g)
        saved = geom["envelope"]
        checks.exact("all_real_gap_lower", F(saved["lower_exact_fraction"]), own["lower"])
        checks.exact("all_real_gap_upper", F(saved["upper_exact_fraction"]), own["upper"])
        checks.exact("CDF_squared_curvature", F(geom["curvature_exact_fraction"]), own["curvature"])
        checks.exact("all_real_width", F(geom["width_exact_fraction"]), own["upper"] - own["lower"])
        checks.exact("all_union_knots", [F(value) for value in saved["knots_exact_fraction"]], own["knots"])
        checks.exact("all_union_knot_gaps", [F(value) for value in saved["knot_gaps_exact_fraction"]], own["gaps"])
        output[ids[index]] = {"F": f, "G": g, "geometry": own, "identity": identities[index]}
    denominator_lcm = 1
    for value in output.values():
        width = value["geometry"]["upper"] - value["geometry"]["lower"]
        if width:
            denominator_lcm = denominator_lcm // math.gcd(denominator_lcm, width.denominator) * width.denominator
            checks.check("actual_prelabel_width_LCM_within_core_guard", denominator_lcm.bit_length() <= 200000)
    diagnostics = read(root / "PREDICTIVE_POWER_DIAGNOSTICS.json")
    bsum = sum((value["geometry"]["curvature"] for value in output.values()), F(0))
    wsum = sum((value["geometry"]["upper"] - value["geometry"]["lower"] for value in output.values()), F(0))
    checks.exact("label_free_diagnostic_curvature_sum", F(diagnostics["B_sum_exact_fraction"]), bsum)
    checks.exact("label_free_diagnostic_width_sum", F(diagnostics["width_sum_exact_fraction"]), wsum)
    checks.exact("label_free_diagnostic_nonzero_widths", diagnostics["nonzero_width_groups"],
                 sum(value["geometry"]["upper"] > value["geometry"]["lower"] for value in output.values()))
    checks.exact("label_free_diagnostic_zero_B_groups", diagnostics["zero_B_groups"],
                 sum(value["geometry"]["curvature"] == 0 for value in output.values()))
    for label, predicate in (("g_zero", lambda value: value == 0), ("g_positive", lambda value: value > 0)):
        chosen = [row["component_id"] for row in forecasts if predicate(row["bounded_descriptors8"][5])]
        saved = diagnostics["predeclared_g0_gpositive_strata_descriptive_only"][label]
        checks.exact("label_free_g_stratum_size", saved["groups"], len(chosen))
        checks.exact("label_free_g_stratum_B_sum", F(saved["B_sum_exact_fraction"]),
                     sum((output[key]["geometry"]["curvature"] for key in chosen), F(0)))
        checks.exact("label_free_g_stratum_width_sum", F(saved["width_sum_exact_fraction"]),
                     sum((output[key]["geometry"]["upper"] - output[key]["geometry"]["lower"] for key in chosen), F(0)))
    return freeze, output, {"exact_denominator_lcm_bit_length": denominator_lcm.bit_length(),
                            "initial_width_exact_fraction": str(wsum),
                            "initial_curvature_exact_fraction": str(bsum)}


def load_forecast_after_review(plan, checks):
    """Reuse an actual earlier route/geometry audit, without repeating inference."""
    pin = plan["pins"]["forecast_review"]
    review = read(pin["file"])
    root = Path(plan["forecast_root"])
    checks.exact("actual_forecast_review_status", review["status"], "PASS_INDEPENDENT_REMAINING512_FORECASTS")
    checks.exact("forecast_review_scientific_plan", review["scientific_plan_sha256"], plan["pins"]["science_plan"]["sha256"])
    checks.exact("forecast_review_current_freeze", review["forecast_freeze_sha256"], sha(root / "FORECAST_FREEZE.json"))
    checks.check("forecast_review_prelabel_core_capacity", review["exact_denominator_lcm_bit_length"] <= 200000)
    freeze = read(root / "FORECAST_FREEZE.json")
    for name, digest in freeze["files_sha256"].items():
        checks.exact("forecast_payloads_still_immutable", sha(root / name), digest)
    forecasts = read(root / "FORECASTS512.json")["records"]
    geometry = read(root / "GEOMETRY512.json")["records"]
    output = {}
    for row, geom in zip(forecasts, geometry):
        checks.exact("reviewed_pair_identity", row["component_id"], geom["component_id"])
        output[row["component_id"]] = {"F": None, "G": None,
            "F_law": row["F_law"], "G_law": row["G_law"],
            "geometry": {"lower": F(geom["envelope"]["lower_exact_fraction"]),
                         "upper": F(geom["envelope"]["upper_exact_fraction"]),
                         "curvature": F(geom["curvature_exact_fraction"])},
            "identity": row["identity"], "g5": row["bounded_descriptors8"][5]}
    checks.exact("reviewed_forecast_pair_count", len(output), 512)
    return freeze, output, review


def jsonlines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def verify_queries(plan, science, helper, checks, freeze, laws):
    root = Path(plan["query_root"])
    close = read(root / "PILOT_CLOSED.json")
    adoption = read(root / "ADOPTION_FREEZE.json")
    seed = read(root / "OS_SEED_AFTER_FREEZE.json")
    snapshot = read(root / "CORE_FINAL_SNAPSHOT.json")
    prefixes = jsonlines(root / "DECISION_PREFIXES.jsonl")
    query_rows = jsonlines(root / "QUERIES.jsonl")
    ticket_rows = jsonlines(root / "TICKETS.jsonl")
    census = read(root / "CENSUS_REMAINING.json")
    census_rows = jsonlines(root / "CENSUS_REQUESTS.jsonl")
    truths = read(root / "ALL512_TARGETS.json")
    science_sha = plan["pins"]["science_plan"]["sha256"]
    forecast_sha = sha(Path(plan["forecast_root"]) / "FORECAST_FREEZE.json")
    adoption_sha, seed_sha = sha(root / "ADOPTION_FREEZE.json"), sha(root / "OS_SEED_AFTER_FREEZE.json")
    forecast_review = read(plan["pins"]["forecast_review"]["file"])
    approval = read(plan["pins"]["query_approval"]["file"])
    checks.exact("actual_query_approval", approval["approved"], True)
    checks.exact("query_approval_science", approval["plan_sha256"], science_sha)
    checks.exact("query_approval_actual_forecast_review", approval["forecast_review"]["sha256"], plan["pins"]["forecast_review"]["sha256"])
    checks.check("query_approval_phase", "query_and_census" in approval["allowed_phases"])
    checks.exact("query_close_status", close["status"], "QUERY_DECISION_AND512_TARGETS_CLOSED_AWAITING_INDEPENDENT_STATE_AUDIT")
    checks.exact("alpha_freeze_status", adoption["status"], "ALPHA_AND_REMAINING_SET_FROZEN_BEFORE_CENSUS")
    checks.exact("OS_seed_receipt_status", seed["status"], "ACTUAL_OS_SEED_RECORDED_AFTER_FORECAST_FREEZE")
    for record in (close, adoption, seed):
        checks.exact("query_science_plan_binding", record["plan_sha256"], science_sha)
        checks.exact("query_forecast_binding", record["forecast_freeze_sha256"], forecast_sha)
    checks.exact("query_source_binding", close["source_sha256"], science["source_sha256"])
    checks.exact("adoption_source_binding", adoption["source_sha256"], science["source_sha256"])
    checks.exact("adoption_OS_seed_binding", adoption["seed_sha256"], seed_sha)
    checks.exact("adoption_final_snapshot_binding", adoption["core_snapshot_sha256"], sha(root / "CORE_FINAL_SNAPSHOT.json"))
    checks.exact("close_adoption_binding", close["decision_freeze_sha256"], adoption_sha)
    for name, digest in close["files_sha256"].items():
        checks.exact("closed_query_payload_hashes", sha(root / name), digest)
    entropy = bytes.fromhex(seed["seed_hex"])
    checks.exact("OS_entropy_bytes", len(entropy), 32)
    checks.exact("OS_entropy_integer_encoding", seed["seed_integer"], int.from_bytes(entropy, "big"))
    checks.exact("OS_entropy_source", seed["entropy_source"], "os.urandom32bytes")
    checks.check("OS_entropy_record_after_law_freeze", timepoint(seed["actual_utc"]) >= timepoint(freeze["actual_utc"]))
    checks.check("OS_entropy_after_actual_forecast_audit", timepoint(seed["actual_utc"]) >= timepoint(forecast_review["actual_end_utc"]))
    rng = random.Random(seed["seed_integer"])
    units = {key: value["geometry"] for key, value in laws.items()}
    total_midpoint = sum(((u["lower"] + u["upper"]) / 2 for u in units.values()), F(0))
    widths = {key: u["upper"] - u["lower"] for key, u in units.items()}
    initial_width = sum(widths.values(), F(0))
    denominator_lcm = 1
    guard_rejected = False
    for key in sorted(units):
        if widths[key]:
            denominator = widths[key].denominator
            denominator_lcm = denominator_lcm // math.gcd(denominator_lcm, denominator) * denominator
            if denominator_lcm.bit_length() > 200000:
                guard_rejected = True
                break
    checks.check("matched_prefix_ticket_query_counts", len(prefixes) == len(query_rows) + 1
                 and len(query_rows) == len(ticket_rows) <= 384)
    observed, residual_sum, sum_z, variance = {}, F(0), F(0), F(0)
    events, queried, invalid = [], [], False
    if guard_rejected:
        checks.exact("prelabel_LCM_guard_no_queries", len(query_rows), 0)
        checks.exact("prelabel_LCM_guard_snapshot", snapshot["status"], "confidence_initialization_abstain")
        checks.check("prelabel_LCM_guard_reason", "LCM bit guard" in snapshot["error"])
        checks.exact("prelabel_LCM_guard_alpha", F(adoption["alpha"]), F(0))
        last = {"remaining": list(laws), "alpha": F(0), "strict": False, "invalid": True}
    else:
        checks.exact("initial_prefix_count", prefixes[0]["query_count"], 0)
        last = helper.prefix_decision(prefixes[0]["decision"], units, observed,
                                       sum_z, variance, initial_width)
        checks.exact("unearned_initial_positive_step", last["alpha"], F(0))
        integer_weights = {key: int(widths[key] * denominator_lcm)
                           for key in sorted(units) if widths[key] > 0}
        for index, (query, directory) in enumerate(zip(query_rows, ticket_rows), 1):
            checks.check("no_query_after_prior_strict_or_invalid_prefix", not last["strict"] and not last["invalid"])
            ticket_path = root / directory["ticket_file"]
            saved_ticket = read(ticket_path)
            ticket = saved_ticket["ticket"]
            checks.exact("ticket_directory_hash", sha(ticket_path), directory["ticket_sha256"])
            checks.exact("ticket_status_precedes_label", saved_ticket["status"], "TICKET_PERSISTED_BEFORE_TARGET")
            checks.exact("ticket_seed_binding", saved_ticket["seed_sha256"], seed_sha)
            checks.exact("ticket_forecast_binding", saved_ticket["forecast_freeze_sha256"], forecast_sha)
            checks.exact("ticket_science_binding", saved_ticket["plan_sha256"], science_sha)
            weights = [(key, weight) for key, weight in integer_weights.items() if key not in observed]
            total_weight = sum(weight for _, weight in weights)
            own_ticket = rng.randrange(total_weight)
            cursor, selected, selected_weight = 0, None, None
            for key, weight in weights:
                cursor += weight
                if own_ticket < cursor:
                    selected, selected_weight = key, weight
                    break
            probability = F(selected_weight, total_weight)
            width_before = sum((widths[key] for key in units if key not in observed), F(0))
            expected_ticket = {"identity": selected, "query_index": index, "ticket": own_ticket,
                "integer_weight": selected_weight, "integer_weight_total": total_weight,
                "propensity": str(probability.numerator) + "/" + str(probability.denominator),
                "width_before": str(width_before.numerator) + "/" + str(width_before.denominator)}
            checks.exact("actual_integer_PPS_ticket_full_record", ticket, expected_ticket)
            checks.exact("ticket_directory_index", directory["query_index"], index)
            checks.exact("ticket_directory_identity", directory["component_id"], selected)
            checks.exact("queried_identity_matches_actual_ticket", query["component_id"], selected)
            checks.check("ticket_persisted_before_actual_target_request",
                         timepoint(seed["actual_utc"]) <= timepoint(saved_ticket["actual_utc"])
                         <= timepoint(query["actual_request_utc"]) <= timepoint(query["actual_return_utc"]))
            permit = query["permit"]
            checks.exact("query_permit_phase", permit["phase"], "query")
            checks.exact("query_permit_identity", permit["component_id"], selected)
            checks.exact("query_permit_science", permit["plan_sha256"], science_sha)
            checks.exact("query_permit_forecast", permit["forecast_freeze"]["sha256"], forecast_sha)
            checks.exact("query_permit_seed", permit["seed"]["sha256"], seed_sha)
            checks.exact("query_permit_actual_ticket", permit["ticket"]["sha256"], sha(ticket_path))
            if laws[selected]["F"] is None:
                laws[selected]["F"] = helper.atoms(laws[selected]["F_law"])
                laws[selected]["G"] = helper.atoms(laws[selected]["G_law"])
            f, g = laws[selected]["F"], laws[selected]["G"]
            target = helper.binary(query["native_target"])
            fscore, gscore = helper.integral_crps(f, target), helper.integral_crps(g, target)
            checks.exact("queried_F_integral_energy_CRPS", fscore, helper.energy_crps(f, target))
            checks.exact("queried_G_integral_energy_CRPS", gscore, helper.energy_crps(g, target))
            gap = gscore - fscore
            checks.exact("queried_exact_gap", F(query["gap_exact_fraction"]), gap)
            checks.check("actual_queried_gap_in_allreal_bounds", units[selected]["lower"] <= gap <= units[selected]["upper"])
            midpoint = (units[selected]["lower"] + units[selected]["upper"]) / 2
            residual = gap - midpoint
            z = total_midpoint + residual_sum + residual / probability
            sum_z += z
            variance += width_before**2
            event = query["core_event"]
            for field, expected in (("gap", gap), ("midpoint", midpoint), ("z", z),
                                     ("sum_z", sum_z), ("sum_width_squared", variance)):
                checks.exact("all_prefix_exact_core_" + field, F(event[field]), expected)
            for field, expected in expected_ticket.items():
                checks.exact("core_event_matches_ticket_" + field, event[field], expected)
            observed[selected] = gap
            residual_sum += residual
            queried.append(selected)
            events.append(event)
            prefix = prefixes[index]
            checks.exact("prefix_count_after_target", prefix["query_count"], index)
            checks.check("prefix_decision_after_target_return", timepoint(prefix["actual_utc"]) >= timepoint(query["actual_return_utc"]))
            last = helper.prefix_decision(prefix["decision"], units, observed,
                                           sum_z, variance, initial_width, invalid)
            invalid = last["invalid"]
            checks.check("full_exact_prefix_confidence_and_remaining_translation", True)
        checks.exact("final_snapshot_schema", snapshot["schema"], "exact_remaining_risk_v1")
        checks.exact("final_snapshot_seed", snapshot["seed"], seed["seed_integer"])
        for field, expected in (("delta", F(1, 20)), ("min_step", F(1, 1000000)),
                ("initial_width", initial_width), ("total_midpoint", total_midpoint),
                ("sum_z", sum_z), ("sum_width_squared", variance),
                ("observed_gap_sum", sum(observed.values(), F(0))), ("observed_residual_sum", residual_sum)):
            checks.exact("final_exact_snapshot_" + field, F(snapshot[field]), expected)
        checks.exact("final_snapshot_precision", snapshot["precision"], 100)
        checks.exact("final_snapshot_lambda_grid", snapshot["lambda_exponents"], list(range(-6, 3)))
        checks.exact("final_snapshot_LCM_cap", snapshot["max_lcm_bits"], 200000)
        checks.exact("final_snapshot_LCM", snapshot["denominator_lcm"], denominator_lcm)
        checks.exact("final_snapshot_no_pending_label", snapshot["pending"], None)
        checks.exact("complete_snapshot_events", snapshot["events"], events)
        expected_units = [{"identity": key, "lower": str(units[key]["lower"].numerator) + "/" + str(units[key]["lower"].denominator),
                           "upper": str(units[key]["upper"].numerator) + "/" + str(units[key]["upper"].denominator),
                           "curvature": str(units[key]["curvature"].numerator) + "/" + str(units[key]["curvature"].denominator)} for key in sorted(units)]
        checks.exact("snapshot_complete_frozen_units", snapshot["units"], expected_units)
    remaining = last["remaining"]
    checks.check("actual_query_cap_and_remaining_floor", len(queried) <= 384 and len(remaining) >= 128)
    if not last["strict"] and not last["invalid"]:
        checks.check("no_premature_noncertificate_stop", len(queried) == 384
                     or sum((widths[key] for key in remaining), F(0)) == 0)
    checks.exact("frozen_queried_identity_order", adoption["queried_ids"], queried)
    checks.exact("frozen_remaining_identity_order", adoption["remaining_ids"], remaining)
    checks.exact("frozen_final_decision_record", adoption["decision"], prefixes[-1]["decision"])
    checks.check("adoption_after_final_prefix", timepoint(adoption["actual_utc"]) >= timepoint(prefixes[-1]["actual_utc"]))
    checks.exact("frozen_alpha_from_strict_prefix", F(adoption["alpha"]), last["alpha"] if last["strict"] else F(0))
    checks.exact("frozen_strict_predicate", adoption["strict_certificate"], bool(last["strict"] and last["alpha"] > 0))
    checks.exact("frozen_query_count", adoption["query_count"], len(queried))
    checks.exact("frozen_remaining_count", adoption["remaining_count"], len(remaining))
    checks.exact("close_frozen_alpha", F(close["alpha"]), F(adoption["alpha"]))
    checks.exact("close_frozen_strict_certificate", close["strict_certificate"], adoption["strict_certificate"])
    expected_reason = ("first_strict_remaining_certificate" if last["strict"] else "confidence_abstain"
                       if last["invalid"] else "maximum384queries_no_strict_certificate"
                       if len(queried) == 384 else "no_nonzero_width_ticket")
    checks.exact("recorded_exact_stop_reason", adoption["stop_reason"], expected_reason)
    checks.exact("census_adoption_binding", census["decision_freeze_sha256"], adoption_sha)
    checks.exact("census_jsonl_exact_match", census["records"], census_rows)
    checks.exact("census_frozen_remaining_order", [row["component_id"] for row in census_rows], remaining)
    target_lookup = {row["component_id"]: row for row in truths["records"]}
    checks.exact("target_table_science_binding", truths["plan_sha256"], science_sha)
    checks.exact("target_table_forecast_binding", truths["forecast_freeze_sha256"], forecast_sha)
    checks.exact("target_table_frozen_adoption_binding", truths["decision_freeze_sha256"], adoption_sha)
    checks.check("512_unique_closed_targets", len(truths["records"]) == len(target_lookup) == 512
                 and set(target_lookup) == set(laws))
    for key, row in target_lookup.items():
        checks.exact("truth_identity_binding", row["identity"], laws[key]["identity"])
        checks.exact("truth_frozen_remaining_role", row["role_at_alpha_freeze"], "queried" if key in observed else "remaining")
    for query in query_rows:
        checks.exact("query_target_closure", query["native_target"], target_lookup[query["component_id"]]["native_target"])
    for row in census_rows:
        key, permit = row["component_id"], row["permit"]
        checks.exact("census_original_identity", row["identity"], laws[key]["identity"])
        checks.exact("census_target_closure", row["native_target"], target_lookup[key]["native_target"])
        checks.check("actual_adoption_precedes_census_label", timepoint(adoption["actual_utc"])
                     <= timepoint(row["actual_request_utc"]) <= timepoint(row["actual_return_utc"]))
        checks.exact("census_permit_phase", permit["phase"], "remaining_census")
        checks.exact("census_permit_science", permit["plan_sha256"], science_sha)
        checks.exact("census_permit_identity", permit["component_id"], key)
        checks.exact("census_permit_forecast", permit["forecast_freeze"]["sha256"], forecast_sha)
        checks.exact("census_permit_seed", permit["seed"]["sha256"], seed_sha)
        checks.exact("census_permit_adoption", permit["decision"]["sha256"], adoption_sha)
    for field, expected in (("primary_query_targets", len(queried)), ("research_census_targets", len(remaining)),
            ("total_actual_targets", 512), ("remaining_count", len(remaining)), ("full_predictive_score_calls", 0),
            ("operational_gap_calculations", len(queried)), ("new_fits", 0), ("new_wet_measurements_or_DFT", 0),
            ("physical_cost_savings_established", False)):
        checks.exact("closed_actual_cost_" + field, close[field], expected)
    return {"close": close, "adoption": adoption, "targets": target_lookup,
            "queried": queried, "remaining": remaining, "alpha": F(adoption["alpha"]),
            "strict": adoption["strict_certificate"], "guard_rejected": guard_rejected}


def own_metrics(model, target, helper):
    target = helper.binary(target)
    score = helper.integral_crps(model, target)
    helper.require(score == helper.energy_crps(model, target), "Every saved score CDF/energy agreement")
    cumulative, quantiles = F(0), {}
    levels = (F(1, 40), F(1, 20), F(1, 10), F(1, 2), F(9, 10), F(19, 20), F(39, 40))
    for point, weight in model:
        cumulative += weight
        for level in levels:
            if level not in quantiles and cumulative >= level:
                quantiles[level] = point
    median = quantiles[F(1, 2)]
    intervals = {}
    for label, lo, hi in (("0.8", F(1, 10), F(9, 10)), ("0.9", F(1, 20), F(19, 20)), ("0.95", F(1, 40), F(39, 40))):
        a, b = quantiles[lo], quantiles[hi]
        intervals[label] = {"lower_exact_fraction": str(a), "upper_exact_fraction": str(b),
                            "covered": a <= target <= b, "width_native": float(b - a)}
    return {"CRPS_exact_fraction": str(score), "CRPS_native": float(score),
            "median_native": float(median), "squared_error_native": float((median - target)**2),
            "intervals": intervals}


def aggregate(rows):
    if not rows:
        return None
    n = len(rows)
    return {"groups": n, "CRPS_native_mean": math.fsum(row["CRPS_native"] for row in rows) / n,
            "RMSE_median_native": math.sqrt(math.fsum(row["squared_error_native"] for row in rows) / n),
            "coverage": {z: sum(row["intervals"][z]["covered"] for row in rows) / n for z in ("0.8", "0.9", "0.95")},
            "width_native_mean": {z: math.fsum(row["intervals"][z]["width_native"] for row in rows) / n for z in ("0.8", "0.9", "0.95")}}


def check_aggregate(saved, expected, checks, label):
    if expected is None:
        checks.exact(label + "_empty", saved, None)
        return
    checks.exact(label + "_groups", saved["groups"], expected["groups"])
    for field in ("CRPS_native_mean", "RMSE_median_native"):
        checks.number(label + "_" + field, saved[field], expected[field], 2e-9, 2e-11)
    for level in ("0.8", "0.9", "0.95"):
        checks.exact(label + "_coverage", saved["coverage"][level], expected["coverage"][level])
        checks.number(label + "_width", saved["width_native_mean"][level], expected["width_native_mean"][level], 2e-9, 2e-11)


def verify_scores(plan, helper, checks, laws):
    qroot, sroot = Path(plan["query_root"]), Path(plan["score_root"])
    query_review = read(plan["pins"]["query_review"]["file"])
    score_approval = read(plan["pins"]["score_approval"]["file"])
    start, result = read(sroot / "START.json"), read(sroot / "RESULTS.json")
    adoption = read(qroot / "ADOPTION_FREEZE.json")
    close = read(qroot / "PILOT_CLOSED.json")
    truths = read(qroot / "ALL512_TARGETS.json")
    checks.exact("actual_prior_query_review", query_review["status"], "PASS_INDEPENDENT_REMAINING512_QUERY")
    bindings = {"scientific_plan_sha256": plan["pins"]["science_plan"]["sha256"],
                "forecast_freeze_sha256": sha(Path(plan["forecast_root"]) / "FORECAST_FREEZE.json"),
                "pilot_close_sha256": sha(qroot / "PILOT_CLOSED.json")}
    for key, expected in bindings.items():
        checks.exact("query_review_score_cross_binding_" + key, query_review[key], expected)
    checks.exact("actual_fullscore_root_approval", score_approval["approved"], True)
    checks.exact("fullscore_root_plan", score_approval["plan_sha256"], bindings["scientific_plan_sha256"])
    checks.exact("fullscore_root_actual_query_review", score_approval["state_review"]["sha256"], plan["pins"]["query_review"]["sha256"])
    checks.check("fullscore_root_allowed_phase", "score_closed_pilot" in score_approval["allowed_phases"])
    checks.check("first_fullscore_after_actual_query_audit", timepoint(start["actual_utc"]) >= timepoint(query_review["actual_end_utc"]))
    checks.exact("fullscore_start_review", start["state_query_review_sha256"], plan["pins"]["query_review"]["sha256"])
    checks.exact("score_result_status", result["status"], "COMPLETE_REMAINING_RISK_FROZEN_REFERENCE_PILOT")
    checks.exact("score_result_plan", result["plan_sha256"], bindings["scientific_plan_sha256"])
    for key in ("forecast_freeze_sha256", "pilot_close_sha256"):
        checks.exact("score_result_cross_binding_" + key, result[key], bindings[key])
    checks.exact("score_result_query_review", result["state_query_audit_sha256"], plan["pins"]["query_review"]["sha256"])
    alpha = F(adoption["alpha"])
    remaining, queried = set(adoption["remaining_ids"]), set(adoption["queried_ids"])
    targets = {row["component_id"]: row for row in truths["records"]}
    checks.exact("scored_frozen_identity_order", [row["component_id"] for row in result["records"]], list(laws))
    own, gap_sum, candidate_gap = {}, F(0), F(0)
    for record in result["records"]:
        key = record["component_id"]
        f, g = helper.atoms(laws[key]["F_law"]), helper.atoms(laws[key]["G_law"])
        mix = {}
        for model, weight in ((f, 1 - alpha), (g, alpha)):
            for point, probability in model:
                mix[point] = mix.get(point, F(0)) + weight * probability
        h = tuple((x, p) for x, p in sorted(mix.items()) if p)
        target = targets[key]["native_target"]
        scores = {label: own_metrics(model, target, helper) for label, model in (("F", f), ("G", g), ("H", h))}
        role = "remaining" if key in remaining else "queried"
        checks.exact("saved_score_frozen_role", record["role_at_alpha_freeze"], role)
        checks.exact("saved_score_target", record["native_target"], target)
        checks.exact("saved_score_g5", record["g5"], laws[key]["g5"])
        for label, metric in scores.items():
            saved = record["scores"][label]
            checks.exact("all1536_exact_CRPS", F(saved["CRPS_exact_fraction"]), F(metric["CRPS_exact_fraction"]))
            for field in ("CRPS_native", "median_native", "squared_error_native"):
                checks.number("all1536_" + field, saved[field], metric[field], 2e-9, 2e-11)
            for level in ("0.8", "0.9", "0.95"):
                for field in ("lower_exact_fraction", "upper_exact_fraction"):
                    checks.exact("all4608_quantile_" + field, F(saved["intervals"][level][field]), F(metric["intervals"][level][field]))
                checks.exact("all4608_coverage_events", saved["intervals"][level]["covered"], metric["intervals"][level]["covered"])
                checks.number("all4608_interval_widths", saved["intervals"][level]["width_native"], metric["intervals"][level]["width_native"], 2e-9, 2e-11)
        d = F(scores["G"]["CRPS_exact_fraction"]) - F(scores["F"]["CRPS_exact_fraction"])
        actual = F(scores["H"]["CRPS_exact_fraction"]) - F(scores["F"]["CRPS_exact_fraction"])
        b = laws[key]["geometry"]["curvature"]
        checks.exact("all512_exact_partial_mixture_identity", actual, alpha * d - alpha * (1 - alpha) * b)
        if key in remaining:
            gap_sum += actual
            candidate_gap += d
        own[key] = {"role": role, "scores": scores, "g5": record["g5"]}
    for role in ("all512_forecast_diagnostic", "remaining", "queried"):
        for method in ("F", "G", "H"):
            expected = aggregate([row["scores"][method] for row in own.values()
                                  if role == "all512_forecast_diagnostic" or row["role"] == role])
            check_aggregate(result["summaries"][role][method], expected, checks, "saved_summary_" + role + "_" + method)
    for stratum, predicate in (("g_zero", lambda v: v == 0), ("g_positive", lambda v: v > 0)):
        rows = [row for row in own.values() if row["role"] == "remaining" and predicate(row["g5"])]
        saved = result["g0_gpositive_remaining_descriptive"][stratum]
        checks.exact("prespecified_g_stratum_groups", saved["groups"], len(rows))
        checks.exact("prespecified_g_stratum_no_weight_change", saved["descriptive_only_no_risk_reweighting"], True)
        for method in ("F", "G", "H"):
            check_aggregate(saved["summaries"][method], aggregate([row["scores"][method] for row in rows]), checks,
                            "saved_stratum_" + stratum + "_" + method)
    primary = gap_sum / len(remaining)
    checks.exact("primary_remaining_sum_exact", F(result["primary_remaining_H_minus_F_sum_exact"]), gap_sum)
    checks.exact("primary_remaining_mean_exact", F(result["primary_remaining_H_minus_F_mean_exact"]), primary)
    checks.number("primary_remaining_mean_native", result["primary_remaining_H_minus_F_mean_native"], float(primary), 2e-9, 2e-11)
    checks.exact("completion_query_truth_gap0_normalization", F(result["completion_normalized512_gap_exact"]), gap_sum / 512)
    checks.number("completion_normalized512_native", result["completion_normalized512_gap_native"], float(gap_sum / 512), 2e-9, 2e-11)
    checks.number("remaining_raw_candidate_diagnostic", result["remaining_G_minus_F_mean_native"], float(candidate_gap / len(remaining)), 2e-9, 2e-11)
    bound = F(adoption["decision"].get("risk_upper_sum", "0"))
    strict = bool(adoption["strict_certificate"] and alpha > 0)
    for field, expected in (("strict_certificate", strict), ("certificate_actual_remaining_bound_violation", bool(strict and gap_sum > bound)),
            ("pilot_effective_increment", bool(strict and alpha > 0 and primary < 0)),
            ("alpha0_is_reference_preservation_not_improvement", alpha == 0),
            ("queried_cells_both_completions_use_actual_truth_pointmass_CRPS_gap0", True),
            ("all512_predictive_summaries_are_retrospective_not_completion_primary", True),
            ("remaining_population_fixed_before_census", True), ("score_positions", 1536),
            ("remaining_groups", len(remaining)), ("queried_groups", len(queried)),
            ("actual_primary_query_targets", len(queried)), ("actual_research_census_targets", len(remaining)),
            ("total_actual_target_cells", 512), ("new_fits_or_optimizers", 0), ("new_wet_measurements_or_DFT", 0),
            ("actual_physical_experiment_savings_measured", False)):
        checks.exact("scored_scope_" + field, result[field], expected)
    checks.exact("score_common_alpha", F(result["alpha_exact_fraction"]), alpha)
    checks.exact("score_certificate_bound", F(result["certificate_risk_upper_sum_exact"]), bound)
    return {"alpha_exact_fraction": str(alpha), "remaining_groups": len(remaining),
            "queried_groups": len(queried), "primary_remaining_mean_exact": str(primary),
            "strict_certificate": strict, "actual_bound_violation": bool(strict and gap_sum > bound),
            "pilot_effective_increment": bool(strict and alpha > 0 and primary < 0)}


def validate_plan(plan_path, phase, approval):
    if sys.get_int_max_str_digits() != 0:
        raise ValueError("Register and run with -X int_max_str_digits=0")
    plan = read(plan_path)
    expected_status = {"forecast": "REGISTERED_REMAINING512_FORECAST_AUDIT",
                       "query": "REGISTERED_REMAINING512_QUERY_AUDIT",
                       "scores": "REGISTERED_REMAINING512_SCORE_AUDIT"}[phase]
    expected_approval = {"forecast": "ROOT_REMAINING512_FORECAST_AUDIT_ONCE",
                         "query": "ROOT_REMAINING512_QUERY_AUDIT_ONCE",
                         "scores": "ROOT_REMAINING512_SCORE_AUDIT_ONCE"}[phase]
    if plan["status"] != expected_status or approval != expected_approval:
        raise ValueError("Exact root phase approval required")
    if sha(__file__) != plan["audit_source_sha256"]:
        raise ValueError("Frozen audit source changed")
    for section in ("pins", "helpers", "artifacts"):
        if not plan[section]:
            raise ValueError("Actual nonempty frozen " + section + " pins required")
        for pin in plan[section].values():
            if sha(pin["file"]) != pin["sha256"]:
                raise ValueError("Registered " + section + " artifact changed")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "query", "scores"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = validate_plan(args.plan, args.phase, args.approval)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    write(args.output_dir / "START.json", {"actual_utc": start, "phase": args.phase,
          "plan_sha256": sha(args.plan), "audit_source_sha256": sha(__file__),
          "new_fits": 0, "new_target_queries": 0, "SUT_imports": 0})
    checks = Checks()
    try:
        mathpin = plan["helpers"]["math"]
        specification = importlib.util.spec_from_file_location("_remaining512_independent_math", mathpin["file"])
        helper = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = helper
        specification.loader.exec_module(helper)
        science = read(plan["pins"]["science_plan"]["file"])
        checks.exact("registered_science_scope", science["scope"], "REMAINING_RISK_MATERIAL_V1")
        checks.exact("registered_science_status", science["status"], "REGISTERED")
        for section in ("inputs", "dependencies"):
            for pin in science[section].values():
                checks.exact("immutable_scientific_" + section, sha(pin["file"]), pin["sha256"])
        for name, digest in science["source_sha256"].items():
            checks.exact("immutable_producer_source", sha(Path(plan["producer_code_root"]) / name), digest)
        if args.phase == "forecast":
            freeze, laws, detail = verify_forecasts(plan, science, helper, checks)
            result = {"status": "PASS_INDEPENDENT_REMAINING512_FORECASTS",
                      "laws_audited": 512, "trees_audited": 256,
                      "exact_tree_routes": 256 * (296 + 512), **detail}
        elif args.phase == "query":
            freeze, laws, prior = load_forecast_after_review(plan, checks)
            detail = verify_queries(plan, science, helper, checks, freeze, laws)
            result = {"status": "PASS_INDEPENDENT_REMAINING512_QUERY",
                      "all_prefixes_audited": len(detail["queried"]) + 1,
                      "queried_groups": len(detail["queried"]),
                      "remaining_groups": len(detail["remaining"]),
                      "alpha_exact_fraction": str(detail["alpha"]),
                      "strict_certificate": detail["strict"],
                      "prelabel_LCM_guard_rejected": detail["guard_rejected"],
                      "actual_total_report_targets": 512,
                      "first_full_predictive_scores_performed": 0}
        else:
            freeze, laws, prior = load_forecast_after_review(plan, checks)
            detail = verify_scores(plan, helper, checks, laws)
            result = {"status": "PASS_INDEPENDENT_REMAINING512_SCORES",
                      "proper_scores_audited": 1536, "coverage_events_audited": 4608, **detail}
        result.update({"scientific_plan_sha256": plan["pins"]["science_plan"]["sha256"],
                       "forecast_freeze_sha256": sha(Path(plan["forecast_root"]) / "FORECAST_FREEZE.json")})
        if args.phase != "forecast":
            result["pilot_close_sha256"] = sha(Path(plan["query_root"]) / "PILOT_CLOSED.json")
    except Exception as error:
        result = {"status": "FAIL", "phase": args.phase,
                  "error_type": type(error).__name__, "error": str(error)}
    result.update({"actual_start_utc": start, "actual_end_utc": datetime.now(timezone.utc).isoformat(),
                   "audit_plan_sha256": sha(args.plan), "audit_source_sha256": sha(__file__),
                   "helper_source_sha256": plan["helpers"]["math"]["sha256"],
                   "counts": checks.counts, "maximum_absolute_errors": checks.errors,
                   "SUT_imports": 0, "new_fits": 0, "new_target_queries": 0,
                   "no_raw_source_decode": True,
                   "independence_scope": "Saved forecast/controller audit by a different author; not an independent review of this agent's production core. Separate core QA is required.",
                   "limitations": ["Historical RF bootstrap/training evidence is limited to the immutable saved source and prior review",
                       "One prospective epoch cannot empirically establish95% coverage",
                       "Seeded PRNG is a recorded implementation under an ideal random-ticket theorem",
                       "All queried and census report accesses count; no wet/DFT savings measured"]})
    name = {"forecast": "INDEPENDENT_FORECAST_AUDIT.json", "query": "INDEPENDENT_QUERY_AUDIT.json",
            "scores": "INDEPENDENT_SCORE_AUDIT.json"}[args.phase]
    destination = args.output_dir / name
    write(destination, result)
    print(json.dumps({"status": result["status"], "output": str(destination), "sha256": sha(destination)}))
    if not result["status"].startswith("PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
