

URGENCY = {
    "normal": {
        "tier": "ROUTINE",
        "color": "none",
        "response_window": "No action required",
        "headline": "Sensor reading within normal range",
        "actions": [
            "Continue routine monitoring — no leak or burst signature detected.",
            "Log this reading for baseline trend tracking in the Analysis dashboard.",
        ],
    },
    "leak": {
        "tier": "MODERATE",
        "color": "medium",
        "response_window": "Inspect within 48 hours",
        "headline": "Leak signature detected",
        "actions": [
            "Schedule a field inspection at this sensor's location within 48 hours.",
            "Check for surface wetness, soft ground, or unexplained water pooling nearby.",
            "Re-run a reading after inspection to confirm whether the signature persists.",
            "Cross-check the Analysis dashboard for repeated leak flags at this sensor over time.",
        ],
    },
    "burst": {
        "tier": "CRITICAL",
        "color": "severe",
        "response_window": "Immediate response required",
        "headline": "Burst signature detected — urgent",
        "actions": [
            "Dispatch an emergency repair crew immediately.",
            "Isolate the line via nearest control valves if redundant supply paths exist.",
            "Notify downstream customers of a possible pressure drop or service interruption.",
            "Inspect for surface flooding or road subsidence near this sensor — bursts can undermine soil fast.",
        ],
    },
}


def get_recommendation(status: str, sensor_id: str, impact: dict) -> dict:
    """impact comes from utils.inference.estimate_impact() — a heuristic,
    not a model output; framed as such in the rendered text."""
    guidance = URGENCY.get(status, URGENCY["normal"])

    impact_note = ""
    if status != "normal":
        impact_note = (
            f"Estimated impact: {impact['level']} "
            f"(pressure {impact['drop_pct']}% below this sensor's own baseline of {impact['baseline_pressure']} bar — "
            f"a heuristic estimate, not a model prediction, since this dataset has no measured water-loss values)."
        )

    return {
        "tier": guidance["tier"],
        "color": guidance["color"],
        "response_window": guidance["response_window"],
        "headline": guidance["headline"],
        "location_text": f"Sensor {sensor_id}",
        "loss_note": impact_note,
        "actions": guidance["actions"],
    }
