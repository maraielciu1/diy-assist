from dataclasses import dataclass


@dataclass
class GuardrailDecision:
    allow: bool
    reason: str | None = None
    escalation_message: str | None = None


HAZARD_KEYWORDS = {
    "gas leak",
    "smell gas",
    "burning smell",
    "smells burnt",
    "burnt smell",
    "burning rubber",
    "sparking",
    "exposed wire",
    "electrocution",
    "electric shock",
    "got shocked",
    "refrigerant leak",
    "smoke from appliance",
}


def evaluate_query_safety(user_query: str) -> GuardrailDecision:
    lowered = user_query.lower()
    for keyword in HAZARD_KEYWORDS:
        if keyword in lowered:
            return GuardrailDecision(
                allow=False,
                reason=f"hazard_detected:{keyword}",
                escalation_message=(
                    "This may involve a hazardous condition. Stop troubleshooting and "
                    "contact a qualified technician."
                ),
            )
    return GuardrailDecision(allow=True)
