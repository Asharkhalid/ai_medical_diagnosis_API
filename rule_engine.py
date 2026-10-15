import re

# FR-09: combine ML predictions with predefined medical rules. Rules are
# matched against the ExpertRule.condition string, e.g. "fever + rash" or
# "chest pain, shortness of breath" — split on '+'/',' and normalized to the
# same underscored symptom tokens extract_symptoms() produces.
CRITICAL_ACTION = "Flag Critical"


def _parse_condition_symptoms(condition):
    parts = re.split(r"[+,]", condition or "")
    return [p.strip().lower().replace(" ", "_") for p in parts if p.strip()]


def apply_expert_rules(extracted_symptoms, prediction, rules):
    """
    Applies the highest-priority ExpertRule whose condition symptoms are all
    present in extracted_symptoms. `rules` must already be ordered by
    priority descending (FR-09: "resolve conflicts using rule priority") —
    only the first match wins.

    Returns (prediction, applied_rule_info | None). A "Flag Critical" match
    overrides the predicted disease with the rule's target; other actions
    (e.g. "Increase Priority", "Require Tests") annotate the result without
    changing the disease.
    """
    symptom_set = set(extracted_symptoms)

    for rule in rules:
        condition_symptoms = _parse_condition_symptoms(rule.condition)
        if condition_symptoms and set(condition_symptoms).issubset(symptom_set):
            applied_rule_info = {
                "condition": rule.condition,
                "target": rule.target,
                "action": rule.action,
                "priority": rule.priority,
            }

            if rule.action == CRITICAL_ACTION:
                prediction = {**prediction, "disease": rule.target}

            return prediction, applied_rule_info

    return prediction, None
