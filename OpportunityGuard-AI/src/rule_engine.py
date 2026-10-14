"""
rule_engine.py — Deterministic rule checks for OpportunityGuard AI.

Each check returns a list of Warning dicts:
  {code, severity, points, field, evidence, recommendation}

Total rule_score is the sum of points, capped at 100.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

# ─── Severity points ────────────────────────────────────────────────────────────
SEVERITY_POINTS = {"critical": 50, "high": 35, "medium": 20, "low": 10}

# ─── Upfront-payment keywords ───────────────────────────────────────────────────
PAYMENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bregistration\s*fee[s]?\b", "registration fee"),
    (r"\bsecurity\s*deposit[s]?\b", "security deposit"),
    (r"\btraining\s*(?:fee[s]?|package|cost)\b", "training fee"),
    (r"\bapplication\s*fee[s]?\b", "application fee"),
    (r"\badvance\s*pay(?:ment)?\b", "advance payment"),
    (r"\bpay\s+before\s+(?:joining|interview|starting)\b", "pay before joining/interview"),
    (r"\brefundable\s*deposit[s]?\b", "refundable deposit"),
    (r"\bprocessing\s*fee[s]?\b", "processing fee"),
    (r"\badmin(?:istration)?\s*fee[s]?\b", "administration fee"),
    (r"\bjoining\s*fee[s]?\b", "joining fee"),
    (r"\bpay\s+to\s+(?:apply|secure|get|activate)\b", "pay to apply/secure/get"),
    (r"\b(?:candidates?\s+must\s+pay|pay\s+(?:a|an|the))\s+(?:processing|registration|training|application|admin)?\s*fee[s]?\b", "mandatory fee payment"),
    (r"\b(?:send|transfer|pay)\s+(?:[₹$€£]|rs\.?|inr)?\s*[\d,]+\s+to\s+(?:secure|activate|confirm|get)\b", "send money to secure offer"),
    (r"\b(?:purchase|buy|pay\s+for)\s+(?:our\s+)?(?:mandatory\s+)?training\s+package\b", "purchase mandatory training package"),
    (r"\bmandatory\s+training\s+package\b", "mandatory training package"),
]

# Explicit denial patterns that negate fee/payment demands in a clause
PAYMENT_DENIAL_PATTERNS: list[str] = [
    # "No registration fee is required", "No payment is required", "No fee to apply", "No joining fee"
    r"\bno\s+(?:registration|application|processing|training|joining|hidden|upfront|additional)?\s*(?:fee[s]?|payment[s]?|deposit[s]?|charge[s]?|cost[s]?)\s*(?:is|are)?\s*(?:required|needed|charged|demanded|asked|collected|to\s+apply)?\b",
    # "There is no fee to apply", "there is no joining fee", "there are no charges"
    r"\bthere\s+(?:is|are)\s+no\s+(?:registration|application|processing|training|joining|hidden|upfront|additional)?\s*(?:fee[s]?|payment[s]?|deposit[s]?|charge[s]?|cost[s]?)\b",
    # "We do not charge candidates any fee", "we never ask candidates for money", "don't charge any fee"
    r"\b(?:we\s+)?(?:do\s+not|don['’]t|never|does\s+not|doesn['’]t|shall\s+not|will\s+not)\s+(?:charge|ask|demand|request|require|collect)\s+(?:candidates?\s+|applicants?\s+|any\s+)?(?:any\s+)?(?:fee[s]?|payment[s]?|deposit[s]?|charge[s]?|money)\b",
    # "Candidates are not required to make any payment", "not required to pay any fee"
    r"\b(?:candidates?|applicants?|you)\s+are\s+(?:not|never)\s+required\s+to\s+(?:make\s+any\s+payment|pay|deposit|transfer|purchase)\b",
    # "The application process is completely free", "completely free of cost", "100% free"
    r"\b(?:application\s+process\s+is\s+)?(?:completely|totally|entirely|100%|absolutely)\s+free\b",
    r"\bfree\s+of\s+(?:any\s+|all\s+)?(?:cost[s]?|charge[s]?|fee[s]?)\b",
    r"\b(?:free\s+to\s+apply|no\s+cost\s+to\s+apply)\b",
    # "No payment is required at any stage of recruitment"
    r"\bno\s+(?:payment|fee|money)\s+is\s+required\s+(?:at\s+any\s+stage\s+of\s+recruitment)?\b",
    # "without any fee / deposit / payment"
    r"\bwithout\s+(?:paying\s+)?(?:any\s+)?(?:fee[s]?|deposit[s]?|payment[s]?|charge[s]?|cost)\b",
]

# ─── Unrealistic earnings keywords ──────────────────────────────────────────────
UNREALISTIC_PATTERNS: list[tuple[str, str]] = [
    (r"\bguaranteed?\s+(?:earning|income|profit|return|salary)\b", "guaranteed earnings"),
    (r"\bearn\s+(?:up\s+to\s+)?(?:[₹$€£]|rs\.?|inr\s*)?\s*[\d,]+\s*(?:every\s+day|daily|per\s+(?:week|day|hour))\b", "very high per-period earnings claim"),
    (r"\bwork\s+from\s+home\s+earn\b", "work-from-home earnings"),
    (r"\bno\s+(?:experience|skills?)\s+(?:needed|required|necessary)\b", "no experience or skills needed"),
    (r"\bmake\s+money\s+fast\b", "make money fast"),
    (r"\binstant\s+(?:cash|income|payment)\b", "instant cash/payment"),
    (r"\bunlimited\s+(?:income|earning|commission)\b", "unlimited income"),
    (r"\bpassive\s+income\b", "passive income claim"),
    (r"\bfinancial\s+freedom\b", "financial freedom"),
]

# Denial patterns for earnings claims
UNREALISTIC_DENIAL_PATTERNS: list[str] = [
    r"\b(?:do\s+not|don['’]t|does\s+not|doesn['’]t|not|never|no|cannot)\s+guarantee\s+(?:any\s+)?(?:earning|income|profit|return|salary|placement)\b",
    r"\bno\s+guaranteed?\s+(?:earning|income|profit|return|salary)\b",
]

# ─── Email format regex ─────────────────────────────────────────────────────────
EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def _make_warning(
    code: str,
    severity: str,
    field: str,
    evidence: str,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "points": SEVERITY_POINTS.get(severity, 10),
        "field": field,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _split_into_clauses(text: str) -> list[str]:
    """Split text into sentences and contrastive clauses.

    Splits across punctuation boundaries (.!?;\n) as well as contrastive
    conjunctions (but, however, yet, although) so that a disclaimer in one
    clause cannot suppress a payment demand in another.
    """
    sentences = re.split(r"(?<=[.!?;\n])\s+|[\r\n]+", text)
    clauses: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        parts = re.split(
            r",?\s+\b(?:but|however|yet|although|whereas)\b\s*",
            s,
            flags=re.IGNORECASE,
        )
        for p in parts:
            p = p.strip()
            if p:
                clauses.append(p)
    return clauses


# ─── Individual checks ──────────────────────────────────────────────────────────

def check_missing_employer_info(listing: dict[str, Any]) -> list[dict]:
    warnings = []
    missing = []

    if not (listing.get("company_name") or "").strip():
        missing.append("company name")
    if not (listing.get("company_profile") or "").strip():
        missing.append("company profile")
    if not (listing.get("website") or "").strip():
        missing.append("company website")
    if not (listing.get("email") or "").strip() and not (listing.get("phone") or "").strip():
        missing.append("employer contact (email or phone)")

    if len(missing) >= 3:
        severity = "high"
    elif len(missing) == 2:
        severity = "medium"
    elif len(missing) == 1:
        severity = "low"
    else:
        return warnings

    warnings.append(
        _make_warning(
            code="missing_employer_info",
            severity=severity,
            field="company_name / company_profile / website / email / phone",
            evidence=f"Missing: {', '.join(missing)}",
            recommendation=(
                "Search the company name independently, check LinkedIn or official "
                "government registries, and verify the website before proceeding."
            ),
        )
    )
    return warnings


def check_upfront_payment(listing: dict[str, Any]) -> list[dict]:
    """Check description and requirements for upfront-payment language.

    Negation-aware: Explicit denials of fees (e.g. 'No registration fee is required')
    are respected on a per-clause basis. An explicit fee demand anywhere in the
    listing will still trigger the warning.
    """
    raw_text = " . ".join(
        str(listing.get(f, "") or "")
        for f in ("description", "requirements", "benefits", "title")
    )
    if not raw_text.strip():
        return []

    clauses = _split_into_clauses(raw_text)
    matched_phrases: list[str] = []

    for clause in clauses:
        clause_matched: list[str] = []
        for pattern, label in PAYMENT_PATTERNS:
            if re.search(pattern, clause, re.IGNORECASE):
                clause_matched.append(label)

        if not clause_matched:
            continue

        # Check if this clause explicitly denies fees/payments
        is_denied = any(re.search(dp, clause, re.IGNORECASE) for dp in PAYMENT_DENIAL_PATTERNS)
        if not is_denied:
            matched_phrases.extend(clause_matched)

    if not matched_phrases:
        return []

    unique_matched = list(dict.fromkeys(matched_phrases))

    return [
        _make_warning(
            code="upfront_payment_language",
            severity="high",
            field="description / requirements / benefits",
            evidence=f"Matched phrase(s): {', '.join(unique_matched)}",
            recommendation=(
                "Legitimate employers do not charge candidates to apply, register, or join. "
                "Do not pay any fee before independently verifying this employer. "
                "Report suspicious listings to your institution's placement cell."
            ),
        )
    ]


def check_unrealistic_claims(listing: dict[str, Any]) -> list[dict]:
    """Check for unrealistic salary or earnings claims with negation handling."""
    raw_text = " . ".join(
        str(listing.get(f, "") or "")
        for f in ("description", "title", "benefits", "salary_claim", "salary_range")
    )
    if not raw_text.strip():
        return []

    clauses = _split_into_clauses(raw_text)
    matched_phrases: list[str] = []

    for clause in clauses:
        clause_matched: list[str] = []
        for pattern, label in UNREALISTIC_PATTERNS:
            if re.search(pattern, clause, re.IGNORECASE):
                clause_matched.append(label)

        if not clause_matched:
            continue

        # Check for explicit negation (e.g. "Our company does not guarantee earnings")
        is_denied = any(re.search(dp, clause, re.IGNORECASE) for dp in UNREALISTIC_DENIAL_PATTERNS)
        if not is_denied:
            matched_phrases.extend(clause_matched)

    if not matched_phrases:
        return []

    unique_matched = list(dict.fromkeys(matched_phrases))

    return [
        _make_warning(
            code="unrealistic_earnings_claim",
            severity="medium",
            field="description / title / benefits / salary_claim",
            evidence=f"Matched phrase(s): {', '.join(unique_matched)}",
            recommendation=(
                "Compare stated compensation with industry benchmarks for this role and "
                "location. Unusually high or guaranteed earnings with no skill requirements "
                "are a common feature of fraudulent listings."
            ),
        )
    ]


def check_deadline_consistency(listing: dict[str, Any]) -> list[dict]:
    """Check application deadline validity."""
    warnings = []
    deadline_str = (listing.get("deadline") or "").strip()

    if not deadline_str:
        warnings.append(
            _make_warning(
                code="missing_deadline",
                severity="low",
                field="deadline",
                evidence="No application deadline provided.",
                recommendation="Ask the employer for a concrete deadline and application process.",
            )
        )
        return warnings

    try:
        if isinstance(deadline_str, date):
            dl = deadline_str
        else:
            dl = date.fromisoformat(str(deadline_str))
        today = date.today()
        if dl < today:
            warnings.append(
                _make_warning(
                    code="expired_deadline",
                    severity="medium",
                    field="deadline",
                    evidence=f"Deadline {dl} has already passed.",
                    recommendation=(
                        "Contact the employer to confirm whether the position is still open. "
                        "Expired deadlines may indicate an outdated or copied listing."
                    ),
                )
            )
    except (ValueError, TypeError):
        warnings.append(
            _make_warning(
                code="invalid_deadline",
                severity="low",
                field="deadline",
                evidence=f"Could not parse deadline: '{deadline_str}'",
                recommendation="Verify the application deadline directly with the employer.",
            )
        )

    return warnings


def check_contact_consistency(listing: dict[str, Any]) -> list[dict]:
    """Check email format and email-domain vs website-domain alignment."""
    warnings = []
    email = (listing.get("email") or "").strip()
    website = (listing.get("website") or "").strip()

    if email:
        if not EMAIL_RE.match(email):
            warnings.append(
                _make_warning(
                    code="invalid_email_format",
                    severity="medium",
                    field="email",
                    evidence=f"Email '{email}' does not match a standard format.",
                    recommendation="Verify the contact email by reaching the employer through the official website.",
                )
            )
        elif website:
            email_domain = email.split("@")[-1].lower().strip()
            web_domain = re.sub(r"^https?://", "", website.lower()).split("/")[0].strip()
            web_domain_clean = re.sub(r"^www\.", "", web_domain)
            email_domain_clean = re.sub(r"^www\.", "", email_domain)

            FREE_DOMAINS = {
                "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                "live.com", "ymail.com", "protonmail.com", "icloud.com",
            }

            if email_domain_clean in FREE_DOMAINS:
                warnings.append(
                    _make_warning(
                        code="free_email_domain",
                        severity="low",
                        field="email",
                        evidence=f"Contact email uses '{email_domain}', not the company domain.",
                        recommendation=(
                            "Employers at established companies typically use corporate email. "
                            "This is not proof of fraud, but verify the company independently."
                        ),
                    )
                )
            elif email_domain_clean != web_domain_clean:
                warnings.append(
                    _make_warning(
                        code="email_website_domain_mismatch",
                        severity="medium",
                        field="email / website",
                        evidence=(
                            f"Email domain '{email_domain}' does not match "
                            f"website domain '{web_domain}'."
                        ),
                        recommendation=(
                            "Verify that both the email and website belong to the same organisation "
                            "by checking the company's official communications."
                        ),
                    )
                )
    else:
        if not (listing.get("phone") or "").strip():
            warnings.append(
                _make_warning(
                    code="missing_contact_details",
                    severity="medium",
                    field="email / phone",
                    evidence="No email address or phone number provided.",
                    recommendation=(
                        "Legitimate listings include at least one direct contact method. "
                        "Search for the company independently before submitting any application."
                    ),
                )
            )

    return warnings


# ─── Composite rule engine ──────────────────────────────────────────────────────

def run_all_rules(listing: dict[str, Any]) -> dict[str, Any]:
    """Run every rule check and return:

    warnings   — list of individual warning dicts
    rule_score — integer 0-100 representing total rule-based risk
    """
    all_warnings: list[dict] = []
    all_warnings.extend(check_missing_employer_info(listing))
    all_warnings.extend(check_upfront_payment(listing))
    all_warnings.extend(check_unrealistic_claims(listing))
    all_warnings.extend(check_deadline_consistency(listing))
    all_warnings.extend(check_contact_consistency(listing))

    total_points = sum(w["points"] for w in all_warnings)
    rule_score = min(100, max(0, total_points))

    return {
        "warnings": all_warnings,
        "rule_score": rule_score,
    }
