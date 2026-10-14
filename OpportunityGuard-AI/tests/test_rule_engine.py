"""
tests/test_rule_engine.py — Tests for all rule-based checks.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rule_engine import (
    check_missing_employer_info,
    check_upfront_payment,
    check_unrealistic_claims,
    check_deadline_consistency,
    check_contact_consistency,
    run_all_rules,
)


# ─── Missing employer info ─────────────────────────────────────────────────────
class TestMissingEmployerInfo:
    def test_all_present_no_warning(self):
        listing = {
            "company_name": "Acme", "company_profile": "A company",
            "website": "https://acme.com", "email": "hr@acme.com",
        }
        assert check_missing_employer_info(listing) == []

    def test_all_missing_high_severity(self):
        warnings = check_missing_employer_info({})
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "high"
        assert warnings[0]["code"] == "missing_employer_info"

    def test_one_missing_low_severity(self):
        listing = {
            "company_name": "Acme", "company_profile": "A company",
            "website": "https://acme.com", "email": "",  "phone": "",
        }
        warnings = check_missing_employer_info(listing)
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "low"

    def test_two_missing_medium_severity(self):
        listing = {"company_name": "Acme", "company_profile": "A company"}
        warnings = check_missing_employer_info(listing)
        assert warnings[0]["severity"] == "medium"


# ─── Upfront payment ──────────────────────────────────────────────────────────
class TestUpfrontPayment:
    def test_clean_listing_no_warning(self):
        listing = {"description": "Build great software with us. No prior experience required."}
        assert check_upfront_payment(listing) == []

    def test_registration_fee_detected(self):
        listing = {"description": "Pay a registration fee of $50 to apply."}
        warnings = check_upfront_payment(listing)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "upfront_payment_language"
        assert warnings[0]["severity"] == "high"

    def test_security_deposit_detected(self):
        listing = {"requirements": "A security deposit is required before joining."}
        assert len(check_upfront_payment(listing)) == 1

    def test_training_fee_detected(self):
        listing = {"description": "A training fee of $100 applies to all new hires."}
        assert len(check_upfront_payment(listing)) == 1

    def test_multiple_matches_single_warning(self):
        listing = {"description": "Pay a registration fee and a security deposit."}
        warnings = check_upfront_payment(listing)
        assert len(warnings) == 1
        evidence = warnings[0]["evidence"]
        assert "registration fee" in evidence
        assert "security deposit" in evidence

    def test_legitimate_payment_not_flagged(self):
        # "payment" alone should NOT trigger; need specific patterns
        listing = {"description": "We process salary payments on the 1st of each month."}
        assert check_upfront_payment(listing) == []

    # ─── Negative examples (SHOULD NOT trigger when fee is explicitly denied) ──
    @pytest.mark.parametrize("disclaimer", [
        "No registration fee is required.",
        "There is no fee to apply.",
        "We do not charge candidates any fee.",
        "Candidates are not required to make any payment.",
        "No payment is required at any stage of recruitment.",
        "The application process is completely free.",
        "There is no joining fee.",
        "We are hiring a Software Development Intern. No registration fee or payment is required from candidates.",
        "The entire internship is 100% free of charge.",
        "Applying is completely free to apply without paying any fee.",
    ])
    def test_negation_disclaimers_do_not_trigger(self, disclaimer):
        listing = {
            "title": "Software Development Intern",
            "description": f"Join our engineering team to build scalable services. {disclaimer}",
        }
        warnings = check_upfront_payment(listing)
        assert warnings == [], f"Expected no warning for negated disclaimer: '{disclaimer}', got: {warnings}"

    # ─── Positive examples (SHOULD trigger) ────────────────────────────────────
    @pytest.mark.parametrize("demand", [
        "Pay a registration fee of ₹999 to activate your account.",
        "Candidates must pay a processing fee before the interview.",
        "Send ₹2,000 to secure your internship.",
        "Pay a refundable security deposit before joining.",
        "Purchase our mandatory training package to receive your offer letter.",
        "Candidates are required to pay a registration fee before starting.",
    ])
    def test_actual_payment_demands_trigger(self, demand):
        listing = {
            "title": "Data Analyst Intern",
            "description": f"Exciting internship role. {demand}",
        }
        warnings = check_upfront_payment(listing)
        assert len(warnings) == 1, f"Expected upfront payment warning for demand: '{demand}'"
        assert warnings[0]["code"] == "upfront_payment_language"
        assert warnings[0]["severity"] == "high"

    # ─── Multi-sentence test cases ─────────────────────────────────────────────
    def test_denial_in_sentence_one_does_not_suppress_demand_in_sentence_two(self):
        listing = {
            "description": (
                "No registration fee is required for the initial application. "
                "However, candidates must pay a processing fee before the interview."
            )
        }
        warnings = check_upfront_payment(listing)
        assert len(warnings) == 1
        assert "processing fee" in warnings[0]["evidence"]

    def test_compound_sentence_contrastive_clause_triggers(self):
        listing = {
            "description": (
                "We do not charge an application fee, "
                "but you must pay a registration fee of ₹999 to activate your account."
            )
        }
        warnings = check_upfront_payment(listing)
        assert len(warnings) == 1
        assert "registration fee" in warnings[0]["evidence"]


# ─── Unrealistic claims ────────────────────────────────────────────────────────
class TestUnrealisticClaims:
    def test_clean_listing_no_warning(self):
        listing = {"description": "Join our team and grow your skills."}
        assert check_unrealistic_claims(listing) == []

    def test_guaranteed_earnings_detected(self):
        listing = {"description": "Guaranteed income of $5000 per week."}
        warnings = check_unrealistic_claims(listing)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "unrealistic_earnings_claim"

    def test_passive_income_detected(self):
        listing = {"benefits": "Earn passive income from home."}
        assert len(check_unrealistic_claims(listing)) == 1

    def test_financial_freedom_detected(self):
        listing = {"description": "Achieve financial freedom in 3 months."}
        assert len(check_unrealistic_claims(listing)) == 1

    def test_earn_5000_daily_working_30_minutes_detected(self):
        listing = {
            "description": "Earn ₹5,000 every day by working for 30 minutes with no skills required."
        }
        warnings = check_unrealistic_claims(listing)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "unrealistic_earnings_claim"

    def test_guaranteed_income_for_every_applicant_detected(self):
        listing = {"description": "Guaranteed income for every applicant."}
        warnings = check_unrealistic_claims(listing)
        assert len(warnings) == 1
        assert warnings[0]["code"] == "unrealistic_earnings_claim"

    def test_competitive_salary_legitimate_not_flagged(self):
        listing = {
            "description": "Competitive salary based on experience and qualifications.",
            "salary_claim": "Competitive, negotiable",
        }
        assert check_unrealistic_claims(listing) == []

    def test_salary_negotiable_based_on_interview_not_flagged(self):
        listing = {
            "description": "Salary is negotiable based on the interview.",
            "salary_claim": "Negotiable",
        }
        assert check_unrealistic_claims(listing) == []

    def test_company_does_not_guarantee_earnings_negated(self):
        listing = {
            "description": (
                "Work on client projects. Please note that our company does not guarantee earnings. "
                "All compensation is based on billable hours."
            )
        }
        assert check_unrealistic_claims(listing) == []


# ─── Deadline checks ──────────────────────────────────────────────────────────
class TestDeadlineConsistency:
    def test_future_deadline_no_warning(self):
        listing = {"deadline": "2099-12-31"}
        assert check_deadline_consistency(listing) == []

    def test_missing_deadline_low_warning(self):
        warnings = check_deadline_consistency({"deadline": ""})
        assert len(warnings) == 1
        assert warnings[0]["code"] == "missing_deadline"
        assert warnings[0]["severity"] == "low"

    def test_past_deadline_medium_warning(self):
        warnings = check_deadline_consistency({"deadline": "2000-01-01"})
        assert len(warnings) == 1
        assert warnings[0]["code"] == "expired_deadline"
        assert warnings[0]["severity"] == "medium"

    def test_invalid_deadline_format(self):
        warnings = check_deadline_consistency({"deadline": "not-a-date"})
        assert len(warnings) == 1
        assert warnings[0]["code"] == "invalid_deadline"


# ─── Contact consistency ──────────────────────────────────────────────────────
class TestContactConsistency:
    def test_matching_domain_no_warning(self):
        listing = {"email": "hr@acme.com", "website": "https://acme.com"}
        assert check_contact_consistency(listing) == []

    def test_free_email_domain_low_warning(self):
        listing = {"email": "acme.jobs@gmail.com", "website": "https://acme.com"}
        warnings = check_contact_consistency(listing)
        assert any(w["code"] == "free_email_domain" for w in warnings)

    def test_domain_mismatch_medium_warning(self):
        listing = {"email": "hr@otherdomain.com", "website": "https://acme.com"}
        warnings = check_contact_consistency(listing)
        assert any(w["code"] == "email_website_domain_mismatch" for w in warnings)

    def test_invalid_email_format_warning(self):
        listing = {"email": "not-an-email", "website": ""}
        warnings = check_contact_consistency(listing)
        assert any(w["code"] == "invalid_email_format" for w in warnings)

    def test_no_email_no_phone_medium_warning(self):
        listing = {"email": "", "phone": ""}
        warnings = check_contact_consistency(listing)
        assert any(w["code"] == "missing_contact_details" for w in warnings)

    def test_www_prefix_handled_correctly(self):
        listing = {"email": "hr@acme.com", "website": "https://www.acme.com"}
        assert check_contact_consistency(listing) == []


# ─── Full rule runner ─────────────────────────────────────────────────────────
class TestRunAllRules:
    def test_clean_listing_zero_rule_score(self):
        listing = {
            "company_name": "Acme", "company_profile": "A company",
            "website": "https://acme.com", "email": "hr@acme.com", "phone": "+1234",
            "description": "Join our team of engineers. No registration fee is required.",
            "requirements": "Python skills",
            "benefits": "Health insurance", "title": "Software Engineer",
            "deadline": "2099-01-01",
        }
        result = run_all_rules(listing)
        assert result["rule_score"] == 0
        assert result["warnings"] == []

    def test_result_has_required_keys(self):
        result = run_all_rules({})
        assert "warnings" in result
        assert "rule_score" in result

    def test_rule_score_capped_at_100(self):
        listing = {
            "description": "Pay registration fee of $100. Guaranteed income. No experience.",
            "email": "not-an-email",
            "deadline": "2000-01-01",
        }
        result = run_all_rules(listing)
        assert result["rule_score"] <= 100

    def test_rule_score_positive_for_suspicious_listing(self):
        listing = {
            "description": "Pay a registration fee of $100. Guaranteed earnings of $5000/week!",
        }
        result = run_all_rules(listing)
        assert result["rule_score"] > 0
