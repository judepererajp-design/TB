from types import SimpleNamespace


def test_validator_issue_text_returns_fallback_when_issue_list_empty():
    from signals.aggregator import _validator_issue_text

    assert _validator_issue_text(SimpleNamespace(issues=[])) == "validator flagged signal"


def test_validator_issue_text_uses_first_non_empty_issue():
    from signals.aggregator import _validator_issue_text

    assert _validator_issue_text(SimpleNamespace(issues=["Bad zone entry", "Other"])) == "Bad zone entry"
