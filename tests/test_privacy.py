from pydantic import JsonValue

from jevpilot.privacy import PrivacyPolicy, redact_json, redact_text


def test_redaction_removes_nested_credentials_without_mutating_input() -> None:
    # Given
    payload: dict[str, JsonValue] = {
        "Authorization": "Bearer hidden",
        "rows": [{"name": "synthetic-secret customer", "token": "another-token"}],
    }
    policy = PrivacyPolicy(secret_values=("synthetic-secret",))

    # When
    actual = redact_json(payload, policy=policy)

    # Then
    assert actual == {
        "Authorization": "[REDACTED]",
        "rows": [{"name": "[REDACTED] customer", "token": "[REDACTED]"}],
    }
    assert payload["Authorization"] == "Bearer hidden"


def test_url_redaction_keeps_origin_and_path_without_credentials() -> None:
    # Given
    url = "https://user:private@example.test:8443/detail?token=hidden#session"

    # When
    actual = redact_text(url, policy=PrivacyPolicy())

    # Then
    assert actual == "https://example.test:8443/detail"


def test_url_redaction_also_removes_query_from_page_text() -> None:
    # Given
    page_text = "Next page: https://example.test/items?token=private-value."

    # When
    actual = redact_text(page_text, policy=PrivacyPolicy())

    # Then
    assert actual == "Next page: https://example.test/items."


def test_redaction_preserves_false_zero_and_null() -> None:
    # Given
    payload: list[JsonValue] = [False, 0, None]

    # When
    actual = redact_json(payload, policy=PrivacyPolicy())

    # Then
    assert actual == [False, 0, None]


def test_policy_representation_does_not_reveal_secret() -> None:
    # Given
    policy = PrivacyPolicy(secret_values=("private-key",))

    # When
    representation = repr(policy)

    # Then
    assert "private-key" not in representation
