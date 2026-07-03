from app.bot import _prepare_agent_message


def test_appends_missing_citations_as_plain_urls() -> None:
    result = _prepare_agent_message(
        "Today it is sunny.",
        ["https://example.com/a", "https://example.com/b"],
    )

    assert "[1](" not in result
    assert "https://example.com/a" in result
    assert "https://example.com/b" in result
    assert "Sources / 来源:" in result


def test_skips_citation_the_model_already_mentioned() -> None:
    result = _prepare_agent_message(
        "Today it is sunny. See https://example.com/a for details.",
        ["https://example.com/a", "https://example.com/b"],
    )

    # Only the URL the model didn't already write should be appended.
    assert result.count("https://example.com/a") == 1
    assert "https://example.com/b" in result


def test_no_footer_when_no_citations() -> None:
    result = _prepare_agent_message("Just a plain reply.", None)

    assert "Sources" not in result
    assert result == "Just a plain reply."


def test_normalizes_bold_and_underscore_markdown() -> None:
    result = _prepare_agent_message("Some **bold** and __underline__ text.", None)

    assert result == "Some *bold* and _underline_ text."
