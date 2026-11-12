import pytest

from tool_agent.tools import ToolError, calculate, run_tool, search_notes


def test_calculate_respects_precedence():
    assert calculate("2 + 3 * 4") == "14"


def test_calculate_handles_parentheses_and_division():
    assert calculate("(12 + 8) * 3 / 4") == "15.0"


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('echo hi')", "open('x')", "a + 1", "[1, 2]", "True + 1"],
)
def test_calculate_rejects_anything_but_arithmetic(expr):
    with pytest.raises(ToolError):
        calculate(expr)


def test_calculate_division_by_zero_is_a_tool_error():
    with pytest.raises(ToolError, match="division by zero"):
        calculate("1 / 0")


def test_calculate_rejects_huge_exponent():
    with pytest.raises(ToolError, match="exponent too large"):
        calculate("9 ** 9999")


def test_calculate_rejects_syntax_errors():
    with pytest.raises(ToolError, match="invalid expression"):
        calculate("2 +")


def test_search_notes_returns_file_and_line(tmp_path):
    (tmp_path / "a.md").write_text("first line\nPCA reduces dimensions\n")
    assert search_notes("pca", tmp_path) == "a.md:2: PCA reduces dimensions"


def test_search_notes_reports_no_matches(tmp_path):
    (tmp_path / "a.md").write_text("nothing relevant\n")
    assert search_notes("zebra", tmp_path) == "no matches"


def test_search_notes_caps_results(tmp_path):
    (tmp_path / "a.md").write_text("\n".join(f"match {i}" for i in range(20)))
    assert len(search_notes("match", tmp_path).splitlines()) == 5


def test_search_notes_rejects_empty_query(tmp_path):
    with pytest.raises(ToolError):
        search_notes("   ", tmp_path)


def test_run_tool_unknown_name(tmp_path):
    with pytest.raises(ToolError, match="unknown tool"):
        run_tool("delete_everything", {}, tmp_path)


def test_run_tool_missing_argument(tmp_path):
    with pytest.raises(ToolError, match="bad input"):
        run_tool("calculate", {}, tmp_path)
