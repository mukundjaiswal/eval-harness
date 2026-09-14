import pytest

from eval_harness.case import EvalCase, load_cases
from eval_harness.exceptions import DatasetError


def write(tmp_path, body, name="v1.jsonl"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_inputs(tmp_path):
    path = write(tmp_path, '{"input": "a"}\n{"input": "b"}\n')
    assert [c.input for c in load_cases(path)] == ["a", "b"]


def test_custom_input_key_keeps_an_existing_dataset_format(tmp_path):
    """A project should not have to rewrite its datasets to adopt the harness."""
    path = write(tmp_path, '{"question": "How is iron absorbed?"}\n')
    assert load_cases(path, input_key="question")[0].input == "How is iron absorbed?"


def test_custom_reference_key(tmp_path):
    path = write(tmp_path, '{"input": "a", "summary": "gold"}\n')
    assert load_cases(path, reference_key="summary")[0].reference == "gold"


def test_blank_lines_are_skipped(tmp_path):
    path = write(tmp_path, '{"input": "a"}\n\n\n{"input": "b"}\n')
    assert len(load_cases(path)) == 2


def test_reference_is_optional(tmp_path):
    path = write(tmp_path, '{"input": "a"}\n')
    assert load_cases(path)[0].reference is None


def test_unknown_keys_survive_as_metadata(tmp_path):
    path = write(tmp_path, '{"input": "a", "slice": "tables"}\n')
    assert load_cases(path)[0].metadata == {"slice": "tables"}


def test_slice_reads_metadata_with_a_default():
    case = EvalCase(input="a", metadata={"slice": "tables"})
    assert case.slice("slice") == "tables"
    assert case.slice("missing") == "all"


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_cases(tmp_path / "absent.jsonl")


def test_malformed_line_fails_the_load_rather_than_being_skipped(tmp_path):
    """Dropping a case silently changes the denominator of every metric."""
    path = write(tmp_path, '{"input": "a"}\nnot json\n')
    with pytest.raises(DatasetError, match=":2"):
        load_cases(path)


def test_non_object_line_is_rejected(tmp_path):
    path = write(tmp_path, '["a"]\n')
    with pytest.raises(DatasetError, match="not a JSON object"):
        load_cases(path)


def test_missing_input_key_names_the_line_and_the_key(tmp_path):
    path = write(tmp_path, '{"reference": "gold"}\n')
    with pytest.raises(DatasetError, match="'input'"):
        load_cases(path)


def test_empty_dataset_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="no cases"):
        load_cases(write(tmp_path, "\n\n"))
