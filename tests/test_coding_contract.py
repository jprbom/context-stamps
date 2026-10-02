from pathlib import Path

import pytest

from context_stamps import compile_coding_contract


def test_exact_paths_and_staleness(tmp_path):
    source = tmp_path / "input" / "data.csv"
    source.parent.mkdir()
    source.write_text("x\n1\n", encoding="utf-8")
    contract = compile_coding_contract(tmp_path, ("input/data.csv",), ("result.py",),
                                       runtime_root="/app/project")
    assert contract.verify(tmp_path)
    assert "INPUT /app/project/input/data.csv" in contract.render()
    assert contract.check_static_paths("p='/app/project/input/data.csv'\n") == ()
    assert contract.check_static_paths("p='/app/input/data.csv'\n") == ("/app/input/data.csv",)
    assert contract.check_static_paths("p='input/data.csv'\n") == ("input/data.csv",)
    source.write_text("x\n2\n", encoding="utf-8")
    assert not contract.verify(tmp_path)


@pytest.mark.parametrize("path", ("../secret", "/tmp/secret", "a//b", "a/./b", "a\\b"))
def test_bad_source_paths(tmp_path, path):
    with pytest.raises(ValueError):
        compile_coding_contract(tmp_path, (path,), ("out.py",), runtime_root="/app/project")


def test_symlinked_parent_refused(tmp_path):
    outside = tmp_path.parent / "outside-contract-test"
    outside.mkdir(exist_ok=True)
    (outside / "source.py").write_text("pass\n", encoding="utf-8")
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink"):
        compile_coding_contract(tmp_path, ("linked/source.py",), ("out.py",),
                                runtime_root="/app/project")


def test_outputs_are_explicit_and_static_check_is_narrow(tmp_path):
    Path(tmp_path / "in.py").write_text("pass\n", encoding="utf-8")
    contract = compile_coding_contract(tmp_path, ("in.py",), ("out.py",), runtime_root="/app")
    assert contract.check_static_paths("x='/app/out.py'\ny='/tmp/chart.png'\n") == ()
    assert contract.check_static_paths("x='/app/other.py'\n") == ("/app/other.py",)
    assert contract.check_static_paths("x = get_path()\n") == ()  # dynamic path remains unproven
    with pytest.raises(ValueError, match="overlaps"):
        compile_coding_contract(tmp_path, ("in.py",), ("in.py",), runtime_root="/app")
