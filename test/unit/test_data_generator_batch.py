"""Shared batch failure handling without downloaded source-library dependencies."""

from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))


@pytest.fixture(params=["neutron", "photon", "proton"])
def generator_case(request, tmp_path, monkeypatch):
    projectile = request.param
    if projectile != "photon":
        pytest.importorskip("ACEtk")
    generate = import_module(f"data_library_generator.{projectile}.generate")
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    converter = "convert_file"
    if projectile == "photon":
        names = [f"EPDL.ZA{z:03d}000.endf" for z in range(1, 5)]
        selection = ["--elements", "H", "He", "Li", "Be"]
        monkeypatch.setenv("MCDC_EPICS_LIB", str(source))
    else:
        names = ["a.ace", "b.ace", "c.ace", "d.ace"]
        if projectile == "proton":
            converter = "process_ace_file"
            pstar = tmp_path / "pstar"
            pstar.mkdir()
            monkeypatch.setenv("MCDC_ACELIB_PROTON", str(source))
            monkeypatch.setenv("MCDC_PSTAR_LIB", str(pstar))
            monkeypatch.setenv("MCDC_LIB_PROTON", str(output))
            monkeypatch.setattr(generate, "process_pstar_only_file", lambda *args: None)
            selection = ["--only", *names]
        else:
            monkeypatch.setenv("MCDC_ACE_LIB", str(source))
            selection = ["--ace-files", *names]
    for name in names:
        (source / name).touch()
    monkeypatch.setenv("MCDC_LIB", str(output))
    arguments = ["generate.py", "--verbose", *selection]
    return SimpleNamespace(
        generate=generate,
        output=output,
        names=names,
        arguments=arguments,
        converter=converter,
    )


@pytest.mark.parametrize("keep_going", [False, True])
def test_failed_inputs_stop_or_continue_and_report_errors(
    generator_case, monkeypatch, capsys, keep_going
):
    case = generator_case
    arguments = case.arguments.copy()
    if keep_going:
        arguments.append("--keep-going")
    monkeypatch.setattr(sys, "argv", arguments)

    def convert(path, directory, *args, **kwargs):
        if path.name == case.names[1]:
            raise ValueError("MT 22: invalid outgoing-energy table")
        if path.name == case.names[2]:
            raise OSError("cannot read source")
        (directory / path.name).write_text("converted")

    monkeypatch.setattr(case.generate, case.converter, convert)
    with pytest.raises(SystemExit) as error:
        case.generate.main()
    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert f"{case.names[1]}: MT 22: invalid outgoing-energy table" in stderr
    assert (case.output / case.names[0]).read_text() == "converted"
    assert (case.output / case.names[-1]).exists() == keep_going
    assert not (case.output / case.names[1]).exists()
    if keep_going:
        assert "2 file(s)" in stderr
        assert f"{case.names[2]}: cannot read source" in stderr
    else:
        assert case.names[2] not in stderr


def test_keep_going_success_has_zero_exit_status(generator_case, monkeypatch, capsys):
    case = generator_case
    monkeypatch.setattr(sys, "argv", [*case.arguments, "--keep-going"])
    monkeypatch.setattr(case.generate, case.converter, lambda *args, **kwargs: None)
    assert case.generate.main() is None
    assert not capsys.readouterr().err


@pytest.mark.parametrize("generator_case", ["proton"], indirect=True)
@pytest.mark.parametrize("keep_going", [False, True])
def test_proton_pstar_failures_use_the_same_batch_policy(
    generator_case, monkeypatch, capsys, keep_going
):
    case = generator_case
    arguments = case.arguments.copy()
    if keep_going:
        arguments.append("--keep-going")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(case.generate, case.converter, lambda *args, **kwargs: None)

    def convert_pstar(symbol, mass, ratio, directory, *args):
        if (symbol, mass) == ("H", 1):
            raise ValueError("invalid stopping-power table")
        if (symbol, mass) == ("H", 2):
            (directory / "proton-H2.h5").touch()

    monkeypatch.setattr(case.generate, "process_pstar_only_file", convert_pstar)
    with pytest.raises(SystemExit) as error:
        case.generate.main()
    assert error.value.code == 1
    assert "proton-H1.h5: invalid stopping-power table" in capsys.readouterr().err
    assert (case.output / "proton-H2.h5").exists() == keep_going


@pytest.mark.parametrize("generator_case", ["neutron", "proton"], indirect=True)
def test_ace_generators_forward_source_library_override(generator_case, monkeypatch):
    case = generator_case
    monkeypatch.setattr(
        sys, "argv", [*case.arguments, "--source-library", "CustomLibrary"]
    )
    labels = []

    def convert(*args, source_library):
        labels.append(source_library)

    monkeypatch.setattr(case.generate, case.converter, convert)
    case.generate.main()
    assert labels == ["CustomLibrary"] * len(case.names)
