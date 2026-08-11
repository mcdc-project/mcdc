import pytest
import mcdc


def test_edep_cannot_be_grouped_with_other_scores(capsys):
    mesh = mcdc.MeshUniform(
        x=(0.0, 1.0, 1),
        y=(0.0, 1.0, 1),
        z=(0.0, 1.0, 1),
    )

    with pytest.raises(SystemExit):
        mcdc.Tally(mesh=mesh, scores=["flux", "energy_deposition"])

    captured = capsys.readouterr()
    assert "cannot be grouped with other scores yet" in captured.out
