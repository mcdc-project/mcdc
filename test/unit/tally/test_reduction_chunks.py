"""Check chunk boundaries, score normalization and receive-buffer ownership."""

import numpy as np
import pytest
from mpi4py import MPI

import mcdc
import mcdc.config as config
from mcdc.transport.tally.closeout import _reduce


@pytest.mark.parametrize(
    "chunks,expected_lengths", [(1, [10]), (3, [4, 3, 3]), (20, [1] * 10)]
)
@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("gpu_backed", [True, False])
def test_reduce_chunk_views(
    prepare_simulation, monkeypatch, chunks, expected_lengths, master, gpu_backed
):
    mesh = mcdc.MeshStructured(z=np.linspace(0, 1, 11))
    tally = mcdc.Tally(mesh=mesh)

    def configure(simulation):
        simulation.settings.N_particle = 2
        simulation.settings.N_batch = 2
        simulation.settings.tally_reduction_chunks = chunks

    container, data = prepare_simulation(tallies=[tally], configure=configure)
    simulation = container[0]
    simulation["mpi_master"] = master
    record = simulation["tallies"][0]
    start = record["bin_offset"]
    scores = data[start : start + record["bin_length"]]
    values = np.arange(1, 11, dtype=float)
    scores[:] = values
    calls = []
    buffers = []

    class Communicator:
        def Reduce(self, send, receive, op, root):
            assert op == MPI.SUM and root == 0
            array = receive if send is MPI.IN_PLACE else send
            assert np.shares_memory(array, data)
            calls.append(array.copy())
            if gpu_backed:
                assert not np.shares_memory(receive, data)
                buffers.append(receive)
                receive[:] = array + 1
            elif master:
                assert send is MPI.IN_PLACE
                receive[:] += 1
            else:
                assert receive is None

    monkeypatch.setattr(MPI, "COMM_WORLD", Communicator())
    monkeypatch.setattr(config, "target", "gpu" if gpu_backed else "cpu")
    monkeypatch.setattr(config, "gpu_state_storage", "managed")
    _reduce(record, simulation, data)

    assert [len(array) for array in calls] == expected_lengths
    np.testing.assert_array_equal(np.concatenate(calls), values / 2)
    np.testing.assert_array_equal(scores, values / 2 + (1 if master else 0))
    if gpu_backed:
        assert all(np.shares_memory(buffer, buffers[0]) for buffer in buffers)
