import h5py, math, numba
from numba.cuda import atomic
from numpy import euler_gamma

from mpi4py import MPI
from numba import (
    int64,
    literal_unroll,
    njit,
    objmode,
    uint64,
)

import mcdc.objects as objects

import mcdc.adapt as adapt
import mcdc.src.geometry as geometry
import mcdc.src.mesh as mesh_
import mcdc.src.physics as physics
import mcdc.src.surface as surface_
import mcdc.type_ as type_

from mcdc.adapt import toggle, for_cpu, for_gpu
from mcdc.constant import *
from mcdc.prints import print_error
from mcdc.src.algorithm import binary_search, binary_search_with_length, evaluate_from_table


@njit
def round(float_val):
    return float_val
    # int_val = np.float64(float_val).view(np.uint64)
    # if (int_val & 0x10) != 0:
    #    int_val += 0x10
    # int_val = int_val & ~0x0F
    # return np.uint64(int_val).view(np.float64)


# =============================================================================
# Domain Decomposition
# =============================================================================

# =============================================================================
# Domain crossing event
# =============================================================================


@toggle("domain_decomp")
def domain_crossing(P_arr, prog):
    mcdc = adapt.mcdc_global(prog)
    P = P_arr[0]
    # Domain mesh crossing
    seed = P["rng_seed"]
    max_size = mcdc["technique"]["dd_exchange_rate"]
    if mcdc["technique"]["domain_decomposition"]:
        mesh = mcdc["technique"]["dd_mesh"]
        # Determine which dimension is crossed
        ix, iy, iz, it, outside = mesh_.structured.get_indices(P_arr, mesh)

        d_idx = mcdc["dd_idx"]
        d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
        d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
        d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

        d_iz = int(d_idx / (d_Nx * d_Ny))
        d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
        d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

        flag = MESH_NONE
        if d_ix != ix:
            flag = MESH_X
        elif d_iy != iy:
            flag = MESH_Y
        elif d_iz != iz:
            flag = MESH_Z

        # Score on tally
        if flag == MESH_X and P["ux"] > 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_xp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_xp"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_X and P["ux"] < 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_xn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_xn"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Y and P["uy"] > 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_yp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_yp"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Y and P["uy"] < 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_yn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_yn"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Z and P["uz"] > 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_zp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_zp"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Z and P["uz"] < 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_zn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_zn"]) == max_size:
                dd_initiate_particle_send(prog)
        P["alive"] = False


# =============================================================================
# Send full domain bank
# =============================================================================


requests = []


def save_request(req_pair):
    global requests

    updated_requests = []

    status = MPI.Status()
    for req, buf in requests:
        if not req.Test(status):
            updated_requests.append((req, buf))

    updated_requests.append(req_pair)
    requests = updated_requests


def clear_requests():
    global requests
    for req, buf in requests:
        req.Free()

    requests = []


@njit
def dd_check_halt(mcdc):
    return mcdc["domain_decomp"]["work_done"]


@njit
def dd_check_in(mcdc):
    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0
    mcdc["domain_decomp"]["send_total"] = 0
    mcdc["domain_decomp"]["rank_busy"] = True

    with objmode(rank="int64", total="int64"):
        rank = MPI.COMM_WORLD.Get_rank()
        total = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        mcdc["domain_decomp"]["busy_total"] = total
    else:
        mcdc["domain_decomp"]["busy_total"] = 0


@njit
def dd_check_out(mcdc):
    with objmode():
        rank = MPI.COMM_WORLD.Get_rank()
        send_count = mcdc["domain_decomp"]["send_count"]
        recv_count = mcdc["domain_decomp"]["recv_count"]
        send_total = mcdc["domain_decomp"]["send_total"]
        busy_total = mcdc["domain_decomp"]["busy_total"]
        rank_busy = mcdc["domain_decomp"]["rank_busy"]

        if send_count != 0:
            print(
                f"Domain decomposed loop closed out with non-zero send count {send_count} in rank {rank}"
            )
            mcdc["domain_decomp"]["send_count"] = 0

        if recv_count != 0:
            print(
                f"Domain decomposed loop closed out with non-zero recv count {recv_count} in rank {rank}"
            )
            mcdc["domain_decomp"]["recv_count"] = 0

        if send_total != 0:
            print(
                f"Domain decomposed loop closed out with non-zero send total {send_total} in rank {rank}"
            )
            mcdc["domain_decomp"]["send_total"] = 0

        if busy_total != 0:
            print(
                f"Domain decomposed loop closed out with non-zero busy total {busy_total} in rank {rank}"
            )
            mcdc["domain_decomp"]["busy_total"] = 0

        if rank_busy:
            print(
                f"Domain decomposed loop closed out with rank {rank} still marked as busy"
            )
            mcdc["domain_decomp"]["rank_busy"] = 0

        clear_requests()


@njit
def dd_signal_halt(mcdc):

    with objmode():
        for rank in range(1, MPI.COMM_WORLD.Get_size()):
            dummy_buff = np.zeros((1,), dtype=np.int32)
            MPI.COMM_WORLD.Send(dummy_buff, dest=rank, tag=3)

    mcdc["domain_decomp"]["work_done"] = True


@njit
def dd_signal_block(mcdc):

    with objmode(rank="int64"):
        rank = MPI.COMM_WORLD.Get_rank()

    send_delta = (
        mcdc["domain_decomp"]["send_count"] - mcdc["domain_decomp"]["recv_count"]
    )
    if rank == 0:
        mcdc["domain_decomp"]["send_total"] += send_delta
        mcdc["domain_decomp"]["busy_total"] -= 1
    else:
        with objmode():
            buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
            buff[0]["busy_delta"] = -1
            buff[0]["send_delta"] = send_delta
            req = MPI.COMM_WORLD.Isend(
                [buff, type_.dd_turnstile_event_mpi], dest=0, tag=2
            )
            save_request((req, buff))

    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0

    if (
        (rank == 0)
        and (mcdc["domain_decomp"]["busy_total"] == 0)
        and (mcdc["domain_decomp"]["send_total"] == 0)
    ):
        dd_signal_halt(mcdc)


@njit
def dd_signal_unblock(mcdc):

    with objmode(rank="int64"):
        rank = MPI.COMM_WORLD.Get_rank()

    send_delta = (
        mcdc["domain_decomp"]["send_count"] - mcdc["domain_decomp"]["recv_count"]
    )

    if rank == 0:
        mcdc["domain_decomp"]["send_total"] += send_delta
        mcdc["domain_decomp"]["busy_total"] += 1
        if (mcdc["domain_decomp"]["busy_total"] == 0) and (
            mcdc["domain_decomp"]["send_total"] == 0
        ):
            dd_signal_halt(mcdc)
    else:
        with objmode():
            buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
            buff[0]["busy_delta"] = 1
            buff[0]["send_delta"] = send_delta
            req = MPI.COMM_WORLD.Isend(
                [buff, type_.dd_turnstile_event_mpi], dest=0, tag=2
            )
            save_request((req, buff))
    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0


@njit
def dd_distribute_bank(mcdc, bank, dest_list):

    with objmode(send_delta="int64"):
        dest_count = len(dest_list)
        send_delta = 0

        for i, dest in enumerate(dest_list):
            size = get_bank_size(bank)
            ratio = int(size / dest_count)
            start = ratio * i
            end = start + ratio
            if i == dest_count - 1:
                end = size
            sub_bank = np.array(bank["particles"][start:end])
            if sub_bank.shape[0] > 0:
                req = MPI.COMM_WORLD.Isend(
                    [sub_bank, type_.particle_record_mpi], dest=dest, tag=1
                )
                save_request((req, sub_bank))
                send_delta += end - start

    mcdc["domain_decomp"]["send_count"] += send_delta
    set_bank_size(bank, 0)


@for_gpu()
def dd_initiate_particle_send(prog):
    adapt.halt_early(prog)


@for_cpu()
def dd_initiate_particle_send(prog):
    dd_particle_send(prog)


@njit
def dd_particle_send(prog):
    mcdc = adapt.mcdc_global(prog)
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_xp"], mcdc["technique"]["dd_xp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_xn"], mcdc["technique"]["dd_xn_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_yp"], mcdc["technique"]["dd_yp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_yn"], mcdc["technique"]["dd_yn_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_zp"], mcdc["technique"]["dd_zp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_zn"], mcdc["technique"]["dd_zn_neigh"]
    )


# =============================================================================
# Receive particles and clear banks
# =============================================================================


@njit
def dd_get_recv_tag():

    with objmode(tag="int64"):
        status = MPI.Status()
        MPI.COMM_WORLD.Probe(status=status)
        tag = status.Get_tag()

    return tag


@njit
def dd_recv_particles(mcdc):

    buff = np.zeros(
        mcdc["domain_decomp"]["bank_zp"]["particles"].shape[0],
        dtype=type_.particle_record,
    )

    with objmode(size="int64"):
        status = MPI.Status()
        MPI.COMM_WORLD.Recv([buff, type_.particle_record_mpi], status=status)
        size = status.Get_count(type_.particle_record_mpi)
        rank = MPI.COMM_WORLD.Get_rank()

    mcdc["domain_decomp"]["recv_count"] += size

    # Set source bank from buffer
    for i in range(size):
        add_particle(buff[i : i + 1], mcdc["bank_active"])

    if (
        mcdc["domain_decomp"]["recv_count"] > 0
        and not mcdc["domain_decomp"]["rank_busy"]
    ):
        dd_signal_unblock(mcdc)
        mcdc["domain_decomp"]["rank_busy"] = True


@njit
def dd_recv_turnstile(mcdc):

    with objmode(busy_delta="int64", send_delta="int64"):
        event_buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
        MPI.COMM_WORLD.Recv([event_buff, type_.dd_turnstile_event_mpi])
        busy_delta = event_buff[0]["busy_delta"]
        send_delta = event_buff[0]["send_delta"]
        rank = MPI.COMM_WORLD.Get_rank()
        busy_total = mcdc["domain_decomp"]["busy_total"]
        send_total = mcdc["domain_decomp"]["send_total"]

    mcdc["domain_decomp"]["busy_total"] += busy_delta
    mcdc["domain_decomp"]["send_total"] += send_delta

    if (mcdc["domain_decomp"]["busy_total"] == 0) and (
        mcdc["domain_decomp"]["send_total"] == 0
    ):
        dd_signal_halt(mcdc)


@njit
def dd_recv_halt(mcdc):

    with objmode():
        dummy_buff = np.zeros((1,), dtype=np.int32)
        MPI.COMM_WORLD.Recv(dummy_buff)
        work_done = 1
        rank = MPI.COMM_WORLD.Get_rank()

    mcdc["domain_decomp"]["work_done"] = True


@njit
def dd_recv(mcdc):

    if mcdc["domain_decomp"]["rank_busy"]:
        dd_signal_block(mcdc)
        mcdc["domain_decomp"]["rank_busy"] = False

    if not mcdc["domain_decomp"]["work_done"]:
        tag = dd_get_recv_tag()

        if tag == 1:
            dd_recv_particles(mcdc)
        elif tag == 2:
            dd_recv_turnstile(mcdc)
        elif tag == 3:
            dd_recv_halt(mcdc)


# =============================================================================
# Particle in domain
# =============================================================================


# Check if particle is in domain
@njit
def particle_in_domain(P_arr, mcdc):
    P = P_arr[0]
    d_idx = mcdc["dd_idx"]
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

    d_iz = int(d_idx / (d_Nx * d_Ny))
    d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    mesh = mcdc["technique"]["dd_mesh"]
    x_cell, y_cell, z_cell, t_cell, outside = mesh_.structured.get_indices(P_arr, mesh)

    if d_ix == x_cell:
        if d_iy == y_cell:
            if d_iz == z_cell:
                return True
    return False


# =============================================================================
# Source in domain
# =============================================================================


# Check for source in domain
@njit
def source_in_domain(source, domain_mesh, d_idx):
    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1

    d_iz = int(d_idx / (d_Nx * d_Ny))
    d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]

    if (
        d_x[0] <= source["box_x"][0] <= d_x[1]
        or d_x[0] <= source["box_x"][1] <= d_x[1]
        or (source["box_x"][0] < d_x[0] and source["box_x"][1] > d_x[1])
    ):
        if (
            d_y[0] <= source["box_y"][0] <= d_y[1]
            or d_y[0] <= source["box_y"][1] <= d_y[1]
            or (source["box_y"][0] < d_y[0] and source["box_y"][1] > d_y[1])
        ):
            if (
                d_z[0] <= source["box_z"][0] <= d_z[1]
                or d_z[0] <= source["box_z"][1] <= d_z[1]
                or (source["box_z"][0] < d_z[0] and source["box_z"][1] > d_z[1])
            ):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


# =============================================================================
# Compute domain load
# =============================================================================


@njit
def domain_work(mcdc, domain, N):
    domain_mesh = mcdc["technique"]["dd_mesh"]

    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1
    work_start = 0
    for d_idx in range(domain):
        d_iz = int(d_idx / (d_Nx * d_Ny))
        d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
        d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

        d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
        d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
        d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]
        # Compute volumes of sources and numbers of particles

        Psum = 0

        Nm = 0
        num_source = 0
        for source in mcdc["sources"]:
            Psum += source["prob"]
            num_source += 1
        Vi = np.zeros(num_source)
        Vim = np.zeros(num_source)
        Ni = np.zeros(num_source)
        i = 0
        for source in mcdc["sources"]:
            Ni[i] = N * source["prob"] / Psum
            Vi[i] = 1
            Vim[i] = 1
            if source["box"] == True:
                xV = source["box_x"][1] - source["box_x"][0]
                if xV != 0:
                    Vi[i] *= xV
                    Vim[i] *= min(source["box_x"][1], d_x[1]) - max(
                        source["box_x"][0], d_x[0]
                    )
                yV = source["box_y"][1] - source["box_y"][0]
                if yV != 0:
                    Vi[i] *= yV
                    Vim[i] *= min(source["box_y"][1], d_y[1]) - max(
                        source["box_y"][0], d_y[0]
                    )
                zV = source["box_z"][1] - source["box_z"][0]
                if zV != 0:
                    Vi[i] *= zV
                    Vim[i] *= min(source["box_z"][1], d_z[1]) - max(
                        source["box_z"][0], d_z[0]
                    )
            if not source_in_domain(source, domain_mesh, d_idx):
                Vim[i] = 0
            i += 1
        for source in range(num_source):
            Nm += Ni[source] * Vim[source] / Vi[source]
        work_start += Nm
    d_idx = domain
    d_iz = int(mcdc["dd_idx"] / (d_Nx * d_Ny))
    d_iy = int((mcdc["dd_idx"] - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(mcdc["dd_idx"] - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]
    # Compute volumes of sources and numbers of particles
    num_source = len(mcdc["sources"])
    Vi = np.zeros(num_source)
    Vim = np.zeros(num_source)
    Ni = np.zeros(num_source)
    Psum = 0

    Nm = 0
    for source in mcdc["sources"]:
        Psum += source["prob"]
    i = 0
    for source in mcdc["sources"]:
        Ni[i] = N * source["prob"] / Psum
        Vi[i] = 1
        Vim[i] = 1

        if source["box"] == True:
            xV = source["box_x"][1] - source["box_x"][0]
            if xV != 0:
                Vi[i] *= xV
                Vim[i] *= min(source["box_x"][1], d_x[1]) - max(
                    source["box_x"][0], d_x[0]
                )
            yV = source["box_y"][1] - source["box_y"][0]
            if yV != 0:
                Vi[i] *= yV
                Vim[i] *= min(source["box_y"][1], d_y[1]) - max(
                    source["box_y"][0], d_y[0]
                )
            zV = source["box_z"][1] - source["box_z"][0]
            if zV != 0:
                Vi[i] *= zV
                Vim[i] *= min(source["box_z"][1], d_z[1]) - max(
                    source["box_z"][0], d_z[0]
                )
        i += 1
    for source in range(num_source):
        Nm += Ni[source] * Vim[source] / Vi[source]
    Nm /= mcdc["technique"]["dd_work_ratio"][domain]
    rank = mcdc["mpi_rank"]
    if mcdc["technique"]["dd_work_ratio"][domain] > 1:
        work_start += Nm * (rank - np.sum(mcdc["technique"]["dd_work_ratio"][0:d_idx]))
    total_v = 0
    for source in range(len(mcdc["sources"])):
        total_v += Vim[source]
    i = 0
    for source in mcdc["sources"]:
        if total_v != 0:
            source["prob"] *= 2 * Vim[i] / total_v
        i += 1
    return (int(Nm), int(work_start))


# =============================================================================
# Source particle in domain only
# =============================================================================


@njit()
def source_particle_dd(seed, mcdc):
    domain_mesh = mcdc["technique"]["dd_mesh"]
    d_idx = mcdc["dd_idx"]

    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1

    d_iz = int(mcdc["dd_idx"] / (d_Nx * d_Ny))
    d_iy = int((mcdc["dd_idx"] - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(mcdc["dd_idx"] - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]

    P_arr = np.zeros(1, dtype=type_.particle_record)
    P = P_arr[0]

    P["rng_seed"] = seed
    # Sample source
    xi = rng(P_arr)
    tot = 0.0
    for source in mcdc["sources"]:
        if source_in_domain(source, domain_mesh, d_idx):
            tot += source["prob"]
            if tot >= xi:
                break

    # Position
    if source["box"]:
        x = sample_uniform(
            max(source["box_x"][0], d_x[0]), min(source["box_x"][1], d_x[1]), P
        )
        y = sample_uniform(
            max(source["box_y"][0], d_y[0]), min(source["box_y"][1], d_y[1]), P
        )
        z = sample_uniform(
            max(source["box_z"][0], d_z[0]), min(source["box_z"][1], d_z[1]), P
        )

    else:
        x = source["x"]
        y = source["y"]
        z = source["z"]

    # Direction
    if source["isotropic"]:
        ux, uy, uz = sample_isotropic_direction(P_arr)
    elif source["white"]:
        ux, uy, uz = sample_white_direction(
            source["white_x"], source["white_y"], source["white_z"], P
        )
    else:
        ux = source["ux"]
        uy = source["uy"]
        uz = source["uz"]

    # Energy and time
    g = sample_discrete(source["group"], P_arr)
    t = sample_uniform(source["time"][0], source["time"][1], P_arr)

    # Make and return particle
    P["x"] = x
    P["y"] = y
    P["z"] = z
    P["t"] = t
    P["ux"] = ux
    P["uy"] = uy
    P["uz"] = uz
    P["g"] = g
    P["w"] = 1
    return P


@njit
def distribute_work_dd(N, mcdc, precursor=False):
    work_start = 0
    work_size = N
    work_size_total = N

    if not precursor:
        mcdc["mpi_work_start"] = work_start
        mcdc["mpi_work_size"] = work_size
        mcdc["mpi_work_size_total"] = work_size_total
    else:
        mcdc["mpi_work_start_precursor"] = work_start
        mcdc["mpi_work_size_precursor"] = work_size
        mcdc["mpi_work_size_total_precursor"] = work_size_total


# =============================================================================
# Random sampling
# =============================================================================


@njit
def sample_isotropic_direction(P_arr):
    P = P_arr[0]
    # Sample polar cosine and azimuthal angle uniformly
    mu = 2.0 * rng(P_arr) - 1.0
    azi = 2.0 * PI * rng(P_arr)

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2) ** 0.5
    y = math.cos(azi) * c
    z = math.sin(azi) * c
    x = mu
    return x, y, z


@njit
def sample_white_direction(nx, ny, nz, P_arr):
    P = P_arr[0]
    # Sample polar cosine
    mu = math.sqrt(rng(P_arr))

    # Sample azimuthal direction
    azi = 2.0 * PI * rng(P_arr)
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu**2) ** 0.5

    if nz != 1.0:
        B = (1.0 - nz**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * nz * cos_azi - ny * sin_azi) * C
        y = ny * mu + (ny * nz * cos_azi + nx * sin_azi) * C
        z = nz * mu - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the formula
    else:
        B = (1.0 - ny**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * ny * cos_azi - nz * sin_azi) * C
        z = nz * mu + (nz * ny * cos_azi + nx * sin_azi) * C
        y = ny * mu - cos_azi * Ac * B
    return x, y, z


@njit
def sample_uniform(a, b, P_arr):
    P = P_arr[0]
    return a + rng(P_arr) * (b - a)


# TODO: use cummulative density function and binary search
@njit
def sample_discrete(group, P_arr):
    P = P_arr[0]
    tot = 0.0
    xi = rng(P_arr)
    for i in range(group.shape[0]):
        tot += group[i]
        if tot > xi:
            return i


@njit
def sample_piecewise_linear(cdf, P_arr):
    P = P_arr[0]
    xi = rng(P_arr)

    # Get bin
    idx = binary_search(xi, cdf[1])

    # Linear interpolation
    x1 = cdf[1, idx]
    x2 = cdf[1, idx + 1]
    y1 = cdf[0, idx]
    y2 = cdf[0, idx + 1]
    return y1 + (xi - x1) * (y2 - y1) / (x2 - x1)


# =============================================================================
# Random number generator
#   LCG with hash seed-split
# =============================================================================


@njit
def wrapping_add(a, b):
    return a + b


def wrapping_mul_python(a, b):
    a = uint64(a)
    b = uint64(b)
    with np.errstate(all="ignore"):
        return a * b


def wrapping_add_python(a, b):
    a = uint64(a)
    b = uint64(b)
    with np.errstate(all="ignore"):
        return a + b


def adapt_rng(object_mode=False):
    global wrapping_add, wrapping_mul
    if object_mode:
        wrapping_add = wrapping_add_python
        wrapping_mul = wrapping_mul_python


@njit
def rng_(seed):
    seed = uint64(seed)
    return wrapping_add(wrapping_mul(RNG_G, seed), RNG_C) & RNG_MOD_MASK


@njit
def rng(state_arr):
    state = state_arr[0]
    state["rng_seed"] = rng_(state["rng_seed"])
    return state["rng_seed"] / RNG_MOD


@njit
def rng_from_seed(seed):
    return rng_(seed) / RNG_MOD


@njit
def rng_array(seed, shape, size):
    xi = np.zeros(size)
    for i in range(size):
        xi_seed = split_seed(i, seed)
        xi[i] = rng_from_seed(xi_seed)
    xi = xi.reshape(shape)
    return xi


# =============================================================================
# Particle source operations
# =============================================================================


@njit
def source_particle(P_rec_arr, seed, mcdc):
    P_rec = P_rec_arr[0]
    P_rec["rng_seed"] = seed

    # Sample source
    xi = rng(P_rec_arr)
    tot = 0.0
    for source in mcdc["sources"]:
        tot += source["prob"]
        if tot >= xi:
            break

    # Position
    if source["box"]:
        x = sample_uniform(source["box_x"][0], source["box_x"][1], P_rec_arr)
        y = sample_uniform(source["box_y"][0], source["box_y"][1], P_rec_arr)
        z = sample_uniform(source["box_z"][0], source["box_z"][1], P_rec_arr)
    else:
        x = source["x"]
        y = source["y"]
        z = source["z"]

    # Direction
    if source["isotropic"]:
        ux, uy, uz = sample_isotropic_direction(P_rec_arr)
    elif source["white"]:
        ux, uy, uz = sample_white_direction(
            source["white_x"], source["white_y"], source["white_z"], P_rec_arr
        )
    else:
        ux = source["ux"]
        uy = source["uy"]
        uz = source["uz"]

    # Energy and time
    if mcdc["setting"]["mode_MG"]:
        g = sample_discrete(source["group"], P_rec_arr)
        E = 0.0
    else:
        g = 0
        E = sample_piecewise_linear(source["energy"], P_rec_arr)

    # Time
    t = sample_uniform(source["time"][0], source["time"][1], P_rec_arr)

    # Make and return particle
    P_rec["x"] = x
    P_rec["y"] = y
    P_rec["z"] = z
    P_rec["t"] = t
    P_rec["ux"] = ux
    P_rec["uy"] = uy
    P_rec["uz"] = uz
    P_rec["g"] = g
    P_rec["E"] = E
    P_rec["w"] = 1.0


# =============================================================================
# Particle bank operations
# =============================================================================


@njit
def get_bank_size(bank):
    return bank["size"][0]


@njit
def set_bank_size(bank, value):
    bank["size"][0] = value


@njit
def add_bank_size(bank, value):
    return adapt.global_add(bank["size"], 0, value)


@for_cpu()
def full_bank_print(bank):
    with objmode():
        print_error("Particle %s bank is full." % bank["tag"])


@for_gpu()
def full_bank_print(bank):
    pass


@njit
def add_particle(P_arr, bank):
    P = P_arr[0]

    idx = add_bank_size(bank, 1)

    # Check if bank is full
    if idx >= bank["particles"].shape[0]:
        full_bank_print(bank)

    # Set particle
    copy_recordlike(bank["particles"][idx : idx + 1], P_arr)


@njit
def get_particle(P_arr, bank, mcdc):
    P = P_arr[0]

    idx = add_bank_size(bank, -1) - 1

    # Check if bank is empty
    if idx < 0:
        return False
        # with objmode():
        #    print_error("Particle %s bank is empty." % bank["tag"])

    # Set attribute
    P_rec = bank["particles"][idx]
    P["x"] = P_rec["x"]
    P["y"] = P_rec["y"]
    P["z"] = P_rec["z"]
    P["t"] = P_rec["t"]
    P["ux"] = P_rec["ux"]
    P["uy"] = P_rec["uy"]
    P["uz"] = P_rec["uz"]
    P["g"] = P_rec["g"]
    P["E"] = P_rec["E"]
    P["w"] = P_rec["w"]
    P["rng_seed"] = P_rec["rng_seed"]

    if mcdc["technique"]["iQMC"]:
        P["iqmc"]["w"] = P_rec["iqmc"]["w"]

    P["alive"] = True

    # Set default IDs and event
    P["material_ID"] = -1
    P["cell_ID"] = -1
    P["surface_ID"] = -1
    P["event"] = -1
    return True


@njit
def check_future_bank(mcdc):
    # Get the data needed
    bank_future = mcdc["bank_future"]
    bank_census = mcdc["bank_census"]
    next_census_time = mcdc["setting"]["census_time"][mcdc["idx_census"] + 1]

    # Particle container
    P_arr = adapt.local_array(1, type_.particle_record)
    P = P_arr[0]

    # Loop over all particles in future bank
    N = get_bank_size(bank_future)
    for i in range(N):
        # Get the next future particle index
        idx = i - get_bank_size(bank_census)
        copy_recordlike(P_arr, bank_future["particles"][idx : idx + 1])

        # Promote the future particle to census bank
        if P["t"] < next_census_time:
            adapt.add_census(P_arr, mcdc)
            add_bank_size(bank_future, -1)

            # Consolidate the emptied space in the future bank
            j = get_bank_size(bank_future)
            copy_recordlike(
                bank_future["particles"][idx : idx + 1],
                bank_future["particles"][j : j + 1],
            )


@njit
def manage_particle_banks(seed, mcdc):
    # Record time
    if mcdc["mpi_master"]:
        with objmode(time_start="float64"):
            time_start = MPI.Wtime()

    # Reset source bank
    set_bank_size(mcdc["bank_source"], 0)

    # Normalize weight
    if mcdc["setting"]["mode_eigenvalue"]:
        normalize_weight(mcdc["bank_census"], mcdc["setting"]["N_particle"])

    # Population control
    if mcdc["technique"]["population_control"]:
        population_control(seed, mcdc)
    else:
        # Swap census and source bank
        size = get_bank_size(mcdc["bank_census"])
        set_bank_size(mcdc["bank_source"], size)
        mcdc["bank_source"]["particles"][:size] = mcdc["bank_census"]["particles"][
            :size
        ]
    # TODO: Population control future bank?

    # MPI rebalance
    if not mcdc["technique"]["domain_decomposition"]:
        bank_rebalance(mcdc)

    # Zero out census bank
    set_bank_size(mcdc["bank_census"], 0)

    # Manage IC bank
    if mcdc["technique"]["IC_generator"] and mcdc["cycle_active"]:
        manage_IC_bank(mcdc)

    # Accumulate time
    if mcdc["mpi_master"]:
        with objmode(time_end="float64"):
            time_end = MPI.Wtime()
        mcdc["runtime_bank_management"] += time_end - time_start


@njit
def manage_IC_bank(mcdc):
    # Buffer bank
    buff_n = np.zeros(
        mcdc["technique"]["IC_bank_neutron_local"]["particles"].shape[0],
        dtype=type_.particle_record,
    )
    buff_p = np.zeros(
        mcdc["technique"]["IC_bank_precursor_local"]["precursors"].shape[0],
        dtype=type_.precursor,
    )

    with objmode(Nn="int64", Np="int64"):
        # Create MPI-supported numpy object
        Nn = get_bank_size(mcdc["technique"]["IC_bank_neutron_local"])
        Np = get_bank_size(mcdc["technique"]["IC_bank_precursor_local"])

        neutrons = MPI.COMM_WORLD.gather(
            mcdc["technique"]["IC_bank_neutron_local"]["particles"][:Nn]
        )
        precursors = MPI.COMM_WORLD.gather(
            mcdc["technique"]["IC_bank_precursor_local"]["precursors"][:Np]
        )

        if mcdc["mpi_master"]:
            neutrons = np.concatenate(neutrons[:])
            precursors = np.concatenate(precursors[:])

            # Set output buffer
            Nn = neutrons.shape[0]
            Np = precursors.shape[0]
            for i in range(Nn):
                buff_n[i] = neutrons[i]
            for i in range(Np):
                buff_p[i] = precursors[i]

    # Set global bank from buffer
    if mcdc["mpi_master"]:
        start_n = add_bank_size(mcdc["technique"]["IC_bank_neutron"], Nn)
        start_p = add_bank_size(mcdc["technique"]["IC_bank_precursor"], Np)
        for i in range(Nn):
            mcdc["technique"]["IC_bank_neutron"]["particles"][start_n + i] = buff_n[i]
        for i in range(Np):
            mcdc["technique"]["IC_bank_precursor"]["precursors"][start_p + i] = buff_p[
                i
            ]

    # Reset local banks
    set_bank_size(mcdc["technique"]["IC_bank_neutron_local"], 0)
    set_bank_size(mcdc["technique"]["IC_bank_precursor_local"], 0)


@njit
def bank_scanning(bank, mcdc):
    N_local = get_bank_size(bank)

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    N_global = buff[0]

    return idx_start, N_local, N_global


@njit
def bank_scanning_weight(bank, mcdc):
    # Local weight CDF
    N_local = get_bank_size(bank)
    w_cdf = np.zeros(N_local + 1)
    for i in range(N_local):
        w_cdf[i + 1] = w_cdf[i] + bank["particles"][i]["w"]
    W_local = w_cdf[-1]

    # Starting weight
    buff = np.zeros(1, dtype=np.float64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([W_local]), buff, MPI.SUM)
    w_start = buff[0]
    w_cdf += w_start

    # Global weight
    buff[0] = w_cdf[-1]
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    W_global = buff[0]

    return w_start, w_cdf, W_global


@njit
def bank_scanning_DNP(bank, mcdc):
    N_DNP_local = get_bank_size(bank)

    # Get sum of ceil-ed local DNP weights
    N_local = 0
    for i in range(N_DNP_local):
        DNP = bank["precursors"][i]
        N_local += math.ceil(DNP["w"])

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    N_global = buff[0]

    return idx_start, N_local, N_global


@njit
def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for P in bank["particles"]:
        P["w"] *= norm / W


@njit
def total_weight(bank):
    # Local total weight
    W_local = np.zeros(1)
    for i in range(get_bank_size(bank)):
        W_local[0] += bank["particles"][i]["w"]

    # MPI Allreduce
    buff = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(W_local, buff, MPI.SUM)
    return buff[0]


@njit
def allreduce(value):
    total = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array([value], np.float64), total, MPI.SUM)
    return total[0]


@njit
def allreduce_array(array):
    buff = np.zeros_like(array)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array(array), buff, op=MPI.SUM)
    array[:] = buff


@njit
def bank_rebalance(mcdc):
    # Scan the bank
    idx_start, N_local, N = bank_scanning(mcdc["bank_source"], mcdc)
    idx_end = idx_start + N_local

    # Abort if source bank is empty
    if N == 0:
        return

    distribute_work(N, mcdc)

    # Rebalance not needed if there is only one rank
    if mcdc["mpi_size"] <= 1:
        return

    # Some constants
    work_start = mcdc["mpi_work_start"]
    work_end = work_start + mcdc["mpi_work_size"]
    left = mcdc["mpi_rank"] - 1
    right = mcdc["mpi_rank"] + 1

    # Need more or less?
    more_left = idx_start < work_start
    less_left = idx_start > work_start
    more_right = idx_end > work_end
    less_right = idx_end < work_end

    # Offside?
    offside_left = idx_end <= work_start
    offside_right = idx_start >= work_end

    # MPI nearest-neighbor send/receive
    buff = np.zeros(
        mcdc["bank_source"]["particles"].shape[0], dtype=type_.particle_record
    )

    with objmode(size="int64"):
        # Create MPI-supported numpy object
        size = get_bank_size(mcdc["bank_source"])
        bank = np.array(mcdc["bank_source"]["particles"][:size])

        # If offside, need to receive first
        if offside_left:
            # Receive from right
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))
            less_right = False
        if offside_right:
            # Receive from left
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
            less_left = False

        # Send
        if more_left:
            n = work_start - idx_start
            request_left = MPI.COMM_WORLD.isend(bank[:n], dest=left)
            bank = bank[n:]
        if more_right:
            n = idx_end - work_end
            request_right = MPI.COMM_WORLD.isend(bank[-n:], dest=right)
            bank = bank[:-n]

        # Receive
        if less_left:
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
        if less_right:
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))

        # Wait until sent massage is received
        if more_left:
            request_left.Wait()
        if more_right:
            request_right.Wait()

        # Set output buffer
        size = bank.shape[0]
        for i in range(size):
            buff[i] = bank[i]

    # Set source bank from buffer
    set_bank_size(mcdc["bank_source"], size)
    for i in range(size):
        mcdc["bank_source"]["particles"][i] = buff[i]


@njit
def distribute_work(N, mcdc, precursor=False):
    size = mcdc["mpi_size"]
    rank = mcdc["mpi_rank"]

    # Total # of work
    work_size_total = N

    # Evenly distribute work
    work_size = math.floor(N / size)

    # Starting index (based on even distribution)
    work_start = work_size * rank

    # Count reminder
    rem = N % size

    # Assign reminder and update starting index
    if rank < rem:
        work_size += 1
        work_start += rank
    else:
        work_start += rem

    if not precursor:
        mcdc["mpi_work_start"] = work_start
        mcdc["mpi_work_size"] = work_size
        mcdc["mpi_work_size_total"] = work_size_total
    else:
        mcdc["mpi_work_start_precursor"] = work_start
        mcdc["mpi_work_size_precursor"] = work_size
        mcdc["mpi_work_size_total_precursor"] = work_size_total


# =============================================================================
# IC generator
# =============================================================================


@for_cpu()
def pn_over_one():
    with objmode():
        print_error("Pn > 1.0.")


@for_gpu()
def pn_over_one():
    pass


@for_cpu()
def pp_over_one():
    with objmode():
        print_error("Pp > 1.0.")


@for_gpu()
def pp_over_one():
    pass


@njit
def bank_IC(P_arr, prog):
    P = P_arr[0]

    mcdc = adapt.mcdc_global(prog)

    # TODO: Consider multi-nuclide material
    material = mcdc["nuclides"][P["material_ID"]]

    # =========================================================================
    # Neutron
    # =========================================================================

    # Neutron weight
    g = P["g"]
    SigmaT = material["total"][g]
    weight = P["w"]
    flux = weight / SigmaT
    v = material["speed"][g]
    wn = flux / v

    # Neutron target weight
    Nn = mcdc["technique"]["IC_N_neutron"]
    tally_n = mcdc["technique"]["IC_neutron_density"]
    N_cycle = mcdc["setting"]["N_active"]
    wn_prime = tally_n * N_cycle / Nn

    # Sampling probability
    Pn = wn / wn_prime

    # TODO: Splitting for Pn > 1.0
    if Pn > 1.0:
        pn_over_one()

    # Sample particle
    if rng(P_arr) < Pn:
        P_new_arr = adapt.local_array(1, type_.particle)
        P_new = P_new_arr[0]
        split_as_record(P_new_arr, P_arr)
        P_new["w"] = 1.0
        P_new["t"] = 0.0
        adapt.add_IC(P_new_arr, prog)

        # Accumulate fission
        SigmaF = material["fission"][g]
        # mcdc["technique"]["IC_fission_score"][0] += v * SigmaF
        adapt.global_add(mcdc["technique"]["IC_fission_score"], 0, round(v * SigmaF))

    # =========================================================================
    # Precursor
    # =========================================================================

    # Sample precursor?
    Np = mcdc["technique"]["IC_N_precursor"]
    if Np == 0:
        return

    # Precursor weight
    J = material["J"]
    nu_d = material["nu_d"][g]
    SigmaF = material["fission"][g]
    decay = material["decay"]
    total = 0.0
    for j in range(J):
        total += nu_d[j] / decay[j]
    wp = flux * total * SigmaF / mcdc["k_eff"]

    # Material has no precursor
    if total == 0.0:
        return

    # Precursor target weight
    tally_C = mcdc["technique"]["IC_precursor_density"]
    wp_prime = tally_C * N_cycle / Np

    # Sampling probability
    Pp = wp / wp_prime

    # TODO: Splitting for Pp > 1.0
    if Pp > 1.0:
        pp_over_one()

    # Sample precursor
    if rng(P_arr) < Pp:
        idx = add_bank_size(mcdc["technique"]["IC_bank_precursor_local"], 1)
        precursor = mcdc["technique"]["IC_bank_precursor_local"]["precursors"][idx]
        precursor["x"] = P["x"]
        precursor["y"] = P["y"]
        precursor["z"] = P["z"]
        precursor["w"] = wp_prime / wn_prime

        # Sample group
        xi = rng(P_arr) * total
        total = 0.0
        for j in range(J):
            total += nu_d[j] / decay[j]
            if total > xi:
                break
        precursor["g"] = j

        # Set inducing neutron group
        precursor["n_g"] = g


# =============================================================================
# Population control techniques
# =============================================================================
# TODO: Make it a stand-alone function that takes (bank_init, bank_final, M).
#       The challenge is in the use of type-dependent copy_record which is
#       required due to pure-Python behavior of taking things by reference.


@njit
def population_control(seed, mcdc):
    if mcdc["technique"]["pct"] == PCT_COMBING:
        pct_combing(seed, mcdc)
    elif mcdc["technique"]["pct"] == PCT_COMBING_WEIGHT:
        pct_combing_weight(seed, mcdc)
    elif mcdc["technique"]["pct"] == PCT_SPLITTING_ROULETTE:
        pct_splitting_roulette(seed, mcdc)
    elif mcdc["technique"]["pct"] == PCT_SPLITTING_ROULETTE_WEIGHT:
        pct_splitting_roulette_weight(seed, mcdc)


@njit
def pct_combing(seed, mcdc):
    bank_census = mcdc["bank_census"]
    M = mcdc["setting"]["N_particle"]
    bank_source = mcdc["bank_source"]

    # Scan the bank
    idx_start, N_local, N = bank_scanning(bank_census, mcdc)
    idx_end = idx_start + N_local

    # Abort if census bank is empty
    if N == 0:
        return

    # Teeth distance
    td = N / M

    # Update population control factor
    mcdc["technique"]["pc_factor"] *= td

    xi = rng_from_seed(seed)
    offset = xi * td

    # First hiting tooth
    tooth_start = math.ceil((idx_start - offset) / td)

    # Last hiting tooth
    tooth_end = math.floor((idx_end - offset) / td) + 1

    P_rec_arr = adapt.local_array(1, type_.particle_record)
    P_rec = P_rec_arr[0]

    # Locally sample particles from census bank
    set_bank_size(bank_source, 0)
    for i in range(tooth_start, tooth_end):
        tooth = i * td + offset
        idx = math.floor(tooth) - idx_start
        split_as_record(P_rec_arr, bank_census["particles"][idx : idx + 1])
        # Set weight
        P_rec["w"] *= td
        adapt.add_source(P_rec_arr, mcdc)


@njit
def pct_combing_weight(seed, mcdc):
    bank_census = mcdc["bank_census"]
    M = mcdc["setting"]["N_particle"]
    bank_source = mcdc["bank_source"]

    # Scan the bank based on weight
    w_start, w_cdf, W = bank_scanning_weight(bank_census, mcdc)
    w_end = w_cdf[-1]

    # Abort if census bank is empty
    if W == 0.0:
        return

    # Teeth distance
    td = W / M

    # Update population control factor
    mcdc["technique"]["pc_factor"] *= td  # This may be incorrect

    # Tooth offset
    xi = rng_from_seed(seed)
    offset = xi * td

    # First hiting tooth
    tooth_start = math.ceil((w_start - offset) / td)

    # Last hiting tooth
    tooth_end = math.floor((w_end - offset) / td) + 1

    P_rec_arr = adapt.local_array(1, type_.particle_record)
    P_rec = P_rec_arr[0]

    # Locally sample particles from census bank
    set_bank_size(bank_source, 0)
    idx = 0
    for i in range(tooth_start, tooth_end):
        tooth = i * td + offset
        idx += binary_search(tooth, w_cdf[idx:])
        split_as_record(P_rec_arr, bank_census["particles"][idx : idx + 1])
        # Set weight
        P_rec["w"] = td
        adapt.add_source(P_rec_arr, mcdc)


@njit
def pct_splitting_roulette(seed, mcdc):
    bank_census = mcdc["bank_census"]
    M = mcdc["setting"]["N_particle"]
    bank_source = mcdc["bank_source"]

    # Scan the bank
    idx_start, N_local, N = bank_scanning(bank_census, mcdc)
    idx_end = idx_start + N_local

    # Abort if census bank is empty
    if N == 0:
        return

    # Weight scaling
    ws = float(N) / float(M)

    # Splitting Number
    sn = 1.0 / ws

    # Update population control factor
    mcdc["technique"]["pc_factor"] *= ws

    P_rec_arr = adapt.local_array(1, type_.particle_record)
    P_rec = P_rec_arr[0]

    # Perform split-roulette to all particles in local bank
    set_bank_size(bank_source, 0)
    for idx in range(N_local):
        # Weight of the surviving particles
        w = bank_census["particles"][idx]["w"]
        w_survive = w * ws

        # Determine number of guaranteed splits
        N_split = math.floor(sn)

        # Survive the russian roulette?
        xi = rng(bank_census["particles"][idx : idx + 1])
        if xi < sn - N_split:
            N_split += 1

        # Split the particle
        for i in range(N_split):
            split_as_record(P_rec_arr, bank_census["particles"][idx : idx + 1])
            # Set weight
            P_rec["w"] = w_survive
            adapt.add_source(P_rec_arr, mcdc)


@njit
def pct_splitting_roulette_weight(seed, mcdc):
    bank_census = mcdc["bank_census"]
    M = mcdc["setting"]["N_particle"]
    bank_source = mcdc["bank_source"]

    # Scan the bank based on weight
    N_local = get_bank_size(bank_census)
    w_start, w_cdf, W = bank_scanning_weight(bank_census, mcdc)
    w_end = w_cdf[-1]

    # Abort if census bank is empty
    if W == 0.0:
        return

    # Weight of the surviving particles
    w_survive = W / M

    # Update population control factor
    mcdc["technique"]["pc_factor"] *= w_survive  # This may be incorrect

    P_rec_arr = adapt.local_array(1, type_.particle_record)
    P_rec = P_rec_arr[0]

    # Perform split-roulette to all particles in local bank
    set_bank_size(bank_source, 0)
    for idx in range(N_local):
        # Splitting number
        w = bank_census["particles"][idx]["w"]
        sn = w / w_survive

        # Determine number of guaranteed splits
        N_split = math.floor(sn)

        # Survive the russian roulette?
        xi = rng(bank_census["particles"][idx : idx + 1])
        if xi < sn - N_split:
            N_split += 1

        # Split the particle
        for i in range(N_split):
            split_as_record(P_rec_arr, bank_census["particles"][idx : idx + 1])
            # Set weight
            P_rec["w"] = w_survive
            adapt.add_source(P_rec_arr, mcdc)


# =============================================================================
# Particle operations
# =============================================================================


@njit
def move_particle(P_arr, distance, mcdc):
    P = P_arr[0]
    P["x"] += P["ux"] * distance
    P["y"] += P["uy"] * distance
    P["z"] += P["uz"] * distance
    P["t"] += distance / physics.get_speed(P_arr, mcdc)


@njit
def copy_recordlike(P_new_arr, P_rec_arr):
    P_new = P_new_arr[0]
    P_rec = P_rec_arr[0]
    P_new["x"] = P_rec["x"]
    P_new["y"] = P_rec["y"]
    P_new["z"] = P_rec["z"]
    P_new["t"] = P_rec["t"]
    P_new["ux"] = P_rec["ux"]
    P_new["uy"] = P_rec["uy"]
    P_new["uz"] = P_rec["uz"]
    P_new["g"] = P_rec["g"]
    P_new["E"] = P_rec["E"]
    P_new["w"] = P_rec["w"]
    P_new["rng_seed"] = P_rec["rng_seed"]
    P_new["iqmc"]["w"] = P_rec["iqmc"]["w"]


@njit
def copy_particle(P_new_arr, P_arr):
    P_new = P_new_arr[0]
    P = P_arr[0]
    P_new = P_new_arr[0]
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["ux"] = P["ux"]
    P_new["uy"] = P["uy"]
    P_new["uz"] = P["uz"]
    P_new["g"] = P["g"]
    P_new["w"] = P["w"]
    P_new["alive"] = P["alive"]
    P_new["fresh"] = P["fresh"]
    P_new["material_ID"] = P["material_ID"]
    P_new["cell_ID"] = P["cell_ID"]
    P_new["surface_ID"] = P["surface_ID"]
    P_new["event"] = P["event"]
    P_new["rng_seed"] = P["rng_seed"]
    P_new["iqmc"]["w"] = P["iqmc"]["w"]


@njit
def recordlike_to_particle(P_new_arr, P_rec_arr):
    P_new = P_new_arr[0]
    P_rec = P_rec_arr[0]
    copy_recordlike(P_new_arr, P_rec_arr)
    P_new["fresh"] = True
    P_new["alive"] = True
    P_new["material_ID"] = -1
    P_new["cell_ID"] = -1
    P_new["surface_ID"] = -1
    P_new["event"] = -1


@njit
def split_as_record(P_new_rec_arr, P_rec_arr):
    P_rec = P_rec_arr[0]
    P_new_rec = P_new_rec_arr[0]
    copy_recordlike(P_new_rec_arr, P_rec_arr)
    P_new_rec["rng_seed"] = split_seed(P_rec["rng_seed"], SEED_SPLIT_PARTICLE)
    rng(P_rec_arr)


# =============================================================================
# Mesh operations
# =============================================================================


@njit
def mesh_get_angular_index(P_arr, mesh):
    P = P_arr[0]
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]

    P_mu = uz
    P_azi = math.acos(ux / math.sqrt(ux * ux + uy * uy))
    if uy < 0.0:
        P_azi *= -1

    mu = binary_search(P_mu, mesh["mu"])
    azi = binary_search(P_azi, mesh["azi"])
    return mu, azi


@njit
def mesh_get_energy_index(P_arr, mesh, mode_MG):
    P = P_arr[0]
    # Check if outside grid
    outside = False

    Ng = mesh["Ng"]

    if mode_MG:
        return binary_search_with_length(P["g"], mesh["g"], Ng), outside

    else:
        E = P["E"]
        if E < mesh["g"][0] or E > mesh["g"][Ng]:
            outside = True
            return 0, outside
        return binary_search_with_length(P["E"], mesh["g"], Ng), outside


# =============================================================================
# Tally operations
# =============================================================================


@njit
def score_mesh_tally(P_arr, distance, tally, data_tally, mcdc, mcdc_new):
    P = P_arr[0]
    material = objects.materials[P["material_ID"]]
    mesh = tally["filter"]
    stride = tally["stride"]

    # Particle 4D direction
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]
    ut = 1.0 / physics.get_speed(P_arr, mcdc, mcdc_new)

    # Particle initial and final coordinate
    x = P["x"]
    y = P["y"]
    z = P["z"]
    t = P["t"]
    x_final = x + ux * distance
    y_final = y + uy * distance
    z_final = z + uz * distance
    t_final = t + ut * distance

    # Easily identified tally bin indices
    mu, azi = mesh_get_angular_index(P_arr, mesh)
    g, outside_energy = mesh_get_energy_index(P_arr, mesh, objects.settings.multigroup_mode)

    # Get starting indices
    ix, iy, iz, it, outside = mesh_.structured.get_indices(P_arr, mesh)

    # Outside grid?
    if outside or outside_energy:
        return

    # The tally index
    idx = (
        stride["tally"]
        + mu * stride["mu"]
        + azi * stride["azi"]
        + g * stride["g"]
        + it * stride["t"]
        + ix * stride["x"]
        + iy * stride["y"]
        + iz * stride["z"]
    )

    # Sweep through the distance
    distance_swept = 0.0
    while distance_swept < distance - COINCIDENCE_TOLERANCE:
        # Find distances to the mesh grids
        if ux == 0.0:
            dx = INF
        else:
            if ux > 0.0:
                x_next = min(mesh["x"][ix + 1], x_final)
            else:
                x_next = max(mesh["x"][ix], x_final)
            dx = (x_next - x) / ux
        if uy == 0.0:
            dy = INF
        else:
            if uy > 0.0:
                y_next = min(mesh["y"][iy + 1], y_final)
            else:
                y_next = max(mesh["y"][iy], y_final)
            dy = (y_next - y) / uy
        if uz == 0.0:
            dz = INF
        else:
            if uz > 0.0:
                z_next = min(mesh["z"][iz + 1], z_final)
            else:
                z_next = max(mesh["z"][iz], z_final)
            dz = (z_next - z) / uz
        dt = (min(mesh["t"][it + 1], t_final) - t) / ut

        # Get the grid crossed
        distance_scored = INF
        mesh_crossed = MESH_NONE
        if dx <= distance_scored:
            mesh_crossed = MESH_X
            distance_scored = dx
        if dy <= distance_scored:
            mesh_crossed = MESH_Y
            distance_scored = dy
        if dz <= distance_scored:
            mesh_crossed = MESH_Z
            distance_scored = dz
        if dt <= distance_scored:
            mesh_crossed = MESH_T
            distance_scored = dt

        # Score
        flux = distance_scored * P["w"]
        mu = P["ux"]
        for i in range(tally["N_score"]):
            score_type = tally["scores"][i]
            score = 0
            if score_type == SCORE_FLUX:
                score = flux
            elif score_type == SCORE_DENSITY:
                score = flux * ut
            elif score_type == SCORE_TOTAL:
                SigmaT = get_MacroXS(XS_TOTAL, material, P_arr, mcdc)
                score = flux * SigmaT
            elif score_type == SCORE_FISSION:
                SigmaF = get_MacroXS(XS_FISSION, material, P_arr, mcdc)
                score = flux * SigmaF
            if score_type == SCORE_NET_CURRENT:
                score = flux * mu
            if score_type == SCORE_MU_SQ:
                score = flux * mu * mu
            elif score_type == SCORE_TIME_MOMENT_FLUX:
                score = flux * (t - (mesh["t"][it - 1] + mesh["t"][it]) / 2)
            elif score_type == SCORE_SPACE_MOMENT_FLUX:
                score = flux * (x - (mesh["x"][ix + 1] + mesh["x"][ix]) / 2)
            elif score_type == SCORE_TIME_MOMENT_CURRENT:
                score = flux * mu * (t - (mesh["t"][it - 1] + mesh["t"][it]) / 2)
            elif score_type == SCORE_SPACE_MOMENT_CURRENT:
                score = flux * mu * (x - (mesh["x"][ix + 1] + mesh["x"][ix]) / 2)
            elif score_type == SCORE_TIME_MOMENT_MU_SQ:
                score = flux * mu * mu * (t - (mesh["t"][it - 1] + mesh["t"][it]) / 2)
            elif score_type == SCORE_SPACE_MOMENT_MU_SQ:
                score = flux * mu * mu * (x - (mesh["x"][ix + 1] + mesh["x"][ix]) / 2)
            adapt.global_add(data_tally, (TALLY_SCORE, idx + i), round(score))

        # Accumulate distance swept
        distance_swept += distance_scored

        # Move the 4D position
        x += distance_scored * ux
        y += distance_scored * uy
        z += distance_scored * uz
        t += distance_scored * ut

        # Increment index and check if out of bound
        if mesh_crossed == MESH_X:
            if ux > 0.0:
                ix += 1
                if ix == mesh["Nx"]:
                    break
                idx += stride["x"]
            else:
                ix -= 1
                if ix == -1:
                    break
                idx -= stride["x"]
        elif mesh_crossed == MESH_Y:
            if uy > 0.0:
                iy += 1
                if iy == mesh["Ny"]:
                    break
                idx += stride["y"]
            else:
                iy -= 1
                if iy == -1:
                    break
                idx -= stride["y"]
        elif mesh_crossed == MESH_Z:
            if uz > 0.0:
                iz += 1
                if iz == mesh["Nz"]:
                    break
                idx += stride["z"]
            else:
                iz -= 1
                if iz == -1:
                    break
                idx -= stride["z"]
        elif mesh_crossed == MESH_T:
            it += 1
            if it == mesh["Nt"]:
                break
            idx += stride["t"]


@njit
def score_surface_tally(P_arr, surface, tally, data_tally, mcdc):
    # TODO: currently not supporting filters
    P = P_arr[0]

    stride = tally["stride"]

    # The tally index
    idx = stride["tally"]

    # Flux
    speed = physics.get_speed(P_arr, mcdc)
    mu = surface_.get_normal_component(P_arr, speed, surface)
    flux = P["w"] / abs(mu)

    # Score
    for i in range(tally["N_score"]):
        score_type = tally["scores"][i]
        if score_type == SCORE_FLUX:
            score = flux
        elif score_type == SCORE_NET_CURRENT:
            score = flux * mu
        adapt.global_add(data_tally, (TALLY_SCORE, idx + i), round(score))


@njit
def score_cell_tally(P_arr, distance, tally, data_tally, mcdc):
    P = P_arr[0]
    material = mcdc["materials"][P["material_ID"]]
    mesh = tally["filter"]
    stride = tally["stride"]

    # Particle/track properties
    ut = 1.0 / physics.get_speed(P_arr, mcdc)
    t = P["t"]
    t_final = t + ut * distance
    Nt = mesh["Nt"]

    # Get starting indices
    g, outside_energy = mesh_get_energy_index(P_arr, mesh, mcdc["setting"]["mode_MG"])

    # Outside grid?
    if (
        t < mesh["t"][0] - COINCIDENCE_TOLERANCE_TIME
        or t > mesh["t"][Nt] + COINCIDENCE_TOLERANCE_TIME
        or (abs(t - mesh["t"][Nt]) < COINCIDENCE_TOLERANCE_TIME)
        or outside_energy
    ):
        return

    it = mesh_.structured._grid_index(
        t, 1.0, mesh["t"], Nt + 1, COINCIDENCE_TOLERANCE_TIME
    )

    # Tally index
    idx = stride["tally"] + g * stride["g"] + it * stride["t"]

    # Sweep through the distance
    distance_swept = 0.0
    while distance_swept < distance - COINCIDENCE_TOLERANCE:

        # Find distance to mesh grids
        dt = (min(mesh["t"][it + 1], t_final) - t) / ut

        # Get the grid crossed
        distance_scored = INF
        mesh_crossed = MESH_NONE
        if dt <= distance_scored:
            mesh_crossed = MESH_T
            distance_scored = dt

        # Score
        flux = distance_scored * P["w"]
        for i in range(tally["N_score"]):
            score_type = tally["scores"][i]
            score = 0
            if score_type == SCORE_FLUX:
                score = flux
            elif score_type == SCORE_TOTAL:
                SigmaT = get_MacroXS(XS_TOTAL, material, P_arr, mcdc)
                score = flux * SigmaT
            elif score_type == SCORE_FISSION:
                SigmaF = get_MacroXS(XS_FISSION, material, P_arr, mcdc)
                score = flux * SigmaF
            adapt.global_add(data_tally, (TALLY_SCORE, idx + i), round(score))

        # Accumulate distance swept
        distance_swept += distance_scored

        # Move the 4D (just t for this tally) position
        t += distance_scored * ut

        # Increment index and check if out of bounds
        if mesh_crossed == MESH_T:
            it += 1
            if it == Nt:
                break
            idx += stride["t"]


@njit
def score_cs_tally(P_arr, distance, tally, data_tally, mcdc, mcdc_new):
    # Each time that this function is called, EVERY cs bin needs to be checked to see if the particle is in it.
    # The particle needs to score into all the bins that it is within
    P = P_arr[0]
    material = objects.materials[P["material_ID"]]
    N_cs_bins = tally["filter"]["N_cs_bins"]

    cs_bin_size = tally["filter"]["cs_bin_size"]
    cs_centers = tally["filter"]["cs_centers"]

    stride = tally["stride"]
    bin_idx = stride["tally"]

    # Particle 4D direction
    ux = P["ux"]
    uy = P["uy"]
    uz = P["uz"]
    ut = 1.0 / physics.get_speed(P_arr, mcdc, mcdc_new)

    # Particle initial and final coordinate
    x = P["x"]
    y = P["y"]
    z = P["z"]
    t = P["t"]
    x_final = x + ux * distance
    y_final = y + uy * distance
    z_final = z + uz * distance
    t_final = t + ut * distance

    # Check each coarse bin
    for j in range(N_cs_bins):
        center = adapt.local_array(2, type_.float64)
        start = adapt.local_array(2, type_.float64)
        end = adapt.local_array(2, type_.float64)
        #
        center[0] = cs_centers[0][j]
        center[1] = cs_centers[1][j]
        start[0] = x
        start[1] = y
        end[0] = x_final
        end[1] = y_final

        distance_inside = calculate_distance_in_coarse_bin(
            start, end, distance, center, cs_bin_size
        )

        # Last bin covers the whole problem
        if j == N_cs_bins - 1:
            cs_bin_size_full_problem = adapt.local_array(2, type_.float64)
            cs_bin_size_full_problem[0] = INF
            cs_bin_size_full_problem[1] = INF
            distance_inside = calculate_distance_in_coarse_bin(
                start, end, distance, center, cs_bin_size_full_problem
            )

        if distance < distance_inside:
            distance_in_bin = distance
        else:
            distance_in_bin = distance_inside

        # Calculate flux and other scores
        flux = distance_in_bin * P["w"]
        for i in range(tally["N_score"]):
            score_type = tally["scores"][i]
            if score_type == SCORE_FLUX:
                score = flux
            elif score_type == SCORE_DENSITY:
                score = flux / physics.get_speed(P_arr, mcdc)
            elif score_type == SCORE_TOTAL:
                SigmaT = get_MacroXS(XS_TOTAL, material, P_arr, mcdc)
                score = flux * SigmaT
            elif score_type == SCORE_FISSION:
                SigmaF = get_MacroXS(XS_FISSION, material, P_arr, mcdc)
                score = flux * SigmaF

            adapt.global_add(
                data_tally,
                (TALLY_SCORE, bin_idx + j * tally["N_score"] + i),
                round(score),
            )


@njit
def cs_clip(p, q, t0, t1):
    if p < 0:
        t = q / p
        if t > t1:
            return False, t0, t1
        if t > t0:
            t0 = t
    elif p > 0:
        t = q / p
        if t < t0:
            return False, t0, t1
        if t < t1:
            t1 = t
    elif q < 0:
        return False, t0, t1
    return True, t0, t1


@njit
def cs_tracklength_in_box(start, end, x_min, x_max, y_min, y_max):
    # Uses Liang-Barsky algorithm for finding tracklength in box
    t0, t1 = 0.0, 1.0
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # Perform clipping for each boundary
    result, t0, t1 = cs_clip(-dx, start[0] - x_min, t0, t1)
    if not result:
        return 0.0
    result, t0, t1 = cs_clip(dx, x_max - start[0], t0, t1)
    if not result:
        return 0.0
    result, t0, t1 = cs_clip(-dy, start[1] - y_min, t0, t1)
    if not result:
        return 0.0
    result, t0, t1 = cs_clip(dy, y_max - start[1], t0, t1)
    if not result:
        return 0.0

    # Update start and end points based on clipping results
    if t1 < 1:
        end[0] = start[0] + t1 * dx
        end[1] = start[1] + t1 * dy
    if t0 > 0:
        start[0] = start[0] + t0 * dx
        start[1] = start[1] + t0 * dx

    # Return the norm
    X = end[0] - start[0]
    Y = end[1] - start[1]
    return math.sqrt(X**2 + Y**2)


@njit
def calculate_distance_in_coarse_bin(start, end, distance, center, cs_bin_size):
    # Edges of the coarse bin
    x_min = center[0] - cs_bin_size[0] / 2
    x_max = center[0] + cs_bin_size[0] / 2
    y_min = center[1] - cs_bin_size[1] / 2
    y_max = center[1] + cs_bin_size[1] / 2

    distance_inside = cs_tracklength_in_box(start, end, x_min, x_max, y_min, y_max)

    return distance_inside


@njit
def dd_reduce(data_tally, mcdc):

    # find number of subdomains
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

    with objmode():
        # assign processors to their subdomain group
        i = 0
        for n in range(d_Nx * d_Ny * d_Nz):
            dd_group = []
            for r in range(int(mcdc["technique"]["dd_work_ratio"][n])):
                dd_group.append(i)
                i += 1
            # create MPI Comm group out of subdomain processors
            dd_group = MPI.COMM_WORLD.group.Incl(dd_group)
            dd_comm = MPI.COMM_WORLD.Create(dd_group)
            # MPI Reduce on subdomain processors
            buff = np.zeros_like(data_tally[TALLY_SCORE])
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Reduce(data_tally[TALLY_SCORE], buff, MPI.SUM, 0)
            if mcdc["dd_idx"] == n:
                data_tally[TALLY_SCORE][:] = buff
            # free comm group
            dd_group.Free()
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Free()


@njit
def tally_reduce(data_tally, mcdc):
    N_bin = data_tally.shape[1]

    # Normalize
    N_particle = mcdc["setting"]["N_particle"]
    for i in range(N_bin):
        data_tally[TALLY_SCORE][i] /= N_particle

    if not mcdc["technique"]["domain_decomposition"]:
        # MPI Reduce
        buff = np.zeros_like(data_tally[TALLY_SCORE])
        with objmode():
            MPI.COMM_WORLD.Reduce(data_tally[TALLY_SCORE], buff, MPI.SUM, 0)
        data_tally[TALLY_SCORE][:] = buff

    else:
        # find number of subdomains
        N_dd = 1
        N_dd *= mcdc["technique"]["dd_mesh"]["x"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["y"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["z"].size - 1
        # DD Reduce if multiple processors per subdomain
        if N_dd != mcdc["mpi_size"]:
            dd_reduce(data_tally, mcdc)


@njit
def tally_accumulate(data_tally, mcdc):
    N_bin = data_tally.shape[1]

    for i in range(N_bin):
        # Accumulate score and square of score into sum and sum_sq
        score = data_tally[TALLY_SCORE, i]
        data_tally[TALLY_SUM, i] += score
        data_tally[TALLY_SUM_SQ, i] += score * score

        # Reset score bin
        data_tally[TALLY_SCORE, i] = 0.0


@njit
def census_based_tally_output(data_tally, mcdc):
    idx_batch = mcdc["idx_batch"]
    idx_census = mcdc["idx_census"]
    N_bin = data_tally.shape[1]

    for i in range(N_bin):
        # Store score and square of score
        score = data_tally[TALLY_SCORE, i]
        data_tally[TALLY_SUM, i] = score
        data_tally[TALLY_SUM_SQ, i] = score * score

        # Reset score bin
        data_tally[TALLY_SCORE, i] = 0.0

    for ID, tally in enumerate(mcdc["mesh_tallies"]):
        mesh = tally["filter"]

        # Get grid
        Nx = mesh["Nx"]
        Ny = mesh["Ny"]
        Nz = mesh["Nz"]
        Nt = mesh["Nt"]
        Nmu = mesh["Nmu"]
        N_azi = mesh["N_azi"]
        Ng = mesh["Ng"]
        #
        grid_x = mesh["x"][: Nx + 1]
        grid_y = mesh["y"][: Ny + 1]
        grid_z = mesh["z"][: Nz + 1]
        grid_t = mesh["t"][: Nt + 1]
        grid_mu = mesh["mu"][: Nmu + 1]
        grid_azi = mesh["azi"][: N_azi + 1]
        grid_g = mesh["g"][: Ng + 1]
        #'''
        with objmode():
            if ID == 0:
                f = h5py.File(
                    mcdc["setting"]["output_name"]
                    + "-batch_%i-census_%i.h5" % (idx_batch, idx_census),
                    "w",
                )
            else:
                f = h5py.File(
                    mcdc["setting"]["output_name"]
                    + "-batch_%i-census_%i.h5" % (idx_batch, idx_census),
                    "a",
                )
            # Save to dataset
            f.create_dataset("tallies/mesh_tally_%i/grid/x" % ID, data=grid_x)
            f.create_dataset("tallies/mesh_tally_%i/grid/y" % ID, data=grid_y)
            f.create_dataset("tallies/mesh_tally_%i/grid/z" % ID, data=grid_z)
            f.create_dataset("tallies/mesh_tally_%i/grid/t" % ID, data=grid_t)
            f.create_dataset("tallies/mesh_tally_%i/grid/mu" % ID, data=grid_mu)
            f.create_dataset("tallies/mesh_tally_%i/grid/azi" % ID, data=grid_azi)
            f.create_dataset("tallies/mesh_tally_%i/grid/g" % ID, data=grid_g)

            # Set tally shape
            N_score = tally["N_score"]
            if not mcdc["technique"]["uq"]:
                shape = (3, Nmu, N_azi, Ng, Nt, Nx, Ny, Nz, N_score)
            else:
                shape = (5, Nmu, N_azi, Ng, Nt, Nx, Ny, Nz, N_score)

            # Reshape tally
            N_bin = tally["N_bin"]
            start = tally["stride"]["tally"]
            tally_bin = data_tally[:, start : start + N_bin]
            tally_bin = tally_bin.reshape(shape)

            # Roll tally so that score is in the front
            tally_bin = np.rollaxis(tally_bin, 8, 0)

            # Iterate over scores
            for i in range(N_score):
                score_type = tally["scores"][i]
                score_tally_bin = np.squeeze(tally_bin[i])
                score_name = ""
                if score_type == SCORE_FLUX:
                    score_name = "flux"
                elif score_type == SCORE_DENSITY:
                    score_name = "density"
                elif score_type == SCORE_TOTAL:
                    score_name = "total"
                elif score_type == SCORE_FISSION:
                    score_name = "fission"
                group_name = "tallies/mesh_tally_%i/%s/" % (ID, score_name)

                tally_sum = score_tally_bin[TALLY_SUM]
                tally_sum_sq = score_tally_bin[TALLY_SUM_SQ]

                f.create_dataset(group_name + "score", data=tally_sum)
                f.create_dataset(group_name + "score_sq", data=tally_sum_sq)
                if mcdc["technique"]["uq"]:
                    mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                    tot_var = score_tally_bin[TALLY_UQ_BATCH]
                    uq_var = tot_var - mc_var
                    f.create_dataset(group_name + "uq_var", data=uq_var)
            f.close()


@njit
def dd_closeout(data_tally, mcdc):
    # find number of subdomains
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

    with objmode():
        # assign processors to their subdomain group
        i = 0
        for n in range(d_Nx * d_Ny * d_Nz):
            dd_ranks = []
            for r in range(int(mcdc["technique"]["dd_work_ratio"][n])):
                dd_ranks.append(i)
                i += 1
            # create MPI Comm group out of subdomain processors
            dd_group = MPI.COMM_WORLD.group.Incl(dd_ranks)
            dd_comm = MPI.COMM_WORLD.Create(dd_group)
            # MPI Reduce on subdomain processors
            buff = np.zeros_like(data_tally[TALLY_SUM])
            buff_sq = np.zeros_like(data_tally[TALLY_SUM_SQ])
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Reduce(data_tally[TALLY_SUM], buff, MPI.SUM, 0)
                dd_comm.Reduce(data_tally[TALLY_SUM_SQ], buff_sq, MPI.SUM, 0)
            if mcdc["dd_idx"] == n:
                data_tally[TALLY_SUM] = buff
                data_tally[TALLY_SUM_SQ] = buff_sq

            # free comm group
            dd_group.Free()
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Free()


@njit
def tally_closeout(data_tally, mcdc):
    N_history = mcdc["setting"]["N_particle"]

    if mcdc["setting"]["N_batch"] > 1:
        N_history = mcdc["setting"]["N_batch"]

    elif mcdc["setting"]["mode_eigenvalue"]:
        N_history = mcdc["setting"]["N_active"]

    elif not mcdc["technique"]["domain_decomposition"]:
        # MPI Reduce
        buff = np.zeros_like(data_tally[TALLY_SUM])
        buff_sq = np.zeros_like(data_tally[TALLY_SUM_SQ])
        with objmode():
            MPI.COMM_WORLD.Reduce(data_tally[TALLY_SUM], buff, MPI.SUM, 0)
            MPI.COMM_WORLD.Reduce(data_tally[TALLY_SUM_SQ], buff_sq, MPI.SUM, 0)
        data_tally[TALLY_SUM] = buff
        data_tally[TALLY_SUM_SQ] = buff_sq

    else:
        # find number of subdomains
        N_dd = 1
        N_dd *= mcdc["technique"]["dd_mesh"]["x"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["y"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["z"].size - 1
        # DD Reduce if multiple processors per subdomain
        if N_dd != mcdc["mpi_size"]:
            dd_closeout(data_tally, mcdc)
    # Calculate and store statistics
    #   sum --> mean
    #   sum_sq --> standard deviation
    N_bin = data_tally.shape[1]
    for i in range(N_bin):
        data_tally[TALLY_SUM][i] = data_tally[TALLY_SUM][i] / N_history
        radicand = (
            data_tally[TALLY_SUM_SQ][i] / N_history
            - np.square(data_tally[TALLY_SUM][i])
        ) / (N_history - 1)
        # Check for round-off error
        if abs(radicand) < 1e-18:
            data_tally[TALLY_SUM_SQ][i] = 0.0
        else:
            data_tally[TALLY_SUM_SQ][i] = np.sqrt(radicand)


# =============================================================================
# Eigenvalue tally operations
# =============================================================================


@njit
def eigenvalue_tally(P_arr, distance, mcdc):
    P = P_arr[0]
    material = mcdc["materials"][P["material_ID"]]
    flux = distance * P["w"]

    # Get nu-fission
    nuSigmaF = get_MacroXS(XS_NU_FISSION, material, P_arr, mcdc)

    # Fission production (needed even during inactive cycle)
    # mcdc["eigenvalue_tally_nuSigmaF"][0] += flux * nuSigmaF
    adapt.global_add(mcdc["eigenvalue_tally_nuSigmaF"], 0, round(flux * nuSigmaF))

    if mcdc["cycle_active"]:
        # Neutron density
        v = physics.get_speed(P_arr, mcdc)
        n_density = flux / v
        # mcdc["eigenvalue_tally_n"][0] += n_density
        adapt.global_add(mcdc["eigenvalue_tally_n"], 0, round(n_density))
        # Maximum neutron density
        if mcdc["n_max"] < n_density:
            mcdc["n_max"] = n_density

        # Precursor density
        J = material["J"]
        SigmaF = get_MacroXS(XS_FISSION, material, P_arr, mcdc)
        # Get the decay-wighted multiplicity
        total = 0.0
        if mcdc["setting"]["mode_MG"]:
            g = P["g"]
            for i in range(material["N_nuclide"]):
                ID_nuclide = material["nuclide_IDs"][i]
                nuclide = mcdc["nuclides"][ID_nuclide]
                for j in range(J):
                    nu_d = nuclide["nu_d"][g, j]
                    decay = nuclide["decay"][j]
                    total += nu_d / decay
        else:
            E = P["E"]
            for i in range(material["N_nuclide"]):
                ID_nuclide = material["nuclide_IDs"][i]
                nuclide = mcdc["nuclides"][ID_nuclide]
                if not nuclide["fissionable"]:
                    continue
                for j in range(J):
                    nu_d = get_nu_group(NU_FISSION_DELAYED, nuclide, E, j)
                    decay = nuclide["ce_decay"][j]
                    total += nu_d / decay
        C_density = flux * total * SigmaF / mcdc["k_eff"]
        # mcdc["eigenvalue_tally_C"][0] += C_density
        adapt.global_add(mcdc["eigenvalue_tally_C"], 0, round(C_density))
        # Maximum precursor density
        if mcdc["C_max"] < C_density:
            mcdc["C_max"] = C_density


@njit
def eigenvalue_tally_closeout_history(mcdc):
    N_particle = mcdc["setting"]["N_particle"]

    idx_cycle = mcdc["idx_cycle"]

    # MPI Allreduce
    buff_nuSigmaF = np.zeros(1, np.float64)
    buff_n = np.zeros(1, np.float64)
    buff_nmax = np.zeros(1, np.float64)
    buff_C = np.zeros(1, np.float64)
    buff_Cmax = np.zeros(1, np.float64)
    buff_IC_fission = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(
            np.array(mcdc["eigenvalue_tally_nuSigmaF"]), buff_nuSigmaF, MPI.SUM
        )
        if mcdc["cycle_active"]:
            MPI.COMM_WORLD.Allreduce(
                np.array(mcdc["eigenvalue_tally_n"]), buff_n, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["n_max"]]), buff_nmax, MPI.MAX)
            MPI.COMM_WORLD.Allreduce(
                np.array(mcdc["eigenvalue_tally_C"]), buff_C, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["C_max"]]), buff_Cmax, MPI.MAX)
            if mcdc["technique"]["IC_generator"]:
                MPI.COMM_WORLD.Allreduce(
                    np.array(mcdc["technique"]["IC_fission_score"]),
                    buff_IC_fission,
                    MPI.SUM,
                )

    # Update and store k_eff
    mcdc["k_eff"] = buff_nuSigmaF[0] / N_particle
    mcdc["k_cycle"][idx_cycle] = mcdc["k_eff"]

    # Normalize other eigenvalue/global tallies
    tally_n = buff_n[0] / N_particle
    tally_C = buff_C[0] / N_particle
    tally_IC_fission = buff_IC_fission[0]

    # Maximum densities
    mcdc["n_max"] = buff_nmax[0]
    mcdc["C_max"] = buff_Cmax[0]

    # Accumulate running average
    if mcdc["cycle_active"]:
        mcdc["k_avg"] += mcdc["k_eff"]
        mcdc["k_sdv"] += mcdc["k_eff"] * mcdc["k_eff"]
        mcdc["n_avg"] += tally_n
        mcdc["n_sdv"] += tally_n * tally_n
        mcdc["C_avg"] += tally_C
        mcdc["C_sdv"] += tally_C * tally_C

        N = 1 + mcdc["idx_cycle"] - mcdc["setting"]["N_inactive"]
        mcdc["k_avg_running"] = mcdc["k_avg"] / N
        if N == 1:
            mcdc["k_sdv_running"] = 0.0
        else:
            mcdc["k_sdv_running"] = math.sqrt(
                (mcdc["k_sdv"] / N - mcdc["k_avg_running"] ** 2) / (N - 1)
            )

        if mcdc["technique"]["IC_generator"]:
            mcdc["technique"]["IC_fission"] += tally_IC_fission

    # Reset accumulators
    mcdc["eigenvalue_tally_nuSigmaF"][0] = 0.0
    mcdc["eigenvalue_tally_n"][0] = 0.0
    mcdc["eigenvalue_tally_C"][0] = 0.0
    mcdc["technique"]["IC_fission_score"][0] = 0.0

    # =====================================================================
    # Gyration radius
    # =====================================================================

    if mcdc["setting"]["gyration_radius"]:
        # Center of mass
        N_local = get_bank_size(mcdc["bank_census"])
        total_local = np.zeros(4, np.float64)  # [x,y,z,W]
        total = np.zeros(4, np.float64)
        for i in range(N_local):
            P = mcdc["bank_census"]["particles"][i]
            total_local[0] += P["x"] * P["w"]
            total_local[1] += P["y"] * P["w"]
            total_local[2] += P["z"] * P["w"]
            total_local[3] += P["w"]
        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(total_local, total, MPI.SUM)
        # COM
        W = total[3]
        com_x = total[0] / W
        com_y = total[1] / W
        com_z = total[2] / W

        # Distance RMS
        rms_local = np.zeros(1, np.float64)
        rms = np.zeros(1, np.float64)
        gr_type = mcdc["setting"]["gyration_radius_type"]
        if gr_type == GYRATION_RADIUS_ALL:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += (
                    (P["x"] - com_x) ** 2
                    + (P["y"] - com_y) ** 2
                    + (P["z"] - com_z) ** 2
                ) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_X:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Y:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Z:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_X:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Y:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Z:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["z"] - com_z) ** 2) * P["w"]

        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(rms_local, rms, MPI.SUM)
        rms = math.sqrt(rms[0] / W)

        # Gyration radius
        mcdc["gyration_radius"][idx_cycle] = rms


@njit
def eigenvalue_tally_closeout(mcdc):
    N = mcdc["setting"]["N_active"]
    mcdc["n_avg"] /= N
    mcdc["C_avg"] /= N
    if N > 1:
        mcdc["n_sdv"] = math.sqrt((mcdc["n_sdv"] / N - mcdc["n_avg"] ** 2) / (N - 1))
        mcdc["C_sdv"] = math.sqrt((mcdc["C_sdv"] / N - mcdc["C_avg"] ** 2) / (N - 1))
    else:
        mcdc["n_sdv"] = 0.0
        mcdc["C_sdv"] = 0.0


# ======================================================================================
# Move to event
# ======================================================================================


@njit
def move_to_event(P_arr, data_tally, mcdc, mcdc_new):
    # ==================================================================================
    # Preparation (as needed)
    # ==================================================================================
    P = P_arr[0]

    # Multigroup preparation
    #   In MG mode, particle speed is material-dependent.
    if mcdc["setting"]["mode_MG"]:
        # If material is not identified yet, locate the particle
        if P["material_ID"] == -1:
            if not geometry.locate_particle(P_arr, mcdc):
                # Particle is lost
                P["event"] = EVENT_LOST
                return

    # ==================================================================================
    # Geometry inspection
    # ==================================================================================
    #   - Set particle top cell and material IDs (if not lost)
    #   - Set surface ID (if surface hit)
    #   - Set particle boundary event (surface or lattice crossing, or lost)
    #   - Return distance to boundary (surface or lattice)

    d_boundary = geometry.inspect_geometry(P_arr, mcdc, mcdc_new)

    # Particle is lost?
    if P["event"] == EVENT_LOST:
        return

    # ==================================================================================
    # Get distances to other events
    # ==================================================================================

    # Distance to domain
    speed = physics.get_speed(P_arr, mcdc, mcdc_new)
    d_domain = INF
    if mcdc["technique"]["domain_decomposition"]:
        d_domain = mesh_.structured.get_crossing_distance(
            P_arr, speed, mcdc["technique"]["dd_mesh"]
        )

    # Distance to time boundary
    d_time_boundary = speed * (mcdc["setting"]["time_boundary"] - P["t"])

    # Distance to census time
    idx = mcdc["idx_census"]
    d_time_census = speed * (mcdc["setting"]["census_time"][idx] - P["t"])

    # Distance to next collision
    d_collision = distance_to_collision(P_arr, mcdc)

    # =========================================================================
    # Determine event(s)
    # =========================================================================
    # TODO: Make a function to better maintain the repeating operation

    distance = d_boundary

    # Check distance to domain
    if d_domain < distance - COINCIDENCE_TOLERANCE:
        distance = d_domain
        P["event"] = EVENT_DOMAIN_CROSSING
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_domain, distance):
        P["event"] += EVENT_DOMAIN_CROSSING

    # Check distance to collision
    if d_collision < distance - COINCIDENCE_TOLERANCE:
        distance = d_collision
        P["event"] = EVENT_COLLISION
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_collision, distance):
        P["event"] += EVENT_COLLISION

    # Check distance to time census
    if d_time_census < distance - COINCIDENCE_TOLERANCE:
        distance = d_time_census
        P["event"] = EVENT_TIME_CENSUS
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_time_census, distance):
        P["event"] += EVENT_TIME_CENSUS

    # Check distance to time boundary (exclusive event)
    if d_time_boundary < distance + COINCIDENCE_TOLERANCE:
        distance = d_time_boundary
        P["event"] = EVENT_TIME_BOUNDARY
        P["surface_ID"] = -1

    # =========================================================================
    # Move particle
    # =========================================================================

    # Score tracklength tallies
    if mcdc["cycle_active"]:
        # Mesh tallies
        for tally in mcdc["mesh_tallies"]:
            score_mesh_tally(P_arr, distance, tally, data_tally, mcdc, mcdc_new)

        # Cell tallies
        cell = mcdc["cells"][P["cell_ID"]]
        for i in range(cell["N_tally"]):
            ID = cell["tally_IDs"][i]
            tally = mcdc["cell_tallies"][ID]
            score_cell_tally(P_arr, distance, tally, data_tally, mcdc)

        # CS tallies
        for tally in mcdc["cs_tallies"]:
            score_cs_tally(P_arr, distance, tally, data_tally, mcdc)

    if mcdc["setting"]["mode_eigenvalue"]:
        eigenvalue_tally(P_arr, distance, mcdc)

    # Move particle
    move_particle(P_arr, distance, mcdc)


@njit
def distance_to_collision(P_arr, mcdc):
    P = P_arr[0]
    # Get total cross-section
    material = objects.materials[P["material_ID"]]
    SigmaT = get_macro_xs(REACTION_TOTAL, material, P_arr, mcdc)

    # Vacuum material?
    if SigmaT == 0.0:
        return INF

    # Sample collision distance
    xi = rng(P_arr)
    distance = -math.log(xi) / SigmaT
    return distance


# =============================================================================
# Surface crossing
# =============================================================================


@njit
def surface_crossing(P_arr, data_tally, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    # Apply BC
    surface = mcdc["surfaces"][P["surface_ID"]]
    if surface["BC"] == BC_VACUUM:
        P["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_.reflect(P_arr, surface)

    # Score tally
    # N_tally is an int64, tally_IDs is a numpy.ndarray
    for i in range(surface["N_tally"]):
        ID = surface["tally_IDs"][i]
        tally = mcdc["surface_tallies"][ID]
        score_surface_tally(P_arr, surface, tally, data_tally, mcdc)

    # Need to check new cell later?
    if P["alive"] and not surface["BC"] == BC_REFLECTIVE:
        P["cell_ID"] = -1
        P["material_ID"] = -1


# =============================================================================
# Collision
# =============================================================================


@njit
def collision(P_arr, mcdc):
    P = P_arr[0]

    # Get the reaction cross-sections
    material = objects.materials[P["material_ID"]]
    SigmaT = get_macro_xs(REACTION_TOTAL, material, P_arr, mcdc)
    SigmaS = get_macro_xs(REACTION_ELASTIC_SCATTERING, material, P_arr, mcdc)
    SigmaC = get_macro_xs(REACTION_CAPTURE, material, P_arr, mcdc)
    SigmaF = get_macro_xs(REACTION_FISSION, material, P_arr, mcdc)

    # Implicit capture
    if mcdc["technique"]["implicit_capture"]:
        P["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    # Sample collision type
    xi = rng(P_arr) * SigmaT
    tot = SigmaS
    if tot > xi:
        P["event"] += EVENT_SCATTERING
    else:
        tot += SigmaF
        if tot > xi:
            P["event"] += EVENT_FISSION
        else:
            P["event"] += EVENT_CAPTURE


# =============================================================================
# Scattering
# =============================================================================


@njit
def scattering(P_arr, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)
    # Kill the current particle
    P["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = P["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P["w"]

    # Get number of secondaries
    material = objects.materials[P["material_ID"]]
    g = P["g"]
    if mcdc["setting"]["mode_MG"]:
        nu_s = material["nu_s"][g]
        N = int(math.floor(weight_eff * nu_s + rng(P_arr)))
    else:
        N = 1

    P_new_arr = adapt.local_array(1, type_.particle_record)
    P_new = P_new_arr[0]

    for n in range(N):
        # Create new particle
        split_as_record(P_new_arr, P_arr)

        # Set weight
        P_new["w"] = weight_new

        # Sample scattering phase space
        sample_phasespace_scattering(P_arr, material, P_new_arr, mcdc)

        # Bank, but keep it if it is the last particle
        if n == N - 1:
            P["alive"] = True
            P["ux"] = P_new["ux"]
            P["uy"] = P_new["uy"]
            P["uz"] = P_new["uz"]
            P["g"] = P_new["g"]
            P["E"] = P_new["E"]
            P["w"] = P_new["w"]
        else:
            adapt.add_active(P_new_arr, prog)


@njit
def sample_phasespace_scattering(P_arr, material, P_new_arr, mcdc):
    P_new = P_new_arr[0]
    P = P_arr[0]
    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]

    if mcdc["setting"]["mode_MG"]:
        scattering_MG(P_arr, material, P_new_arr)
    else:
        scattering_CE(P_arr, material, P_new_arr, mcdc)


@njit
def sample_phasespace_scattering_nuclide(P_arr, nuclide, P_new_arr):
    P_new = P_new_arr[0]
    P = P_arr[0]
    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]

    scattering_MG(P_arr, nuclide, P_new_arr)


@njit
def scattering_MG(P_arr, material, P_new_arr):
    P_new = P_new_arr[0]
    P = P_arr[0]
    # Sample scattering angle
    mu0 = 2.0 * rng(P_new_arr) - 1.0

    # Scatter direction
    azi = 2.0 * PI * rng(P_new_arr)
    P_new["ux"], P_new["uy"], P_new["uz"] = scatter_direction(
        P["ux"], P["uy"], P["uz"], mu0, azi
    )

    # Get outgoing spectrum
    g = P["g"]
    G = material["G"]
    chi_s = material["chi_s"][g]

    # Sample outgoing energy
    xi = rng(P_new_arr)
    tot = 0.0
    for g_out in range(G):
        tot += chi_s[g_out]
        if tot > xi:
            break
    P_new["g"] = g_out


@njit
def scattering_CE(P_arr, material, P_new_arr, mcdc):
    P_new = P_new_arr[0]
    P = P_arr[0]
    """
    Scatter with sampled scattering angle mu0, with nucleus mass A
    Scattering is treated in Center of mass (COM) frame
    Current model:
      - Free gas scattering
      - Constant thermal cross section
      - Isotropic in COM
    """
    # Sample nuclide
    nuclide = sample_nuclide(material, P_arr, REACTION_ELASTIC_SCATTERING, mcdc)

    # Get the reaction
    for reaction in nuclide.reactions:
        if reaction.type_enum == REACTION_ELASTIC_SCATTERING:
            break

    # Sample nucleus thermal speed
    A = nuclide.atomic_weight_ratio
    if P["E"] > E_THERMAL_THRESHOLD:
        Vx = 0.0
        Vy = 0.0
        Vz = 0.0
    else:
        Vx, Vy, Vz = sample_nucleus_speed(A, P_arr, mcdc)

    # =========================================================================
    # COM kinematics
    # =========================================================================

    # Particle speed
    P_speed = physics.get_speed(P_arr, mcdc)

    # Neutron velocity - LAB
    vx = P_speed * P["ux"]
    vy = P_speed * P["uy"]
    vz = P_speed * P["uz"]

    # COM velocity
    COM_x = (vx + A * Vx) / (1.0 + A)
    COM_y = (vy + A * Vy) / (1.0 + A)
    COM_z = (vz + A * Vz) / (1.0 + A)

    # Neutron velocity - COM
    vx = vx - COM_x
    vy = vy - COM_y
    vz = vz - COM_z

    # Neutron speed - COM
    P_speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Neutron initial direction - COM
    ux = vx / P_speed
    uy = vy / P_speed
    uz = vz / P_speed

    # ==================================================================================
    # Sample scattering cosine
    # ==================================================================================
    
    # Interpolation factor
    E_grid = reaction.mu_energy_grid
    idx = binary_search(P['E'], E_grid)
    E0 = E_grid[idx]
    E1 = E_grid[idx + 1]
    f = (P['E'] - E0) - (E1 - E0)

    # Sample which table to choose
    xi = rng(P_new_arr)
    if xi > f:
        idx += 1

    # Get the table
    start = reaction.mu_energy_offset[idx]
    if idx + 1 == len(reaction.mu_energy_grid):
        end = len(reaction.mu_energy_grid)
    else:
        end = reaction.mu_energy_offset[idx + 1]
    mu = reaction.mu[start:end]
    PDF = reaction.mu_PDF[start:end]
    CDF = reaction.mu_CDF[start:end]

    # Sample bin index
    xi = rng(P_new_arr)
    idx = binary_search(xi, CDF)
    m = (PDF[idx + 1] - PDF[idx]) / (mu[idx + 1] - mu[idx])
    p = PDF[idx]
    c = CDF[idx]
    if m == 0.0:
        mu0 = mu[idx] + (xi - c) / p
    else:
        mu0 = mu[idx] + 1.0 / m * (math.sqrt(p**2 + 2 * m * (xi - c)) - p)

    # Scatter the direction in COM
    azi = 2.0 * PI * rng(P_arr)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Neutron final velocity - COM
    vx = P_speed * ux_new
    vy = P_speed * uy_new
    vz = P_speed * uz_new

    # =========================================================================
    # COM to LAB
    # =========================================================================

    # Final velocity - LAB
    vx = vx + COM_x
    vy = vy + COM_y
    vz = vz + COM_z

    # Final energy - LAB
    P_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    P_new["E"] = 5.2270376e-13 * P_speed * P_speed
    # constant: 0.5 / (1.60217662e-19 J/eV) * (1.674927471e-27 kg) / (10000 cm^2/m^2)

    # Final direction - LAB
    P_new["ux"] = vx / P_speed
    P_new["uy"] = vy / P_speed
    P_new["uz"] = vz / P_speed


@njit
def sample_nucleus_speed(A, P_arr, mcdc):
    P = P_arr[0]
    # Particle speed
    P_speed = physics.get_speed(P_arr, mcdc)

    # Maxwellian parameter
    beta = math.sqrt(2.0659834e-11 * A)
    # The constant above is
    #   (1.674927471e-27 kg) / (1.38064852e-19 cm^2 kg s^-2 K^-1) / (293.6 K)/2

    # Sample nuclide speed candidate V_tilda and
    #   nuclide-neutron polar cosine candidate mu_tilda via
    #   rejection sampling
    y = beta * P_speed
    while True:
        if rng(P_arr) < 2.0 / (2.0 + PI_SQRT * y):
            x = math.sqrt(-math.log(rng(P_arr) * rng(P_arr)))
        else:
            cos_val = math.cos(PI_HALF * rng(P_arr))
            x = math.sqrt(
                -math.log(rng(P_arr)) - math.log(rng(P_arr)) * cos_val * cos_val
            )
        V_tilda = x / beta
        mu_tilda = 2.0 * rng(P_arr) - 1.0

        # Accept candidate V_tilda and mu_tilda?
        if rng(P_arr) > math.sqrt(
            P_speed * P_speed + V_tilda * V_tilda - 2.0 * P_speed * V_tilda * mu_tilda
        ) / (P_speed + V_tilda):
            break

    # Set nuclide velocity - LAB
    azi = 2.0 * PI * rng(P_arr)
    ux, uy, uz = scatter_direction(P["ux"], P["uy"], P["uz"], mu_tilda, azi)
    Vx = ux * V_tilda
    Vy = uy * V_tilda
    Vz = uz * V_tilda

    return Vx, Vy, Vz


@njit
def scatter_direction(ux, uy, uz, mu0, azi):
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu0**2) ** 0.5

    if uz != 1.0:
        B = (1.0 - uz**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uz * cos_azi - uy * sin_azi) * C
        uy_new = uy * mu0 + (uy * uz * cos_azi + ux * sin_azi) * C
        uz_new = uz * mu0 - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the scattering formula
    else:
        B = (1.0 - uy**2) ** 0.5
        C = Ac / B

        ux_new = ux * mu0 + (ux * uy * cos_azi - uz * sin_azi) * C
        uz_new = uz * mu0 + (uz * uy * cos_azi + ux * sin_azi) * C
        uy_new = uy * mu0 - cos_azi * Ac * B

    return ux_new, uy_new, uz_new


# =============================================================================
# Fission
# =============================================================================


@njit
def fission(P_arr, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    # Kill the current particle
    P["alive"] = False

    # Get effective and new weight
    if mcdc["technique"]["weighted_emission"]:
        weight_eff = P["w"]
        weight_new = 1.0
    else:
        weight_eff = 1.0
        weight_new = P["w"]

    # Sample nuclide if CE
    material = mcdc["materials"][P["material_ID"]]
    nuclide = mcdc["nuclides"][0]  # Default nuclide, will be resampled for CE

    # Get number of secondaries
    if mcdc["setting"]["mode_MG"]:
        g = P["g"]
        nu = material["nu_f"][g]
    else:
        nuclide = sample_nuclide(material, P_arr, XS_FISSION, mcdc)
        E = P["E"]
        nu = get_nu(NU_FISSION, nuclide, E)
    N = int(math.floor(weight_eff * nu / mcdc["k_eff"] + rng(P_arr)))

    P_new_arr = adapt.local_array(1, type_.particle_record)
    P_new = P_new_arr[0]

    for n in range(N):
        # Create new particle
        split_as_record(P_new_arr, P_arr)

        # Set weight
        P_new["w"] = weight_new

        # Sample fission neutron phase space
        if mcdc["setting"]["mode_MG"]:
            sample_phasespace_fission(P_arr, material, P_new_arr, mcdc)
        else:
            sample_phasespace_fission_nuclide(P_arr, nuclide, P_new_arr, mcdc)

        # Eigenvalue mode: bank right away
        if mcdc["setting"]["mode_eigenvalue"]:
            adapt.add_census(P_new_arr, prog)
            continue
        # Below is only relevant for fixed-source problem

        # Skip if it's beyond time boundary
        if P_new["t"] > mcdc["setting"]["time_boundary"]:
            continue

        # Check if it is beyond current or next census times
        hit_census = False
        hit_next_census = False
        idx_census = mcdc["idx_census"]
        if idx_census < mcdc["setting"]["N_census"] - 1:
            if P["t"] > mcdc["setting"]["census_time"][idx_census + 1]:
                hit_census = True
                hit_next_census = True
            elif P_new["t"] > mcdc["setting"]["census_time"][idx_census]:
                hit_census = True

        if not hit_census:
            # Keep it if it is the last particle
            if n == N - 1:
                P["alive"] = True
                P["ux"] = P_new["ux"]
                P["uy"] = P_new["uy"]
                P["uz"] = P_new["uz"]
                P["t"] = P_new["t"]
                P["g"] = P_new["g"]
                P["E"] = P_new["E"]
                P["w"] = P_new["w"]
            else:
                adapt.add_active(P_new_arr, prog)
        elif not hit_next_census:
            # Particle will participate after the current census
            adapt.add_census(P_new_arr, prog)
        else:
            # Particle will participate in the future
            adapt.add_future(P_new_arr, prog)


@njit
def sample_phasespace_fission(P_arr, material, P_new_arr, mcdc):
    P_new = P_new_arr[0]
    P = P_arr[0]
    # Get constants
    G = material["G"]
    J = material["J"]
    g = P["g"]
    nu = material["nu_f"][g]
    nu_p = material["nu_p"][g]
    if J > 0:
        nu_d = material["nu_d"][g]

    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]

    # Sample isotropic direction
    P_new["ux"], P_new["uy"], P_new["uz"] = sample_isotropic_direction(P_new_arr)

    # Prompt or delayed?
    xi = rng(P_new_arr) * nu
    tot = nu_p
    if xi < tot:
        prompt = True
        spectrum = material["chi_p"][g]
    else:
        prompt = False

        # Determine delayed group and nuclide-dependent decay constant and spectrum
        for j in range(J):
            tot += nu_d[j]
            if xi < tot:
                # Delayed group determined, now determine nuclide
                N_nuclide = material["N_nuclide"]
                if N_nuclide == 1:
                    nuclide = mcdc["nuclides"][material["nuclide_IDs"][0]]
                    spectrum = nuclide["chi_d"][j]
                    decay = nuclide["decay"][j]
                    break
                SigmaF = get_MacroXS(XS_FISSION, material, P_arr, mcdc)
                xi = rng(P_new_arr) * nu_d[j] * SigmaF
                tot = 0.0
                for i in range(N_nuclide):
                    nuclide = mcdc["nuclides"][material["nuclide_IDs"][i]]
                    density = material["nuclide_densities"][i]
                    tot += density * nuclide["nu_d"][g, j] * nuclide["fission"][g]
                    if xi < tot:
                        # Nuclide determined, now get the constant and spectruum
                        spectrum = nuclide["chi_d"][j]
                        decay = nuclide["decay"][j]
                        break
                break

    # Sample outgoing energy
    xi = rng(P_new_arr)
    tot = 0.0
    for g_out in range(G):
        tot += spectrum[g_out]
        if tot > xi:
            break
    P_new["g"] = g_out

    # Sample emission time
    if not prompt:
        xi = rng(P_new_arr)
        P_new["t"] -= math.log(xi) / decay


@njit
def sample_phasespace_fission_nuclide(P_arr, nuclide, P_new_arr, mcdc):
    P_new = P_new_arr[0]
    P = P_arr[0]
    # Copy relevant attributes
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]

    # Sample isotropic direction
    P_new["ux"], P_new["uy"], P_new["uz"] = sample_isotropic_direction(P_new_arr)

    if mcdc["setting"]["mode_MG"]:
        fission_MG(P_arr, nuclide, P_new_arr)
    else:
        fission_CE(P_arr, nuclide, P_new_arr)


@njit
def fission_MG(P_arr, nuclide, P_new_arr):
    P_new = P_new_arr[0]
    P = P_arr[0]
    # Get constants
    G = nuclide["G"]
    J = nuclide["J"]
    g = P["g"]
    nu = nuclide["nu_f"][g]
    nu_p = nuclide["nu_p"][g]
    if J > 0:
        nu_d = nuclide["nu_d"][g]

    # Prompt or delayed?
    xi = rng(P_new_arr) * nu
    tot = nu_p
    if xi < tot:
        prompt = True
        spectrum = nuclide["chi_p"][g]
    else:
        prompt = False

        # Determine delayed group
        for j in range(J):
            tot += nu_d[j]
            if xi < tot:
                spectrum = nuclide["chi_d"][j]
                decay = nuclide["decay"][j]
                break

    # Sample outgoing energy
    xi = rng(P_new_arr)
    tot = 0.0
    for g_out in range(G):
        tot += spectrum[g_out]
        if tot > xi:
            break
    P_new["g"] = g_out

    # Sample emission time
    if not prompt:
        xi = rng(P_new_arr)
        P_new["t"] -= math.log(xi) / decay


@njit
def fission_CE(P_arr, nuclide, P_new_arr):
    P_new = P_new_arr[0]
    P = P_arr[0]
    # Get constants
    E = P["E"]
    J = 6
    nu = get_nu(NU_FISSION, nuclide, E)
    nu_p = get_nu(NU_FISSION_PROMPT, nuclide, E)
    nu_d = adapt.local_array(J, type_.float64)
    for j in range(J):
        nu_d[j] = get_nu_group(NU_FISSION_DELAYED, nuclide, E, j)

    # Delayed?
    prompt = True
    delayed_group = -1
    xi = rng(P_new_arr) * nu
    tot = nu_p
    if xi > tot:
        prompt = False

        # Determine delayed group
        for j in range(J):
            tot += nu_d[j]
            if xi < tot:
                delayed_group = j
                break

    # Sample outgoing energy
    if prompt:
        E_chi = nuclide["E_chi_p"]
        NE_chi = nuclide["NE_chi_p"]
        chi = nuclide["ce_chi_p"]
        P_new["E"] = sample_Eout(
            P_new_arr, nuclide["E_chi_p"], nuclide["NE_chi_p"], nuclide["ce_chi_p"]
        )
    else:
        if delayed_group == 0:
            P_new["E"] = sample_Eout(
                P_new_arr,
                nuclide["E_chi_d1"],
                nuclide["NE_chi_d1"],
                nuclide["ce_chi_d1"],
            )
        elif delayed_group == 1:
            P_new["E"] = sample_Eout(
                P_new_arr,
                nuclide["E_chi_d2"],
                nuclide["NE_chi_d2"],
                nuclide["ce_chi_d2"],
            )
        elif delayed_group == 2:
            P_new["E"] = sample_Eout(
                P_new_arr,
                nuclide["E_chi_d3"],
                nuclide["NE_chi_d3"],
                nuclide["ce_chi_d3"],
            )
        elif delayed_group == 3:
            P_new["E"] = sample_Eout(
                P_new_arr,
                nuclide["E_chi_d4"],
                nuclide["NE_chi_d4"],
                nuclide["ce_chi_d4"],
            )
        elif delayed_group == 4:
            P_new["E"] = sample_Eout(
                P_new_arr,
                nuclide["E_chi_d5"],
                nuclide["NE_chi_d5"],
                nuclide["ce_chi_d5"],
            )
        else:
            P_new["E"] = sample_Eout(
                P_new_arr,
                nuclide["E_chi_d6"],
                nuclide["NE_chi_d6"],
                nuclide["ce_chi_d6"],
            )

    # Sample emission time
    if not prompt:
        xi = rng(P_new_arr)
        P_new["t"] -= math.log(xi) / nuclide["ce_decay"][delayed_group]


# =============================================================================
# Branchless collision
# =============================================================================


@njit
def branchless_collision(P_arr, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    material = mcdc["materials"][P["material_ID"]]

    # Adjust weight
    SigmaT = get_MacroXS(XS_TOTAL, material, P_arr, mcdc)
    n_scatter = get_MacroXS(XS_NU_SCATTER, material, P_arr, mcdc)
    n_fission = get_MacroXS(XS_NU_FISSION, material, P_arr, mcdc) / mcdc["k_eff"]
    n_total = n_fission + n_scatter
    P["w"] *= n_total / SigmaT

    P_rec_arr = adapt.local_array(1, type_.particle_record)

    # Set spectrum and decay rate
    if rng(P_arr) < n_scatter / n_total:
        sample_phasespace_scattering(P_arr, material, P_arr, mcdc)
    else:
        if mcdc["setting"]["mode_MG"]:
            sample_phasespace_fission(P_arr, material, P_arr, mcdc)
        else:
            nuclide = sample_nuclide(material, P_arr, XS_NU_FISSION, mcdc)
            sample_phasespace_fission_nuclide(P_arr, nuclide, P_arr, mcdc)

            # Beyond time census or time boundary?
            idx_census = mcdc["idx_census"]
            if P["t"] > mcdc["setting"]["census_time"][idx_census]:
                P["alive"] = False
                split_as_record(P_rec_arr, P_arr)
                adapt.add_active(P_rec_arr, prog)
            elif P["t"] > mcdc["setting"]["time_boundary"]:
                P["alive"] = False


# =============================================================================
# Weight window
# =============================================================================


@njit
def weight_window(P_arr, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    # Get indices
    ix, iy, iz, it, outside = mesh_.structured.get_indices(
        P_arr, mcdc["technique"]["ww"]["mesh"]
    )

    # Target weight
    w_target = mcdc["technique"]["ww"]["center"][it, ix, iy, iz]

    # Population control factor
    w_target *= mcdc["technique"]["pc_factor"]

    # Surviving probability
    p = P["w"] / w_target

    # Window width
    width = mcdc["technique"]["ww"]["width"]

    P_new_arr = adapt.local_array(1, type_.particle_record)

    # If above target
    if p > width:
        # Set target weight
        P["w"] = w_target

        # Splitting (keep the original particle)
        n_split = math.floor(p)
        for i in range(n_split - 1):
            split_as_record(P_new_arr, P_arr)
            adapt.add_active(P_new_arr, prog)

        # Russian roulette
        p -= n_split
        xi = rng(P_arr)
        if xi <= p:
            split_as_record(P_new_arr, P_arr)
            adapt.add_active(P_new_arr, prog)

    # Below target
    elif p < 1.0 / width:
        # Russian roulette
        xi = rng(P_arr)
        if xi > p:
            P["alive"] = False
        else:
            P["w"] = w_target


@njit
def update_weight_windows(data, mcdc):
    idx_batch = mcdc["idx_batch"]
    idx_census = mcdc["idx_census"]
    epsilon = mcdc["technique"]["ww"]["epsilon"]
    # accessing most recent tally dump
    with objmode():
        f = h5py.File(
            mcdc["setting"]["output_name"]
            + "-batch_%i-census_%i.h5" % (idx_batch, idx_census),
            "r",
        )
        tallies = f["tallies/mesh_tally_" + str(mcdc["technique"]["ww"]["tally_idx"])]
        if mcdc["setting"]["census_tally_frequency"] > 1:
            old_flux = tallies["flux"]["score"][-1]
        else:
            old_flux = tallies["flux"]["score"]
        Nx = mcdc["technique"]["ww"]["mesh"]["Nx"]
        Ny = mcdc["technique"]["ww"]["mesh"]["Ny"]
        Nz = mcdc["technique"]["ww"]["mesh"]["Nz"]
        Nt = mcdc["technique"]["ww"]["mesh"]["Nt"]

        ax_expand = []
        if Nx == 1:
            ax_expand.append(0)
        if Ny == 1:
            ax_expand.append(1)
        if Nz == 1:
            ax_expand.append(2)
        for ax in ax_expand:
            old_flux = np.expand_dims(old_flux, axis=ax)
        center = old_flux

        if epsilon[WW_WOLLABER] > 0:
            w_min = epsilon[WW_WOLLABER + 1]
            center = (center) * (
                1
                + (1 / epsilon[WW_WOLLABER] - 1)
                * np.exp(-(center - w_min) / epsilon[WW_WOLLABER])
            )
        if epsilon[WW_MIN] > 0:
            center = center * (1 - epsilon[WW_MIN]) + epsilon[WW_MIN]
            center[center <= 0] = epsilon[WW_MIN]
        center /= np.max(center)
        mcdc["technique"]["ww"]["center"][idx_census + 1] = center


# =============================================================================
# Weight Roulette
# =============================================================================


@njit
def weight_roulette(P_arr, mcdc):
    P = P_arr[0]
    w_survive = mcdc["technique"]["wr_survive"]
    prob_survive = P["w"] / w_survive
    if rng(P_arr) <= prob_survive:
        P["w"] = w_survive
        if mcdc["technique"]["iQMC"]:
            P["iqmc"]["w"][:] = w_survive
    else:
        P["alive"] = False


# =============================================================================
# Continuous Energy Physics
# =============================================================================


@njit
def get_macro_xs(reaction_type, material, P_arr, mcdc):
    P = P_arr[0]
    
    # Continuous-energy XS
    if mcdc["setting"]["mode_CE"]:
        macro_xs = 0.0
        E = P["E"]

        # Sum over all nuclides
        for nuclide in material.nuclides:
            atomic_density = material.nuclide_composition[nuclide]
            micro_xs = get_micro_xs(reaction_type, nuclide, E)

            # Accumulate
            macro_xs += atomic_density * micro_xs

        return macro_xs

    # Multigroup XS
    g = P["g"]

    # Cross sections
    if type_ == XS_TOTAL:
        return material["total"][g]
    elif type_ == XS_SCATTER:
        return material["scatter"][g]
    elif type_ == XS_CAPTURE:
        return material["capture"][g]
    elif type_ == XS_FISSION:
        return material["fission"][g]

    # Productions
    elif type_ == XS_NU_SCATTER:
        nu = material["nu_s"][g]
        scatter = material["scatter"][g]
        return nu * scatter
    elif type_ == XS_NU_FISSION:
        nu = material["nu_f"][g]
        fission = material["fission"][g]
        return nu * fission
    elif type_ == XS_NU_FISSION_PROMPT:
        nu_p = material["nu_p"][g]
        fission = material["fission"][g]
        return nu_p * fission
    elif type_ == XS_NU_FISSION_DELAYED:
        nu_d = 0.0
        for j in range(material["J"]):
            nu_d += material["nu_d"][g, j]
        fission = material["fission"][g]
        return nu_d * fission


@njit
def get_micro_xs(reaction_type, nuclide, E):
    if reaction_type == REACTION_TOTAL:
        xs_data = nuclide.total_xs
    else:
        found = False
        for reaction in nuclide.reactions:
            if reaction.type_enum == reaction_type:
                xs_data = reaction.xs
                found = True
                break
        if not found:
            return 0.0
    return evaluate_from_table(E, nuclide.xs_energy_grid, xs_data)

    """
    # Binary Multiplicities
    elif type_ == XS_NU_SCATTER:
        data = nuclide["ce_scatter"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = 1.0
        return nu * xs
    elif type_ == XS_NU_FISSION:
        if not nuclide["fissionable"]:
            return 0.0
        data = nuclide["ce_fission"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = get_nu(NU_FISSION, nuclide, E)
        return nu * xs
    elif type_ == XS_NU_FISSION_PROMPT:
        if not nuclide["fissionable"]:
            return 0.0
        data = nuclide["ce_fission"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = get_nu(NU_FISSION_PROMPT, nuclide, E)
        return nu * xs
    elif type_ == XS_NU_FISSION_DELAYED:
        if not nuclide["fissionable"]:
            return 0.0
        data = nuclide["ce_fission"]
        xs = get_XS(data, E, nuclide["E_xs"], nuclide["NE_xs"])
        nu = get_nu(NU_FISSION_DELAYED, nuclide, E)
        return nu * xs
    """


@njit
def get_XS(data, E, E_grid, NE):
    # Search XS energy bin index
    idx = binary_search_with_length(E, E_grid, NE)

    # Extrapolate if E is outside the given data
    if idx == -1:
        idx = 0
    elif idx + 1 == NE:
        idx -= 1

    # Linear interpolation
    E1 = E_grid[idx]
    E2 = E_grid[idx + 1]
    XS1 = data[idx]
    XS2 = data[idx + 1]

    return XS1 + (E - E1) * (XS2 - XS1) / (E2 - E1)


@njit
def get_nu_group(type_, nuclide, E, group):
    if type_ == NU_FISSION:
        nu = get_XS(nuclide["ce_nu_p"], E, nuclide["E_nu_p"], nuclide["NE_nu_p"])
        for i in range(6):
            nu += get_XS(
                nuclide["ce_nu_d"][i], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
            )
        return nu

    if type_ == NU_FISSION_PROMPT:
        return get_XS(nuclide["ce_nu_p"], E, nuclide["E_nu_p"], nuclide["NE_nu_p"])

    if type_ == NU_FISSION_DELAYED and group == -1:
        tot = 0.0
        for i in range(6):
            tot += get_XS(
                nuclide["ce_nu_d"][i], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
            )
        return tot

    if type_ == NU_FISSION_DELAYED and group != -1:
        return get_XS(
            nuclide["ce_nu_d"][group], E, nuclide["E_nu_d"], nuclide["NE_nu_d"]
        )


@njit
def get_nu(type_, nuclide, E):
    return get_nu_group(type_, nuclide, E, -1)


@njit
def sample_nuclide(material, P_arr, reaction_type, mcdc):
    P = P_arr[0]
    xi = rng(P_arr) * get_macro_xs(reaction_type, material, P_arr, mcdc)
    total = 0.0
    for nuclide in material.nuclides:
        atomic_density = material.nuclide_composition[nuclide]
        total += atomic_density * get_micro_xs(reaction_type, nuclide, P["E"])
        if total > xi:
            break
    return nuclide


@njit
def sample_Eout(P_new_arr, E_grid, NE, chi):
    P_new = P_new_arr[0]
    xi = rng(P_new_arr)

    # Determine bin index
    idx = binary_search_with_length(xi, chi, NE)

    # Linear interpolation
    E1 = E_grid[idx]
    E2 = E_grid[idx + 1]
    chi1 = chi[idx]
    chi2 = chi[idx + 1]
    return E1 + (xi - chi1) * (E2 - E1) / (chi2 - chi1)


# =============================================================================
# Miscellany
# =============================================================================


@njit
def lartg(f, g):
    """
    Originally a Lapack routine to generate a plane rotation with
    real cosine and real sine.

    Reference
    ----------
    https://netlib.org/lapack/explore-html/df/dd1/group___o_t_h_e_rauxiliary_ga86f8f877eaea0386cdc2c3c175d9ea88.html#:~:text=DLARTG%20generates%20a%20plane%20rotation%20with%20real%20cosine,%3D%20G%20%2F%20R%20Hence%20C%20%3E%3D%200.

    Parameters
    ----------
    f :  The first component of vector to be rotated.
    g :  The second component of vector to be rotated.

    Returns
    -------
    c : The cosine of the rotation.
    s : The sine of the rotation.
    r : The nonzero component of the rotated vector.

    """
    r = np.sign(f) * np.sqrt(f * f + g * g)
    c = f / r
    s = g / r
    return c, s, r


@njit
def modified_gram_schmidt(V, u):
    """
    Modified Gram Schmidt routine

    """
    u = np.reshape(u, (u.size, 1))
    V = np.ascontiguousarray(V)
    w1 = u - np.dot(V, np.dot(V.T, u))
    v1 = w1 / np.linalg.norm(w1)
    w2 = v1 - np.dot(V, np.dot(V.T, v1))
    v2 = w2 / np.linalg.norm(w2)
    V = np.append(V, v2, axis=1)
    return V


# =============================================================================
# Variance Deconvolution
# =============================================================================


@njit
def uq_resample(mean, delta, info):
    # Currently only uniform distribution
    shape = mean.shape
    size = mean.size
    xi = rng_array(info["rng_seed"], shape, size)

    return mean + (2 * xi - 1) * delta


@njit
def reset_material(mcdc, idm, material_uq):
    # Assumes all nuclides have already been re-sampled
    # Basic XS
    material = mcdc["materials"][idm]
    for tag in literal_unroll(("capture", "scatter", "fission", "total")):
        if material_uq["flags"][tag]:
            material[tag][:] = 0.0
            for n in range(material["N_nuclide"]):
                nuc1 = mcdc["nuclides"][material["nuclide_IDs"][n]]
                density = material["nuclide_densities"][n]
                material[tag] += nuc1[tag] * density

    # Effective speed
    if material_uq["flags"]["speed"]:
        material["speed"][:] = 0.0
        for n in range(material["N_nuclide"]):
            nuc2 = mcdc["nuclides"][material["nuclide_IDs"][n]]
            density = material["nuclide_densities"][n]
            material["speed"] += nuc2["speed"] * nuc2["total"] * density
        if max(material["total"]) == 0.0:
            material["speed"][:] = nuc2["speed"][:]
        else:
            material["speed"] /= material["total"]

    # Calculate effective spectra and multiplicities of scattering and prompt fission
    G = material["G"]
    if max(material["scatter"]) > 0.0:
        shape = material["chi_s"].shape
        nuSigmaS = np.zeros(shape)
        for i in range(material["N_nuclide"]):
            nuc3 = mcdc["nuclides"][material["nuclide_IDs"][i]]
            density = material["nuclide_densities"][i]
            SigmaS = np.diag(nuc3["scatter"]) * density
            nu_s = np.diag(nuc3["nu_s"])
            chi_s = np.ascontiguousarray(nuc3["chi_s"].transpose())
            nuSigmaS += chi_s.dot(nu_s.dot(SigmaS))
        chi_nu_s = nuSigmaS.dot(np.diag(1.0 / material["scatter"]))
        material["nu_s"] = np.sum(chi_nu_s, axis=0)
        material["chi_s"] = np.ascontiguousarray(
            chi_nu_s.dot(np.diag(1.0 / material["nu_s"])).transpose()
        )
    if max(material["fission"]) > 0.0:
        nuSigmaF = np.zeros((G, G), dtype=float)
        for n in range(material["N_nuclide"]):
            nuc4 = mcdc["nuclides"][material["nuclide_IDs"][n]]
            density = material["nuclide_densities"][n]
            SigmaF = np.diag(nuc4["fission"]) * density
            nu_p = np.diag(nuc4["nu_p"])
            chi_p = np.ascontiguousarray(np.transpose(nuc4["chi_p"]))
            nuSigmaF += chi_p.dot(nu_p.dot(SigmaF))
        chi_nu_p = nuSigmaF.dot(np.diag(1.0 / material["fission"]))
        material["nu_p"] = np.sum(chi_nu_p, axis=0)
        # Required because the below function otherwise returns an F-contiguous array
        material["chi_p"] = np.ascontiguousarray(
            np.transpose(chi_nu_p.dot(np.diag(1.0 / material["nu_p"])))
        )

    # Calculate delayed and total fission multiplicities
    if max(material["fission"]) > 0.0:
        material["nu_f"][:] = material["nu_p"][:]
        for j in range(material["J"]):
            total = np.zeros(material["G"])
            for n in range(material["N_nuclide"]):
                nuc5 = mcdc["nuclides"][material["nuclide_IDs"][n]]
                density = material["nuclide_densities"][n]
                total += nuc5["nu_d"][:, j] * nuc5["fission"] * density
            material["nu_d"][:, j] = total / material["fission"]
            material["nu_f"] += material["nu_d"][:, j]


@njit
def reset_nuclide(nuclide, nuclide_uq):
    for name in literal_unroll(
        ("speed", "decay", "capture", "fission", "nu_s", "nu_p")
    ):
        if nuclide_uq["flags"][name]:
            nuclide[name] = uq_resample(
                nuclide_uq["mean"][name], nuclide_uq["delta"][name], nuclide_uq["info"]
            )

    if nuclide_uq["flags"]["scatter"]:
        scatter = uq_resample(
            nuclide_uq["mean"]["scatter"],
            nuclide_uq["delta"]["scatter"],
            nuclide_uq["info"],
        )
        nuclide["scatter"] = np.sum(scatter, 0)
        nuclide["chi_s"][:, :] = np.swapaxes(scatter, 0, 1)[:, :]
        for g in range(nuclide["G"]):
            if nuclide["scatter"][g] > 0.0:
                nuclide["chi_s"][g, :] /= nuclide["scatter"][g]

    if nuclide_uq["flags"]["total"]:
        nuclide["total"][:] = (
            nuclide["capture"] + nuclide["scatter"] + nuclide["fission"]
        )

    if nuclide_uq["flags"]["nu_d"]:
        nu_d = uq_resample(
            nuclide_uq["mean"]["nu_d"], nuclide_uq["delta"]["nu_d"], nuclide_uq["info"]
        )
        nuclide["nu_d"][:, :] = np.swapaxes(nu_d, 0, 1)[:, :]

    if nuclide_uq["flags"]["nu_f"]:  # True if either nu_p or nu_d is true
        nuclide["nu_f"] = nuclide["nu_p"]
        for j in range(nuclide["J"]):
            nuclide["nu_f"] += nuclide["nu_d"][:, j]

    # Prompt fission spectrum (If G == 1, all ones)
    if nuclide_uq["flags"]["chi_p"]:
        chi_p = uq_resample(
            nuclide_uq["mean"]["chi_p"],
            nuclide_uq["delta"]["chi_p"],
            nuclide_uq["info"],
        )
        nuclide["chi_p"][:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
        # Normalize
        for g in range(nuclide["G"]):
            if np.sum(nuclide["chi_p"][g, :]) > 0.0:
                nuclide["chi_p"][g, :] /= np.sum(nuclide["chi_p"][g, :])

    # Delayed fission spectrum (matrix of size JxG)
    if nuclide_uq["flags"]["chi_d"]:
        chi_d = uq_resample(
            nuclide_uq["mean"]["chi_d"],
            nuclide_uq["delta"]["chi_d"],
            nuclide_uq["info"],
        )
        # Transpose: [gout, dg] -> [dg, gout]
        nuclide["chi_d"][:, :] = np.swapaxes(chi_d, 0, 1)[:, :]
        # Normalize
        for dg in range(nuclide["J"]):
            if np.sum(nuclide["chi_d"][dg, :]) > 0.0:
                nuclide["chi_d"][dg, :] /= np.sum(nuclide["chi_d"][dg, :])


@njit
def uq_reset(mcdc, seed):
    # Types of uq parameters: materials, nuclides
    N = len(mcdc["technique"]["uq_"]["nuclides"])
    for i in range(N):
        mcdc["technique"]["uq_"]["nuclides"][i]["info"]["rng_seed"] = split_seed(
            i, seed
        )
        idn = mcdc["technique"]["uq_"]["nuclides"][i]["info"]["ID"]
        reset_nuclide(mcdc["nuclides"][idn], mcdc["technique"]["uq_"]["nuclides"][i])

    M = len(mcdc["technique"]["uq_"]["materials"])
    for i in range(M):
        mcdc["technique"]["uq_"]["materials"][i]["info"]["rng_seed"] = split_seed(
            i, seed
        )
        idm = mcdc["technique"]["uq_"]["materials"][i]["info"]["ID"]
        reset_material(mcdc, idm, mcdc["technique"]["uq_"]["materials"][i])


@njit
def uq_tally_closeout_history(data_tally, mcdc):
    # Assumes N_batch > 1
    # Accumulate square of history score, but continue to accumulate bin
    history_bin = data_tally[TALLY_SCORE] - data_tally[TALLY_UQ_BATCH]
    data_tally[TALLY_UQ_BATCH_VAR] += history_bin**2
    data_tally[TALLY_UQ_BATCH] = data_tally[TALLY_SCORE]


@njit
def uq_tally_closeout_batch(data_tally, mcdc):
    # Reset bin
    N_bin = data_tally.shape[1]
    for i in range(N_bin):
        # Reset score bin
        data_tally[TALLY_UQ_BATCH, i] = 0.0

    # MPI Reduce
    buff = np.zeros(N_bin)
    with objmode():
        MPI.COMM_WORLD.Reduce(
            np.array(data_tally[TALLY_UQ_BATCH_VAR]), buff, MPI.SUM, 0
        )
    data_tally[TALLY_UQ_BATCH_VAR][:] = buff


@njit
def uq_tally_closeout(data_tally, mcdc):
    N_history = mcdc["setting"]["N_particle"]

    data_tally[TALLY_UQ_BATCH_VAR] = (
        data_tally[TALLY_UQ_BATCH_VAR] / N_history - data_tally[TALLY_SUM_SQ]
    ) / (N_history - 1)

    # If we're here, N_batch > 1
    N_history = mcdc["setting"]["N_batch"]

    # Store results
    mean = data_tally[TALLY_SUM] / N_history
    data_tally[TALLY_UQ_BATCH_VAR] /= N_history
    data_tally[TALLY_UQ_BATCH] = (
        data_tally[TALLY_SUM_SQ] - N_history * np.square(mean)
    ) / (N_history - 1)
