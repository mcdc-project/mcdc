MC/DC Container Guide
=====================

What Are Containers?
--------------------

A container is a lightweight, portable package that bundles an application
together with everything it needs to run: code, libraries, system tools,
and settings. Think of it like a shipping container — no matter what ship
(computer) carries it, the contents inside stay the same.

**Why does this matter for MC/DC?**

Installing MC/DC requires Python, MPI, Numba, and many other dependencies.
Getting all of these working together — especially on HPC systems where you
don't have admin access — can be painful. A container solves this by giving
you a pre-built environment where everything is already installed and tested.

Tested Platforms
----------------

+-------------+------------+--------+--------------------+--------+
| System      | OS         | Arch   | Container Tool     | Status |
+=============+============+========+====================+========+
| MacBook Pro | macOS 26.3 | arm64  | Docker 29.2.0      | ✓      |
+-------------+------------+--------+--------------------+--------+
| Tuolumne    | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✓      |
+-------------+------------+--------+--------------------+--------+
| Dane        | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✓      |
+-------------+------------+--------+--------------------+--------+
| Tioga       | RHEL 8.10  | x86_64 | Podman 4.9.4       | ✓      |
+-------------+------------+--------+--------------------+--------+
| COE (OSU)   | Rocky 8.10 | x86_64 | Apptainer 1.4.5    | ✓      |
+-------------+------------+--------+--------------------+--------+

All platforms produce identical containers: Debian 13, Python 3.11,
MPICH 4.2.1, MC/DC 0.12.0.

Getting Started (New Users)
---------------------------

This section is for anyone who just wants to **run MC/DC** in a container.
No prior container experience needed.

Step 1: Pull the Pre-Built Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You don't need to build anything. A ready-to-use image is available on
the GitHub Container Registry.

.. rubric:: Local Machine (Docker)

First, install Docker Desktop if you haven't already. Then open a terminal
and run:

.. code-block:: bash

    docker pull ghcr.io/cement-psaap/mcdc:dev
    docker run --rm -it ghcr.io/cement-psaap/mcdc:dev

You are now inside the container. Try:

.. code-block:: bash

    python -c "import mcdc; print('MC/DC OK')"

Type ``exit`` to leave the container.

.. rubric:: LLNL Systems — Tuolumne, Tioga, Dane (Podman)

Podman is already installed on LLNL systems. It works just like Docker.

.. code-block:: bash

    podman pull ghcr.io/cement-psaap/mcdc:dev
    podman run --rm -it ghcr.io/cement-psaap/mcdc:dev

.. note::

    If you see ``lsetxattr: operation not supported``,
    see *LLNL Storage Setup* in Part 2.

.. rubric:: OSU Systems — COE (Apptainer)

Apptainer is already installed on COE.

.. code-block:: bash

    apptainer build --sandbox mcdc_sandbox docker://ghcr.io/cement-psaap/mcdc:dev
    apptainer exec mcdc_sandbox python -c "import mcdc; print('MC/DC OK')"

.. note::

    If ``apptainer pull`` fails with "Out of memory", use ``--sandbox``.

Step 2: Run Your Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Docker / Podman

.. code-block:: bash

    docker run --rm -v $(pwd):/work -w /work mcdc:dev python input.py
    docker run --rm mcdc:dev mpirun -n 4 python input.py

For Podman, replace ``docker`` with ``podman``.

**Flags explanation**

- ``--rm``: Automatically clean up container.
- ``-it``: Interactive terminal.
- ``-v $(pwd):/work``: Share current folder.
- ``-w /work``: Start inside shared folder.

.. rubric:: Apptainer (OSU)

.. code-block:: bash

    apptainer exec mcdc_sandbox python input.py
    apptainer exec mcdc_sandbox mpirun -launcher fork -n 4 python input.py

.. note::

    Apptainer automatically shares your home directory.

Step 3: Docker Compose (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From MC/DC repo root:

.. code-block:: bash

    docker compose -f containers/docker-compose.yml run --rm dev bash
    docker compose -f containers/docker-compose.yml run --rm test
    docker compose -f containers/docker-compose.yml run --rm mpi mpirun -n 4 python input.py
