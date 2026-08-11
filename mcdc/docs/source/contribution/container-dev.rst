MC/DC Container Build & Development
===================================

Building from Source
--------------------

Run builds from the **root directory** of the MC/DC repository.

.. code-block:: bash

    cd /path/to/MCDC
    ls containers/Dockerfile
    ls pyproject.toml

Docker
~~~~~~

.. code-block:: bash

    docker build -f containers/Dockerfile -t mcdc:dev .

Docker Compose
~~~~~~~~~~~~~~

.. code-block:: bash

    docker compose -f containers/docker-compose.yml build

Podman
~~~~~~

.. code-block:: bash

    podman build -f containers/Dockerfile -t mcdc:dev .

Apptainer
~~~~~~~~~

Option A:

.. code-block:: bash

    apptainer build --sandbox mcdc_sandbox docker://ghcr.io/cement-psaap/mcdc:dev

Option B:

.. code-block:: bash

    docker build -f containers/Dockerfile -t mcdc:dev .
    docker save mcdc:dev -o mcdc.tar
    scp mcdc.tar user@host:~/
    apptainer build --sandbox mcdc_sandbox docker-archive://mcdc.tar

LLNL Storage Setup
------------------

If you see:

::

    lsetxattr: operation not supported

Option A:

.. code-block:: bash

    podman --root /var/tmp/$USER/containers/storage run --rm -it mcdc:dev

Option B:

.. code-block:: bash

    mkdir -p ~/.config/containers
    # create storage.conf with overlay config

Troubleshooting
---------------

``lsetxattr`` error
~~~~~~~~~~~~~~~~~~~
Cause: Podman storage on network filesystem.

``setgroups 65534 failed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cause: Rootless Podman user mapping.

``permission denied``
~~~~~~~~~~~~~~~~~~~~~
Fix:

.. code-block:: bash

    podman run --rm -it --user root mcdc:dev

``Out of memory`` (Apptainer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use sandbox mode.

``HYDU_create_process`` error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use:

.. code-block:: bash

    mpirun -launcher fork -n 4 python input.py

For Developers
--------------

Pushing to the Registry
~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: One-Time Setup

1. Go to https://github.com/settings/tokens
2. Generate token with ``write:packages``
3. Login:

.. code-block:: bash

    echo "TOKEN" | docker login ghcr.io -u USER --password-stdin

.. rubric:: Building and Pushing

On Apple Silicon:

.. code-block:: bash

    docker build --platform linux/amd64 -f containers/Dockerfile -t mcdc:dev-amd64 .
    docker tag mcdc:dev-amd64 ghcr.io/cement-psaap/mcdc:dev
    docker push ghcr.io/cement-psaap/mcdc:dev

.. rubric:: Making the Package Public

1. Go to org packages page
2. Find ``mcdc``
3. Change visibility to Public

File Overview
~~~~~~~~~~~~~

::

    containers/
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md
