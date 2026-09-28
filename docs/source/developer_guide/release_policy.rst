.. _release_policy:

==========================
Release Policy and Process
==========================

MC/DC uses `Semantic Versioning <https://semver.org/>`_ as a guide for distinguishing minor and patch releases and maintains a human-readable release history in `CHANGELOG.md <https://github.com/mcdc-project/mcdc/blob/dev/CHANGELOG.md>`_.

Minor Releases
--------------

For MC/DC, a minor release primarily expands what the code can do.
Examples include an additional transport or execution capability, a new public Python API, a new command-line option, a new opt-in behavior, or a deprecation that users need time to accommodate.
A minor release may include compatible fixes and internal improvements alongside its features.

MC/DC plans one minor release in each three-month seasonal cycle.
These releases collect compatible features, improvements, and fixes that have passed the project's required review and validation.
For planning convenience, these cycles follow the Northern Hemisphere meteorological seasons: winter (Dec-Feb), spring (Mar-May), summer (Jun-Aug), and autumn/fall (Sep-Nov).

The seasonal schedule is a target rather than a reason to release unverified work.
A minor release may be delayed when additional testing, documentation, or integration work is needed; it may also be brought forward when the accumulated changes are substantial and users would benefit from earlier availability.

Patch Releases
--------------

For MC/DC, a patch release primarily restores or strengthens behavior already expected from the current stable release.
A change belongs in a patch when it corrects intended or documented behavior, prevents an installation or runtime failure, or fixes an implementation defect without asking users to adopt a new interface.
Tests, documentation corrections, refactoring, build changes, and developer tooling may accompany a patch when they support or validate the fix and do not independently expand user-facing behavior.
If a change introduces a new option, API, capability, or opt-in behavior, it should normally be held for a minor release even when developed alongside a bug fix and separated from the patch branch.
When the classification is not obvious, use the purpose and user impact of the release: capability expansion points to a minor release, while correction of existing behavior points to a patch release.

Bug fixes are not held until the next seasonal minor release.
Once a fix has passed review and the relevant validation and release checks, MC/DC publishes a patch release as soon as practical.
Every patch release must include a non-empty ``Fixed`` section in ``CHANGELOG.md`` that describes the user-visible defect corrected by the release.
Place ``Fixed`` first in a patch release entry so the reason for the release is immediately visible.
Supporting changes may also appear under other headings, but the defect that justifies the patch release must be stated under ``Fixed``.
For dependency-compatibility patches, identify the installation or runtime failure prevented and the affected dependency or version range when known.

Published versions and release notes are available from the `MC/DC releases page <https://github.com/mcdc-project/mcdc/releases>`_.

.. _release_branch_routes:

Release Branch Routes
---------------------

MC/DC uses different integration routes for feature and patch releases because ``upstream/dev`` and ``upstream/main`` have different roles.
The development branch integrates work for the next minor release, while the main branch identifies the current stable release line.

.. list-table:: Release workflow comparison
   :header-rows: 1
   :widths: 24 24 28 44

   * - Workflow
     - Release branch base
     - Route to release
     - Purpose
   * - Feature (minor-release) workflow
     - ``upstream/dev``
     - Fork feature branch → ``upstream/dev`` → ``upstream/main``
     - Includes the compatible features, improvements, and fixes accumulated for the next minor version.
   * - Patch workflow
     - ``upstream/main``
     - Fork patch branch → ``upstream/patch/X.Y.Z`` → ``upstream/main``
     - Releases selected fixes from the current stable line without including unreleased features already present on ``upstream/dev``.

The upstream ``patch/X.Y.Z`` branch is temporary and specific to the intended patch version ``X.Y.Z``.
See the :ref:`release_checklist` for branch preparation, validation, publication, and the return to development.

.. _release_checklist:

Release Checklist
-----------------

Use this checklist for every minor and patch release.
Here, ``upstream`` refers to the canonical ``mcdc-project/mcdc`` repository and ``release_branch`` refers to the contributor's release-preparation branch in their fork for either workflow.
Upstream integration branches are named explicitly as ``upstream/dev``, ``upstream/patch/X.Y.Z``, or ``upstream/main``.

Prepare the Release
^^^^^^^^^^^^^^^^^^^

#. Confirm the intended version and scope against the release policy above.
#. Select the appropriate :ref:`release_branch_routes`.
   Create ``release_branch`` in your fork from ``upstream/dev`` for a minor release or the current stable commit on ``upstream/main`` for a patch release.
   For a patch release, check whether ``patch/X.Y.Z`` already exists in ``upstream``.
   If it does not exist, create it from the current stable commit on ``upstream/main``; otherwise, use the existing branch.
   Prepare the selected fixes and necessary supporting changes on your fork's ``release_branch`` and target ``upstream/patch/X.Y.Z`` when opening the patch pull request.
   If a fix was first developed on ``upstream/dev``, transfer only the fix and its necessary tests, documentation, and supporting changes onto ``release_branch``.
   Do not merge ``upstream/dev`` into either patch branch.
#. Review the ``Unreleased`` section of ``CHANGELOG.md``.
   Ensure every user-visible change is included under the correct heading, remove empty headings, and add contributor attribution where appropriate.
   For a patch release, place ``Fixed`` first and confirm that it is non-empty and clearly states the defect that justifies the release.
#. Finalize the release version and date in ``CHANGELOG.md`` and the ``version`` and ``date-released`` fields of ``CITATION.cff``.
#. Update ``docs/source/_static/switcher.json``:
   - Change the stable entry's display name to the full ``X.Y.Z (stable)`` version while retaining ``stable`` as its version identifier and URL.
   - Add the previous stable release as a historical entry, using its Read the Docs tag identifier and URL, and retain the existing historical entries in newest-to-oldest order.
   - Confirm that every listed historical version is active, built, and reachable on Read the Docs.
#. Review and update ``CITATION.cff``; it supplies GitHub citation guidance and Zenodo metadata.
   Validate the CFF file from the repository root with ``cffconvert --validate``.
#. Review the release diff for user-facing behavior.
   Confirm that each affected interface or workflow is reflected in the relevant documentation and examples.
   If existing users must change how they use MC/DC, include the necessary deprecation notice or migration guidance.
#. If ``release_branch`` requires a newer Python version, add that version to the ``Python Compatibility Tests`` matrix in ``.github/workflows/python_compatibility.yml`` so it is tested during integration.
#. Confirm the required local checks, continuous-integration workflows, and distribution artifact tests pass on the release candidate; resolve any dependency incompatibilities they expose and update ``pyproject.toml`` and ``CHANGELOG.md`` as needed.
#. Run the applicable `MC/DC-VVP campaign <https://github.com/mcdc-project/mcdc-vvp>`_ with the release candidate and the corresponding MC/DC-VVP version.
   Process the completed suites and review their convergence, reference, and comparison results for unexpected behavior.
#. Run ``python prepare_release.py`` in MC/DC-VVP to collect the processed PNG figures and GIF animations into its flat ``release/`` asset directory.
   Confirm that the prepared assets cover the documented VVP cases and that their names match the links used by the MC/DC documentation.

Integrate the Release
^^^^^^^^^^^^^^^^^^^^^

Follow these steps for both minor and patch releases:

#. Merge the completed ``release_branch`` through a reviewed pull request with passing checks into ``upstream/dev`` for a minor release or ``upstream/patch/X.Y.Z`` for a patch release.
#. Manually run the ``Python Compatibility Tests`` workflow in the upstream repository (``dev`` or ``patch/X.Y.Z``).
   Confirm that the unit and Python/Numba regression tests pass for every Python version in the workflow matrix on the final release candidate.
   Resolve any failures and rerun the affected checks if the candidate changes.
#. Open a pull request from the validated upstream branch (``dev`` or ``patch/X.Y.Z``) to ``upstream/main`` and merge it after review and passing checks.
#. Use the resulting ``upstream/main`` commit as the release base.

Publish from Main
^^^^^^^^^^^^^^^^^

#. Create the ``v``-prefixed tag and GitHub Release from the validated release commit using the following settings:

   * **Target:** select ``main``.
   * **Release title:** use the tag exactly, including the ``v`` prefix.
   * **Previous tag:** select the previous published version, then click **Generate release notes**.
   * **Release notes:** place a brief release summary first, followed by a ``## Changelog`` section containing the associated entry from ``CHANGELOG.md``, then the generated release notes.
   * **Release label:** select **Latest**.
   * **Finish:** click **Publish release** when creating the release, or **Update release** when editing an existing release.

#. Replace the assets attached to the mutable `VVP results release <https://github.com/mcdc-project/mcdc/releases/tag/vvp-results>`_ with the contents of the prepared MC/DC-VVP ``release/`` directory.
   Update its release notes to identify the MC/DC and MC/DC-VVP versions used for the published campaign.
#. Confirm that every VVP figure and animation referenced by the documentation is available from the ``vvp-results`` release and renders on its case page.
#. Confirm that the automatically triggered `Publish Python Package to PyPI <https://github.com/mcdc-project/mcdc/actions/workflows/publish-pypi.yml>`_ and `Check citation metadata <https://github.com/mcdc-project/mcdc/actions/workflows/check_citation.yml>`_ workflows complete successfully, and that the release is available from the `stable Read the Docs site <https://mcdc.readthedocs.io/en/stable/>`_.
#. Verify the Zenodo archive's version, date, authors, license, and DOI.
   Submit the record to the CARRE community (``carre``) through Zenodo and confirm inclusion; CFF does not configure community submission.
#. Smoke-test the published PyPI package in a clean environment with ``python -m pip install "mcdc==X.Y.Z"``, then run a minimal MC/DC simulation.

Return to Development
^^^^^^^^^^^^^^^^^^^^^

#. Merge ``upstream/main`` back into ``upstream/dev`` after the release is published and verified.
#. For a patch release, confirm that the back-merge retains the patch while preserving the unreleased feature work already on ``upstream/dev``.
#. Prepare ``upstream/dev`` for the next development cycle and confirm its required checks pass.
#. Remove the merged ``release_branch`` from your fork when it is no longer needed.
#. For a patch release, also delete ``upstream/patch/X.Y.Z`` after publication, verification, and the back-merge into ``dev`` are complete.
