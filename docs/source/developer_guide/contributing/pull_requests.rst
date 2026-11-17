.. _pull_requests:

=============
Pull Requests
=============

MC/DC uses a fork-based contribution workflow.
Open pull requests from your fork against the ``dev`` branch of ``mcdc-project/mcdc``.

Before opening a pull request, make sure:

- An issue describes the problem and its completion criteria.
  Open one first if none exists.
- The applicable tests pass.
- The code follows the project style.
- Tests and documentation have been added or updated as needed.
- ``CHANGELOG.md`` has been updated when the change is notable.

Changelog Updates
-----------------

Add a concise entry under ``[Unreleased]`` in `CHANGELOG.md <https://github.com/mcdc-project/mcdc/blob/dev/CHANGELOG.md>`_ when a pull request introduces a notable user- or developer-visible change.
Follow the format described in that file.
Release headings, versions, and dates are assigned during release preparation.

Pull Request Description
------------------------

Issues carry the problem, completion criteria, planning, and Project tracking.
Pull requests describe the implementation and how it was verified.

Use the pull request template to provide:

- A summary of what changed and why, including behavior before and after for user-facing changes.
- The required linked issue: use ``Closes #N`` if the pull request fully resolves it, or ``Refs #N`` if it addresses only part of it.
  Link related pull requests if applicable.
- How the change was verified and the results, or why testing was incomplete or not performed.
- Review notes highlighting known limitations, compatibility concerns, or anything requiring particular attention.
  Link issues for follow-up work if applicable.

The description should give reviewers enough context to understand the change, verify its scope, and identify any follow-up work.
