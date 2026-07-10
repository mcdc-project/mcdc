.. _documentation_sphinx:

.. highlight:: none

Sphinx and Read the Docs
========================

MC/DC uses `Sphinx <https://www.sphinx-doc.org/>`_ to generate its
documentation and `Read the Docs <https://about.readthedocs.com/>`_ to build
and host the documentation website.

This page introduces the subset of Sphinx needed to contribute to MC/DC's
documentation, including reStructuredText, document organization, automatic API
generation, and local documentation builds.


reStructuredText and Sphinx
---------------------------

We write files for Sphinx using a plaintext markup language called reStructuredText (rst).
`Click here for a rst Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
Sphinx builds an html file for every rst file in the documentation root directory and its subdirectories our documentation root directory is ``mcdc/docs/source/``. 
The root document, ``index.rst``, serves as the welcome page. 
The root directory also contains several subdirectories, each of which has its own ``index.rst`` file and several other rst files. 
It's useful to compare our rst files to their associated webpages to get a feel for how they translate.


Like any plaintext markup language, rst uses "explicit markup" for constructs that need special handling, such as including a code-block or cross-referencing other pages.
 
A block of explicit markup text starts with ".. " and is terminated by the next paragraph at the same level of indentation.

Sphinx creates webpage elements using explicit markup blocks called directives. 

.. tip::
   For example, this block was created using the `tip` directive!

::

  .. tip::
     For example, this block was created using the `tip` directive!

An explicit markup block without a directive is taken as a comment that will not appear on the webpage:
::

  .. For example, this is a comment.

In addition to directives for blocks of explicit markup, Sphinx handles in-line explicit markup with roles. 
For example, this equation :math:`a^2 + b^2 = c^2` was created using the `math` role.
::

  For example, this equation :math:`a^2 + b^2 = c^2` was created using the `math` role.

`Click here for a list of Sphinx directives <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_ and `click here for a list of Sphinx roles <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html>`_. 


The toctree
-----------

Sphinx's main directive is the `toctree` directive, which generates a table of contents tree (toctree) with links to other webpages in the build.
The listed documents should be named relative to the current document and excluding the .rst extension.
For example, the MC/DC docs root directory contains ``index.rst``, ``install.rst``, and a subdirectory ``user/`` that also contains its own ``index.rst``. 
The following on ``index.rst`` creates a table of contents on the main page with links to the install and user pages: 
::

  .. toctree::
     install
     user/index

Sphinx will build an html file for all rst files in the source directory and its subdirectories.
Sphinx will issue a warning if an html file isn't referenced in any toctree because that means that the generated webpage is not reachable through standard navigation.


Using autodoc and autosummary
-----------------------------

Within MC/DC's source code, we document functions and classes using docstrings.
`We use two Sphinx extensions <https://romanvm.pythonanywhere.com/post/autodocumenting-your-python-code-sphinx-part-ii-6/>`_ -- ``autodoc`` and ``autosummary`` -- to generate rst files for Sphinx using the existing docstrings in our source code.
For ``autodoc`` and ``autosummary`` to work, the docstrings within MC/DC's source code must be written in correct rst.

The ``autodoc`` extension includes a set of directives to document different chunks of code (e.g., modules, functions, classes). 
For example, below is the entire rst file that generates the :doc:`../../pythonapi/generated/mcdc.MaterialMG` page:

.. code-block::
   
   mcdc.MaterialMG
   ===============
   
   .. currentmodule:: mcdc
   
   .. autoclass:: MaterialMG

(That in-line reference was created using :code:`:doc:\`../../pythonapi/generated/mcdc.MaterialMG\``, by the way).

A rst file with an ``autodoc`` directive is required for each module or function that we would like to document.
Rather than create all of these rst files by hand, we use the ``autosummary`` extension to do it for us.

For example, let's look at the first ``autosummary`` directive in ``source/pythonapi/index.rst``, the file that governs the :doc:`../../pythonapi/index` page: 

.. code-block::

   .. autosummary::
      
      mcdc.Material
      mcdc.MaterialMG

This directive:
  #. Generates two files in ``pythonapi/generated/``: ``mcdc.Material.rst`` and ``mcdc.MaterialMG.rst``.
  #. Populates each file with the proper autoclass directive.
  #. Creates a table on :doc:`../../pythonapi/index` with entries mcdc.Material and mcdc.MaterialMG that link to the respective generated pages. 


Building Locally
----------------

We can check our work with a local build. 
Make sure you're in ``mcdc/docs/``:

#. Both Sphinx and furo (the package we use for website theming) should have been installed with MC/DC.
   To check, type ``sphinx-build --version`` on the commandline.
   If not installed, ``pip install sphinx furo``.
#. With Sphinx installed, run ``make html``.
   This builds local html files in ``mcdc/docs/build/``.
#. To launch your local html from the commandline, ``open build/html/index.html``.
   Check your work: has your content been added or changed as you expected?
#. Continue making changes to your local rst files, building locally, and launching the built html files until you're satisfied with how the website will look.

.. warning::
   In the process of creating MC/DC's documentation, ``autodoc`` *imports every python module that MC/DC imports*.

   This doesn't cause any issues when you build the webpages locally, because you already have all of MC/DC's requisite packages installed.

   However, this *WILL* cause issues with our documentation website host, readthedocs.
   Like you just did, readthedocs will checkout our repo and use Sphinx to build html files from our rst files, attempting to import all of MC/DC's packages along the way.
   There are some python packages, like ``mpi4py``, that readthedocs is unable to import, causing the documentation build to fail.
   
   **If you've added any new package imports to MC/DC's source code, add them to the** ``MOCK_MODULES`` **list in** ``mcdc/docs/source/conf.py``. 

   This will allow readthedocs to get past the imports without issue.


Once you're satisfied with your changes and have added any new modules to ``conf.py``, submit a PR!


