.. MC/DC documentation master file, created by
   sphinx-quickstart on Fri Oct 27 14:14:47 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=================================
MC/DC: Monte Carlo Dynamic Code
=================================

MC/DC is a performant, scalable, and machine-portable Python-based 
Monte Carlo neutron transport software in active development.
It supports fully transient (aka dynamic) Monte Carlo transport and implements
novel methods and algorithms for neutron transport. MC/DC is purpose built to be
a rapid methods development platform for for modern HPCs and is targeting CPUs and GPUs.

MC/DC has support for continuous energy and multi-group transport.
It can solve more traditional k-eigenvalue problems (used to determine neutron population growth rates in reactors) as well as fully dynamic simulations.
It has a novel continuous geometry movement function that models transient elements (e.g., control rods or pulsed neutron experiments) more accurately than the step functions used by other codes.
It also supports some simple domain decomposition, with more complex algorithms currently being implemented.

MC/DC is machine portable and is validated to run on:

* linux-64 (x86)
* win-64 (x86 windows)
* osx-64 (x86, intel based macs)
* osx-arm64 (apple silicon based macs)
* linux-ppc64 (IBM POWER9)
* linux-nvidia-cuda
* linux-amd-rocm

MC/DC has been run on some of the largest super computers in the world including the Dane, `Lassen (#72) <https://www.top500.org/system/179567>`_, and `Tuolumne (#10) <https://www.top500.org/system/180308/>`_  machines from LLNL and been scaled upto 1000 nodes.
Primary development is done by the `Center for Exascale Monte Carlo Neutron Transport <https://cement-psaap.github.io/>`_ (CEMeNT)


.. image:: images/home/cement-logo-1.png
   :width: 650
   :alt: cement logo
   :align: center
   :target: https://cement-psaap.github.io/

with support from the following institutions

.. image:: images/home/psaapiii.png
   :width: 200
   :alt: PSAAP-III logo
   :target: https://psaap.llnl.gov/
.. image:: images/home/DOE_logo.png
   :width: 275
   :alt: DOE logo
   :target: https://www.energy.gov/
.. image:: images/home/NNSA_Logo.png
   :width: 275
   :alt: NNSA logo
   :target: https://www.energy.gov/nnsa/national-nuclear-security-administration

.. image:: images/home/osu-logo.png
   :width: 400
   :alt: Oregon State University logo
   :target: https://oregonstate.edu/
.. image:: images/home/nd-logo.png
   :width: 125
   :target: https://www.nd.edu/
   :alt: Noter Dame logo
.. image:: images/home/SU.png
   :width: 125
   :alt: Seattle university logo
   :target: https://www.seattleu.edu/
.. image:: images/home/ncsu-logo.png
   :width: 125
   :alt: North Carolina state university logo
   :target: https://www.ncsu.edu/

Work on within MC/DC has resulted in a number of journal publications and conference publications/presentations.
A full list of publications can be found on the `CEMeNT website <https://cement-psaap.github.io/publications/>`_ .

.. only:: html

   --------
   Contents
   --------

.. toctree::
    :maxdepth: 1

    install
    user/index
    examples/index
    contribution/index
    theory/index
    pythonapi/index
    pubs

.. sidebar-links::
    :caption: Links
    :pypi: mcdc
    :github:
    
    CEMeNT <https://cement-psaap.github.io>
    license <https://github.com/CEMeNT-PSAAP/MCDC/blob/main/LICENSE>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


To build the docs
=================

#. Install dependencies (we recommend: ``conda install sphinx`` and ``pip install furo sphinx_toolbox``). Note that these dependencies are not installed as part of base MC/DC.
#. From the `MCDC/docs/` directory, run ``make html`` to compile.
#. Launch ``build/html/index.html`` with your browser of choice.

To Cite MC/DC
=============

If you use MC/DC and would like to provide proper attribution
please cite our article in the Journal of Open Source software

.. code-block:: bibtex
        
    @article{morgan2024mcdc,
        title = {Monte {Carlo} / {Dynamic} {Code} ({MC}/{DC}): {An} accelerated
                 {Python} package for fully transient neutron transport and
                 rapid methods development},
        author = {Morgan, Joanna Piper and Variansyah, Ilham and Pasmann, Samuel L. and 
                  Clements, Kayla B. and Cuneo, Braxton and Mote, Alexander and
                  Goodman, Charles and Shaw, Caleb and Northrop, Jordan and Pankaj, Rohan and
                  Lame, Ethan and Whewell, Benjamin and McClarren, Ryan G. and Palmer, Todd S.
                  and Chen, Lizhong and Anistratov, Dmitriy Y. and Kelley, C. T. and
                  Palmer, Camille J. and Niemeyer, Kyle E.},
        journal = {Journal of Open Source Software},
        volume = {9},
        number = {96},
        year = {2024},
        pages = {6415},
        url = {https://joss.theoj.org/papers/10.21105/joss.06415},
        doi = {10.21105/joss.06415},
    }

If you are developing or working with specific numerical methods please take greater care
to cite the specific publications where that work is presented.
A selected list can be found on our :ref:`pubs` page.
Also check out an even longer list of associated publications on our
`center's publications page <https://cement-psaap.github.io/publications/>`_ .
