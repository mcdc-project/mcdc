#!/bin/bash

# Check python version
if ! { python3 -c 'import sys; assert sys.version_info < (3,12)' > /dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info >= (3,9)' > /dev/null 2>&1; }; then
  v=$(python3 --version)
  p=$(which python)
  echo "ERROR: Python version must be < 3.12 and >= 3.9."
  echo "    Found $v at $p."
  echo "ERROR: Installation failed."
  exit 1
fi

# Install or build mpi4py
if [ $# -eq 0 ]; then
  conda install mpi4py <<< "y"
fi
while [ $# -gt 0 ]; do
  case $1 in
    --hpc)
      # Rename legacy compiler option in conda
      s=$(which python)
      s=${s//bin\/python/compiler_compat}

      if [ ! -f $s/ld.bak ] && [ -f $s/ld ]; then
        mv $s/ld $s/ld.bak
      fi

      mkdir installs; cd installs
      wget https://github.com/mpi4py/mpi4py/releases/download/3.1.4/mpi4py-3.1.4.tar.gz -q
      tar -zxf mpi4py-3.1.4.tar.gz
      cd mpi4py-3.1.4
      python setup.py install
      cd ../../
      rm -rf installs/
      ;;

    --config_cont_lib)
      bash config_cont_energy.sh
  ;;
  esac
  shift
done

# Install MC/DC module (and the dependencies)
pip install -e .

# Install pre-commit hook
pre-commit install
