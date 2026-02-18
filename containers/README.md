# MC/DC Container Images

This directory contains Singularity/Apptainer container definition files
for running MC/DC on HPC systems (LLNL, OSU) where Docker is not available.

## Files

- `mcdc-cpu.def`  — CPU-only execution
- `mcdc-cuda.def` — NVIDIA GPU execution (requires CUDA)
- `mcdc-rocm.def` — AMD GPU execution (requires ROCm)

## Target Systems

| Machine          | Institution | GPU                | Image      |
|------------------|-------------|--------------------|------------|
| Dane             | LLNL        | CPU only           | mcdc-cpu   |
| Tioga            | LLNL        | AMD MI-250X        | mcdc-rocm  |
| Tuolumne         | LLNL        | AMD MI300A APU     | mcdc-rocm  |
| COE DGX-2        | OSU         | NVIDIA V100        | mcdc-cuda  |
| COE DGX H100/H200| OSU         | NVIDIA H100/H200   | mcdc-cuda  |

## Building

### CPU image
```bash
singularity build mcdc-cpu.sif mcdc-cpu.def
```

### NVIDIA GPU image (OSU COE DGX)
```bash
singularity build mcdc-cuda.sif mcdc-cuda.def
```

### AMD GPU image (LLNL Tioga, Tuolumne)
```bash
singularity build mcdc-rocm.sif mcdc-rocm.def
```

## Running on HPC

### CPU (LLNL Dane)
```bash
singularity exec mcdc-cpu.sif python input.py
```

### NVIDIA GPU (OSU COE DGX)
```bash
singularity exec --nv mcdc-cuda.sif python input.py
```

### AMD GPU (LLNL Tioga, Tuolumne)
```bash
singularity exec --rocm mcdc-rocm.sif python input.py
```

### MPI + Singularity (SLURM)
```bash
srun -n 32 singularity exec mcdc-cpu.sif python input.py
```