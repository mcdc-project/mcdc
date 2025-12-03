import h5py, os, sys, argparse, fnmatch
import numpy as np
from colorama import Fore, Style

# Option parser
parser = argparse.ArgumentParser(description="MC/DC regression test")
parser.add_argument("--mode", type=str, choices=["python", "numba"], default="python")
parser.add_argument("--target", type=str, choices=["cpu", "gpu"], default="cpu")
parser.add_argument("--mpiexec", type=int, default=0)
parser.add_argument("--srun", type=int, default=0)
parser.add_argument("--name", type=str, default="ALL")
parser.add_argument("--skip", type=str, default="NONE")
parser.add_argument("--include_native_physics", default=False, action="store_true")
args, unargs = parser.parse_known_args()

# Parse
mode = args.mode
target = args.target
mpiexec = args.mpiexec
srun = args.srun
name = args.name
skip = args.skip
include_native_physics = args.include_native_physics

non_test_files = ["__pycache__", "MCDC-regression_test_data", "tmp"]

# Get test names
if name == "ALL":
    names = []
    for item in os.listdir():
        if os.path.isdir(item) and item not in non_test_files:
            names.append(item)
else:
    names = [item for item in os.listdir() if fnmatch.fnmatch(item, name)]
names.sort()

# Remove skipped if specified
if skip != "NONE":
    skips = [item for item in os.listdir() if fnmatch.fnmatch(item, skip)]
    for name in skips:
        print(Fore.YELLOW + "Note: Skipping %s" % name + Style.RESET_ALL)
        names.remove(name)

# Remove native physics if not incuded
native_physics_tests = ["pincell", "pincell-k_eigenvalue"]
if not include_native_physics:
    for name in native_physics_tests:
        print(Fore.YELLOW + "Note: Skipping %s" % name + Style.RESET_ALL)
        if name in names:
            names.remove(name)

# Skip domain decomp tests unless there are 4 MPI processes
temp = names.copy()
parallel_run = mpiexec > 0 or srun > 0
for name in names:
    if name == "slab_reed_dd" and (
        not parallel_run or not (mpiexec % 4 == 0 and srun % 4 == 0)
    ):
        temp.remove(name)
        print(
            Fore.YELLOW
            + "Note: Skipping %s (require multiple of 4 MPI ranks)" % name
            + Style.RESET_ALL
        )
    elif name == "slab_reed_dd_3d" and (
        not parallel_run or not (mpiexec % 16 == 0 and srun % 16 == 0)
    ):
        temp.remove(name)
        print(
            Fore.YELLOW
            + "Note: Skipping %s (require multiple of 16 MPI ranks)" % name
            + Style.RESET_ALL
        )

names = temp

# Skip iqmc if GPU run
if target == "gpu":
    temp = names.copy()
    for name in names:
        if ("iqmc" in name) or ("eigenvalue" in name):
            temp.remove(name)
            print(
                Fore.YELLOW + "Note: Skipping %s (GPU target)" % name + Style.RESET_ALL
            )
names = temp

# Data for each test
printouts = []
runtimes = []
flags = []
error_msgs = []
crashes = []
all_pass = True

# Run all tests
for i, name in enumerate(names):
    print("\n[%i/%i] " % (i + 1, len(names)) + name)
    error_msgs.append([])
    crashes.append(False)
    runtimes.append([0])

    # Change directory
    os.chdir(name)

    # Check test setup
    if not os.path.exists("input.py"):
        print(Fore.RED + "  input.py is missing\n" + Style.RESET_ALL)
        sys.exit()
    if not os.path.exists("answer.h5"):
        print(Fore.RED + "  answer.h5 is missing\n" + Style.RESET_ALL)
        sys.exit()

    # Delete output if exists
    if os.path.exists("output.h5"):
        os.remove("output.h5")

    # Run the test problem (redirect the stdout)
    if mpiexec > 1:
        gpus_per_task = ""
        if target == "gpu":
            gpus_per_task = f"--gpus-per-task=1 "
        os.system(
            "mpiexec -n %i %s python input.py --mode=%s --target=%s --output=output --no-progress-bar> tmp 2>&1"
            % (mpiexec, gpus_per_task, mode, target)
        )
    elif srun > 1:
        gpus_per_task = ""
        if target == "gpu":
            gpus_per_task = f"--gpus-per-task=1 "
        os.system(
            "srun -n %i %s python input.py --mode=%s --target=%s --output=output --no-progress-bar> tmp 2>&1"
            % (srun, gpus_per_task, mode, target)
        )
    else:
        os.system(
            "python input.py --mode=%s --target=%s --output=output --no-progress-bar > tmp 2>&1"
            % (mode, target)
        )
    with open("tmp") as f:
        printouts.append(f.read())
    os.remove("tmp")

    # Check if crashed
    if not os.path.exists("output.h5"):
        print(Fore.RED + "  Failed: Run crashed" + Style.RESET_ALL)
        all_pass = False
        crashes[-1] = True
        os.chdir("..")
        continue

    # Get the output and the answer key
    output = h5py.File("output.h5", "r")
    answer = h5py.File("answer.h5", "r")

    runtimes[-1] = output["runtime/total"][()]
    print("  (%.2f seconds)" % runtimes[-1][0])

    # Compare mean, sdev, and uq_var (if available)
    if "iqmc" not in output.keys():
        name_root = "tallies"
        for tally in [key for key in answer[name_root].keys()]:
            name_tally = name_root + "/" + tally
            for score in [key for key in answer[name_tally].keys()]:
                if score in ["grid"]:
                    continue
                name_score = name_tally + "/" + score
                for result in [key for key in answer[name_score].keys()]:
                    name = name_score + "/" + result
                    a = output[name][()]
                    b = answer[name][()]

                    # if (("sdev" in result) or ("uq_var" in result)) and (
                    if ("uq_var" in result) and (args.target == "gpu"):
                        continue
                    # Passed?
                    if np.isclose(a, b).all():
                        print(
                            Fore.GREEN + "  {}: Passed".format(name) + Style.RESET_ALL
                        )
                    else:
                        all_pass = False
                        error_msgs[-1].append(
                            "Differences in %s"
                            % (
                                name
                                + "/"
                                + result
                                + "\n"
                                + "{}\n".format(a - b)
                                + "Max difference: {}".format(np.max(np.abs(a - b)))
                            )
                        )
                        print(Fore.RED + "  {}: Failed".format(name) + Style.RESET_ALL)

        # Other quantities
        for result_name in ["k_mean", "k_sdev", "k_cycle", "k_eff"]:
            if result_name not in output.keys():
                continue

            a = output[result_name][()]
            b = answer[result_name][()]

            # Passed?
            if np.isclose(a, b).all():
                print(Fore.GREEN + "  {}: Passed".format(result_name) + Style.RESET_ALL)
            else:
                all_pass = False
                error_msgs[-1].append(
                    "Differences in {}\n{}".format(result_name, a - b)
                )
                print(Fore.RED + "  {}: Failed".format(result_name) + Style.RESET_ALL)

    # iQMC flux
    if "iqmc" in output.keys():
        for score in [key for key in output["iqmc/tally/"].keys()]:
            name = f"iqmc/tally/{score}/mean"
            a = np.squeeze(output[name][()])
            b = np.squeeze(answer[name][()])
            if np.isnan(b).all():
                continue
            # Passed?
            if np.isclose(a, b).all():
                print(Fore.GREEN + "  {}: Passed".format(score) + Style.RESET_ALL)
            else:
                all_pass = False
                error_msgs[-1].append("Differences in {}\n{}".format(score, a - b))
                print(Fore.RED + "  {}: Failed".format(score) + Style.RESET_ALL)

    # Close files
    output.close()
    answer.close()

    # Move back up
    os.chdir("..")

# Report test results
N_fails = 0
for i in range(len(names)):
    if crashes[i] or len(error_msgs[i]) > 0:
        N_fails += 1

print(
    "\nTests passed: "
    + Fore.GREEN
    + "%i/%i" % (len(names) - N_fails, len(names))
    + Style.RESET_ALL
)
print("Tests failed: " + Fore.RED + "%i/%i" % (N_fails, len(names)) + Style.RESET_ALL)
print("  (%.2f seconds)\n" % np.sum(np.array(runtimes)))
for i in range(len(names)):
    if crashes[i]:
        print("\n" + "=" * 80)
        print("\n## {} crashed:".format(names[i]))
        print(printouts[i])
    if len(error_msgs[i]) > 0:
        print("\n" + "=" * 80)
        print("\n## {} failed:".format(names[i]))
        print(printouts[i])
        print("\n===\n")
        for msg in error_msgs[i]:
            print("\n# " + msg + "\n")

assert all_pass
