#!/bin/bash
#
# This script is both a bash script and a python script.
# These first two lines are the bash part, which in turn
# launches python to interpret this file. In python,
# freestanding strings are comments, so these two lines are
# technically legal. The "exit" means that the bash shell
# will exit immediately, meaning the bash interpreter will
# never read any of the definitely-not-bash code that follows.
"python" "$0" "$@"
"exit"
# Yes, this is silly, but it's necessary to get shell-based
# batching commands to work on python scripts that could
# by executed with arbitrary python install paths. If we just
# shoved in '#!/bin/python', that would likely not be the python
# installation we want. Instead we delegate the path lookup to
# bash, which already knows where our python installation is.

import subprocess
import shutil
import sys
import os
import re

###############################################################################
# Here are the variables you will likely wish to modify the most
###############################################################################

# Valid values
# - 'always' : Always used cached info no matter what
# - 'first'  : Set up cached info with an initial run before
#              launching real jobs (reccomended)
# - 'never'  : Never used cached info
cache = "first"

# Valid values:
# - 'python' : Python-mode CPU execution
#              (SUPER slow. Probably shouldn't run this with
#               any of the other modes for any reasonably
#               sized problems.)
# - 'numba'  : Numba-mode CPU execution
#              (Semi-slow. Okay to run with smaller problems.)
# - 'event'  : Event-based GPU execution
# - 'async'  : Async GPU execution
strat_list = [
    #    "python",
    #    "numba",
    "event",
    #    "async",
]


# List of node counts to run jobs on.
node_count_list = [1]

# Number of samples to take for each configuration
sample_count = 3

# List of particle counts to run jobs for.
# This list uses the smallest particle count first so
# that cache-generating runs go quickly. After this, the
# counts goes in descending order to front-load long running
# jobs and allow short jobs to fill in the gaps.
particle_count_list = [
    100000,
    #    464158880000,
    #    166810050000,
    #    59948420000,
    #    21544340000,
    #    7742630000,
    #    2782550000,
    #    1000000000,
    #    359381366,
    #    129154966,
    46415888,
    16681005,
    5994842,
    2154434,
    774263,
    278255,
]


# The space of parameters that will be iterated through by
# the script. The order of iteration nesting is from the bottom up,
# with later entries acting like inner for loops.
param_space = {
    "TIME_SPLITS": [101],  # [ 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025 ],
    "X_SECT": [1.25, 2.5, 5, 10, 20],
    "PARTICLE_COUNT": particle_count_list,
}
# Feel free to add additional parameters, but make sure there are
# corresponding strings in your `input_template.py` file, otherwise
# they won't do anything and you'll have a bunch of additional runs
# that didn't do anything unique (assuming they don't just crash)
#
# NOTE: It is strongly reccomended that you place cache-breaking
# parameters before parameters that don't break the cache. This
# allows for the most parallelism and hence the fastest completion
# of jobs.

# A list of parameters which should force a cache reset whenever
# they change. ( The node count and strat both automatically
# force cache rests. ) Force-resets are disabled when cache="always".
reset_cache_on = ["TIME_SPLITS"]


# Set the email variable to your email if you want to be
# emailed when the program has completed.
email = None  # "bcuneo@seattleu.edu"


###############################################################################
# Handling command-line input
###############################################################################


mode_set = {
    "run": None,
    "dry_run": None,
    "collect": None,
}

if len(sys.argv) < 3:
    print(
        "A mode and batch name is required to run this script.\n"
        + "Example: 'scraper.py mode_here batch_name_here'"
    )
    exit(1)

mode = sys.argv[1]
batch_name = sys.argv[2]

if not mode in mode_set:
    mode_list = ", ".join([f"'{name}'" for name in mode_set])
    print(f"Mode '{mode}' not recognized.\nValid modes: {mode_list}")
    exit(1)

if (mode != "collect") and len(sys.argv) > 3:
    extra_args = ", ".join([f"'{arg}'" for arg in sys.argv[3:]])
    print(f"Unrecognized argument(s): {extra_args}")
    exit(1)


stat_types = ["avg", "min", "max", "each"]

stats = {}

if mode == "collect":
    bad = []
    for extra_arg in sys.argv[3:]:
        if not extra_arg in stat_types:
            bad.append(extra_arg)
        else:
            stats[extra_arg] = None

    if len(bad) > 0:
        bad_args = ", ".join([f"'{arg}'" for arg in bad])
        print(f"Unrecognized argument(s): {bad_args}")
        exit(1)

if len(stats) == 0:
    stats["avg"] = None

###############################################################################
# Function for making sure a directory exists
###############################################################################


def ensure_dir(dir_path):

    global mode

    if mode in ["run", "dry_run"]:
        print(f"Ensuring directory exists at '{dir_path}'")

    if mode != "run":
        return

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    elif not os.path.isdir(dir_path):
        print(f"ERROR: Creation of directory at '{dir_path}' blocked by file")
        exit(1)


###############################################################################
# Convenience function for running commands without arguments containing
# any whitespace
###############################################################################


def go_run(cmd_text, cwd=None, always=False, quiet=False):

    global mode

    if mode in ["run", "dry_run"]:
        print(cmd_text)

    if (not always) and (mode != "run"):
        return

    try:
        return subprocess.check_output(
            cmd_text.split(),
            stderr=subprocess.STDOUT,
            cwd=cwd,
        ).decode()
    except subprocess.CalledProcessError as exc:
        print(
            "ERROR: Child process exited with non-zero status code ",
            exc.returncode,
            "\nOUTPUT:\n",
            exc.output.decode(),
        )
        raise exc


###############################################################################
# Figuring out general configuration based on host, user, and
# current working directory
###############################################################################

machine = "mystery_machine"
base_path = os.getcwd()
base_job_path = base_path
username = go_run("id -u -n", always=True, quiet=True).strip()
hostname = go_run("hostname", always=True, quiet=True).strip()


if "tioga" in hostname:
    machine = "tioga"
    base_job_path = f"/p/lustre1/{username}/{batch_name}"
    machine_arena_size_opt = "--gpu_arena_size=100000000"
elif "lassen" in hostname:
    machine = "lassen"
    base_job_path = f"/p/gpfs1/{username}/{batch_name}"
    machine_arena_size_opt = "--gpu_arena_size=20000000"
else:
    print(
        "Zoinks, Scoob -- Looks like we're on a Mystery Machine!\n"
        + "Edit the script if you want it to run on this host!"
    )
    exit(1)


base_out_path = f"{base_path}/{batch_name}"
ensure_dir(base_out_path)
ensure_dir(base_job_path)


###############################################################################
# Function for replacing strings in a file
###############################################################################


def specialize_template(template_path, repl_list, output_path):

    template_file = open(template_path)
    text = template_file.read()
    template_file.close()

    for key, val in repl_list.items():
        key = str(key)
        val = str(val)
        text = re.sub(key, val, text)

    output_file = open(output_path, "w")
    output_file.write(text)
    output_file.close()


###############################################################################
# Function for iterating through parameter space
###############################################################################


def unroll_param_space(param_space):
    result = [{}]

    for param in reversed(param_space.keys()):
        sub_result = result
        result = []
        for value in param_space[param]:
            for sub_config in sub_result:
                entry = {param: value}
                for p, v in sub_config.items():
                    entry[p] = v
                result.append(entry)

    return result


###############################################################################
# Per-job logic
###############################################################################


# Fetch the latest Total time in the file called 'out' in the directory at
# the provided path. If no such time exists, return None
def get_result(out_path):
    result = None
    try:
        file = open(f"{out_path}/out")
        text = file.read()
        file.close()
        total_matches = re.findall(r"Total.*\(", text)
        total_match = total_matches[-1]
        total_text = re.search(r" [0-9]+\.[0-9]+ ", total_match)
        total = float(total_text.group(0).strip())
        if "minutes" in total_match:
            total *= 60
        result = str(total)
    except Exception as e:
        pass
    return result


# Returns the text to put at the start of a command to launch it on the
# job system used by the current machine
def launch_preamble(out_path, immediate=True):
    # Figure out the launch command that should receive our python command
    if machine == "tioga":
        rank_count = node_count * 8
        if immediate:
            directive = "submit"
        else:
            directive = "run"
        return f"flux {directive} -N {node_count} -n {rank_count} -g 1 --output={out_path}/out --error={out_path}/err --flags=waitable"
    elif machine == "lassen":
        rank_count = node_count * 4
        if immediate:
            immediate_opt = "-i"
        else:
            immediate_opt = ""
        return f"jsrun {immediate_opt} -n {rank_count} -r 4 -a 1 -g 1 -o {out_path}/out -k {out_path}/err"
    else:
        print("ERROR: Mystery machine")
        return None


def handle_job(strat, node_count, sample_index, config):

    global cache_needs_refresh
    global last_cache

    # Create a string identifier to signify the combination of
    # parameters being used
    param_sig = ""
    for k, v in config.items():
        param_sig += f"_{k}-{v}"
    dir_name = f"run_strat-{strat}_nodes-{node_count}{param_sig}-sample_{sample_index}"

    # Setup directories to execute the job and to store the output
    out_path = f"{base_out_path}/{dir_name}"
    ensure_dir(out_path)
    job_path = f"{base_job_path}/{dir_name}"
    ensure_dir(job_path)

    script_path = f"{job_path}/input.py"
    cache_path = f"{job_path}/__harmonize_cache__"

    # Check to see if a result was already found during a previous scraper
    # batch execution
    result = get_result(out_path)

    # If in collection mode, just print the result
    if mode == "collect":
        if result == None:
            return {dir_name: None}
        else:
            return {dir_name: float(result)}
    # If NOT in collection mode, skip run if a result is already present
    elif result != None:
        return {dir_name: float(result)}

    # Script files don't need to be generated for dry runs
    if mode == "run":
        # Perform find/replace on `input_template.py`, saving the
        # modified text to the path the job should run in
        specialize_template("./input_template.py", config, script_path)

    target_opt = ""
    arena_size_opt = ""
    mode_opt = ""
    strat_opt = ""
    caching_opt = ""

    # Disable caching if the settings indicate so
    if cache != "never":
        caching_opt = "--caching"

    # Handle options that should be supplied for each strat
    if strat == "python":
        target_opt = "--target=cpu"
        mode_opt = "--mode=python"
    elif strat == "numba":
        target_opt = "--target=cpu"
        mode_opt = "--mode=numba"
    elif strat == "event":
        target_opt = "--target=gpu"
        strat_opt = "--gpu_strat=event"
        mode_opt = "--mode=numba"
    elif strat == "async":
        target_opt = "--target=gpu"
        strat_opt = "--gpu_strat=async"
        mode_opt = "--mode=numba"
    else:
        print(f"Unrecognized strat '{strat}'")

    # The actual python command we care about running
    base_cmd = f"python3 {script_path} {mode_opt} {target_opt} {arena_size_opt} {strat_opt} {caching_opt}"

    launch_cmd = launch_preamble(out_path)

    # Print the command for diagnostic purposes, then run it
    cmd = f"{launch_cmd} {base_cmd}"

    # If we need to prep the cache, perform a prior foreground run with caching off
    if cache_needs_refresh:

        wait_all(quiet=True)

        # Clear out the old cache
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)

        # Run the command in the foreground
        print("Refreshing cache.")
        cache_launch_cmd = launch_preamble(out_path, immediate=False)
        cache_cmd = f"{cache_launch_cmd} {base_cmd} --clear_cache"
        go_run(cache_cmd, cwd=job_path)

        cache_needs_refresh = False
        last_cache = cache_path

    # If we didn't already just produce the cache, either copy it over
    # from our previous cache-making run or (in the case of cache="never")
    # remove the current cache without replacing it.
    if (mode == "run") and (cache_path != last_cache):
        # Copy over the __harmonize_cache__, if applicable
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        if os.path.exists(last_cache) and (cache != "never"):
            shutil.copytree(last_cache, cache_path)

    # Run the actual command
    go_run(cmd, cwd=job_path)

    return {}


###############################################################################
# Waiting logic
###############################################################################
def wait_all(quiet=False):
    wait_success = False
    if machine == "tioga":
        try:
            result = go_run("flux job wait --all")
            if result != None:
                print(result)
            wait_success = True
        except subprocess.CalledProcessError as exc:
            pass
    elif machine == "lassen":
        try:
            result = go_run("jswait all")
            if result != None:
                print(result)
            wait_success = True
        except subprocess.CalledProcessError as exc:
            pass

    if not quiet:
        try:
            if wait_success and (email != None):
                subject = f"{batch_name} finished"
                message = f"All jobs finished for scraper batch {batch_name}"
                subprocess.check_output(
                    ["mail", "-s", subject, email],
                    stderr=subprocess.STDOUT,
                    input=str.encode(message),
                ).decode()
        except subprocess.CalledProcessError as exc:
            pass


###############################################################################
# Logic for printing stats
###############################################################################
def print_stats(result_set):
    global sample_count

    minim = None
    maxim = None
    avg = None

    if "each" in stats:
        for key, val in result_set.items():
            print(f"{key}, {val}")

    some_key = None

    for key, val in result_set.items():

        some_key = key

        if val == None:
            minim = None
            maxim = None
            avg = None
            break

        if minim == None:
            minim = val
        else:
            minim = min(minim, val)

        if maxim == None:
            maxim = val
        else:
            maxim = max(maxim, val)

        if avg == None:
            avg = val
        else:
            avg += val

    if avg != None:
        avg /= sample_count

    key_comps = some_key.split("_")
    signature = "_".join(key_comps[:-1])

    if "min" in stats:
        print(f"{signature}_min, {minim}")

    if "max" in stats:
        print(f"{signature}_max, {maxim}")

    if "avg" in stats:
        print(f"{signature}_avg, {avg}")


###############################################################################
# Logic for re-caching as variables change
###############################################################################
last_config = None
last_cache = None


def param_broke_cache(last_config, config):
    if last_config == None:
        return True

    for param in reset_cache_on:
        if last_config[param] != config[param]:
            return True

    return False


###############################################################################
# Diagnostic info
###############################################################################

print(f"""
SCRAPER

running on  : '{machine}'
username    : '{username}'
working dir : '{base_path}'
job dir     : '{base_job_path}'
""")


###############################################################################
# Job launching loops
###############################################################################


for strat in strat_list:

    for node_count in node_count_list:

        cache_needs_refresh = True

        for config in unroll_param_space(param_space):

            if param_broke_cache(last_config, config):
                cache_needs_refresh = True

            result_set = {}
            for sample_index in range(sample_count):
                result = handle_job(strat, node_count, sample_index, config)
                last_config = config

                for key, val in result.items():
                    result_set[key] = val

            if mode == "collect":
                print_stats(result_set)

if mode == "run":
    wait_all()
