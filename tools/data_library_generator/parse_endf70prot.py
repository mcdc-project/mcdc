# This script was written by ChatGPT with Ethan Lame's instructions
import os

input_file = "/home/ethan_lame/MCDC/acelib/endf70prot"  # your big file
output_dir = "/home/ethan_lame/MCDC/acelib/"  # where split files go

os.makedirs(output_dir, exist_ok=True)

current_file = None

with open(input_file, "r") as f:
    for line in f:
        # Check for start of new isotope block
        if ".70h" in line:
            # Close previous file if open
            if current_file is not None:
                current_file.close()

            # Extract filename (first token)
            filename = line.strip().split()[0]

            # Open new file
            filepath = os.path.join(output_dir, filename)
            current_file = open(filepath, "w")

            print(f"Creating {filename}")

        # Write line if a file is open
        if current_file is not None:
            current_file.write(line)

# Close last file
if current_file is not None:
    current_file.close()
