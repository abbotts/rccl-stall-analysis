#! /usr/bin/env python3
import os

def cleanup_file(input_path, output_path, unique_string):
    """Sweep the file and see if we need to insert any new lines into clobbered sections."""
    with open(input_path, 'r') as file:
        lines = file.readlines()

    with open(output_path, 'w') as file:
        for line in lines:
            # Check if the string "frontier" occurs more than once in the line
            if line.count(unique_string) > 1:
                # If it does, we will insert a newline before the second occurrence
                parts = line.split(unique_string)
                # Write the first part, then a newline, then the second part with "frontier" re-added
                for part in parts:
                    file.write(unique_string + part)
                    file.write("\n")
                continue
            # if line doesn't start with the unique string but contains it,
            # we will insert a newline before the first occurrence
            if "NCCL INFO" in line and not line.startswith(unique_string):
                #print(line)
                parts = line.split(unique_string, 1)
                #print(parts)
                # Write the first part, then a newline, then the second part with "frontier" re-added
                file.write(parts[0] + "\n")
                file.write(unique_string + parts[1])
                continue
        # line is clean, write it as is
            file.write(line)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cleanup output files by inserting newlines where necessary.")
    parser.add_argument("directory", type=str, help="Path to the directory containing files to be cleaned up.")
    parser.add_argument("--unique", type=str, default='frontier', help="String that should only occur at the start of a line, once per line.")
    args = parser.parse_args()

    outpath = args.directory + "-cleaned"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for filename in os.listdir(args.directory):
        cleanup_file(os.path.join(args.directory, filename), os.path.join(outpath, filename), args.unique)
        print(f"File {filename} has been cleaned up.")

if __name__ == "__main__":
    main()