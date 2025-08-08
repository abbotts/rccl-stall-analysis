#! /usr/bin/env python3
import os
import sRTp
import sRTp.ArmRankFiles


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cleanup output files by inserting newlines where necessary.")
    parser.add_argument("directory", type=str, help="Path to the directory containing files to be cleaned up.")
    parser.add_argument("--unique", type=str, default='frontier', help="String that should only occur at the start of a line, once per line.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Enable verbose output.")
    parser.add_argument("--truncate", action='store_true', help="Silently drop truncated lines at the end of files instead of warning.")
    args = parser.parse_args()

    outpath = args.directory + "-cleaned"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for filename in os.listdir(args.directory):
        sRTp.ArmRankFiles.cleanup_file(os.path.join(args.directory, filename), os.path.join(outpath, filename), args.unique, args.truncate)
        if args.verbose:
            print(f"File {filename} has been cleaned up.")

if __name__ == "__main__":
    main()