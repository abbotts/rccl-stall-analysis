#! /usr/bin/env python3
import os
import sRTp


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
        sRTp.rankProcessor.cleanup_file(os.path.join(args.directory, filename), os.path.join(outpath, filename), args.unique)
        #print(f"File {filename} has been cleaned up.")

if __name__ == "__main__":
    main()