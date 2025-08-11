#! /usr/bin/env python3
import sRTp
import os
import sys
import glob
import multiprocessing as mp


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process logs that Arm got on Frontier to extract communicator imformation.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory storing the rank logs.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output pickle to store the parsed information.")
    parser.add_argument("-p", "--pattern", type=str, default='rank-*.out', help="Pattern, as a shell glob, to match rank files.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Enable verbose output.")
    parser.add_argument("--strict", action='store_true', help="Enable strict mode, which will raise exceptions on parsing errors instead of skipping them.")
    parser.add_argument("--jobs", "-j", type=int, default=mp.cpu_count(), help="Number of parallel jobs to run for processing.")
    args = parser.parse_args()

    verbose = args.verbose
    if not os.path.isdir(args.directory):
        print(f"Error: Directory {args.directory} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing rank files in {args.directory} with pattern {args.pattern}...")
    rank_files = glob.glob(os.path.join(args.directory, args.pattern))
    if not rank_files:
        print(f"No rank files found matching pattern {args.pattern} in directory {args.directory}.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(rank_files)} rank files.")
    
    communicators = {}
    with mp.Pool(processes=args.jobs) as pool:
        results = []
        for rank_file in rank_files:
            if verbose:
                print(f"Processing {rank_file}...")
            global_rank = int(os.path.basename(rank_file).split('-')[1].split('.')[0])
            result = pool.apply_async(sRTp.process_rank_file, (rank_file,))
            results.append((result, global_rank))
        for result, global_rank in results:
            comms = result.get()
            try:
                for comm in comms:
                    if comm.commId not in communicators:
                        communicators[comm.commId] = sRTp.globalComm(comm.commId, comm.size)
                    communicators[comm.commId].add_local_communicator(comm, global_rank)
            except Exception as e:
                print(f"Error processing {rank_file}: {e}", file=sys.stderr)
                print("Try running cleanup_output.py on the rank files to fix any issues with the output.", file=sys.stderr)
                sys.exit(1)

    print(f"Found {len(communicators)} unique communicators across all ranks.")
    comms_by_len = {}
    for comm_id, comm in communicators.items():
        if comm.size not in comms_by_len:
            comms_by_len[comm.size] = []
        comms_by_len[comm.size].append(comm_id)
    for key in comms_by_len.keys():
        print(f"Communicators of size {key}: {len(comms_by_len[key])} found.")
    
    output_file = args.output
    with open(output_file, 'wb') as f:
        import pickle
        pickle.dump(communicators, f)
    print(f"Communicators saved to {output_file}.")

if __name__ == "__main__":
    main()