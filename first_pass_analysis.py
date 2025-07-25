#! /usr/bin/env python3

import sRTp
import os
import sys

# This script expects to read in a pickled file containing communicator information then will
# run basic consistency and analysis checks on it and print pertinent communicator information.
# If you give it a communicator ID, it will run the proxy_stall analysis on that communicator and print the results.

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze communicator information from a pickled file.")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input pickle file containing communicator information.")
    parser.add_argument("-c", "--comm_ids", type=str, nargs='+', help="Specific communicator IDs to analyze for proxy stalls.")
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(input_file, 'rb') as f:
        import pickle
        communicators = pickle.load(f)
    print(f"Loaded {len(communicators)} communicators from {input_file}.")

    trace_ids = args.comm_ids if args.comm_ids else []

    comms_by_len = {}
    for comm_id, comm in communicators.items():
        if comm.size not in comms_by_len:
            comms_by_len[comm.size] = []
        comms_by_len[comm.size].append(comm_id)

    print("Communicators by size:")
    for size, comm_ids in comms_by_len.items():
        print(f"Size {size}: {len(comm_ids)} communicators")
    print("Focusing only on the largest communicator sizes.")
    biggest = sorted(comms_by_len.keys(), reverse=True)[0]

    for comm_id in comms_by_len[biggest]:
        print("\n##########################################\n")
        print(f"Communicator {comm_id} has size {biggest}.")
        comm = communicators[comm_id]
        print(f"Consistent: {comm.check_consistency()}")
        print("\n")
        print(f"{comm.get_completed_opcounts()[0]} Operations completed and pending (nans suggest pending):")
        durations = comm.get_completed_durations(fillMissing=True)

        try:
            print("\nCompleted operations:")
            for iop, op in enumerate(comm.get_completed_operations()):
                print(f"\tOperation {iop}: {op.op_type} count: {op.count} dtype: {op.dtype} Times: max {durations[:, iop].max()} ms min {durations[:, iop].min()} ms mean {durations[:, iop].mean()} ms")

        except ValueError:
            print("Not all local ranks have the same number of operations completed, cannot print operation details.")

        try:
            print("\nPending operations:")
            for iop, op in enumerate(comm.get_pending_operations()):
                print(f"\tOperation {iop}: {op.op_type} count: {op.count} dtype: {op.dtype}")
        except ValueError:
            print("Not all local ranks have the same number of pending operations, cannot print operation details.")

        if comm_id in trace_ids:
            print(f"\nAnalyzing proxy stalls for communicator {comm_id}.")
            stalls = comm.get_proxy_stall_counts().sum()
            if stalls > 0:
                print(f"Proxy stalls detected: {stalls}")
                comm.trace_proxy_stalls()
                print("Proxy stall analysis complete.")
            else:
                print("No proxy stalls detected.")
            trace_ids.remove(comm_id)

    if trace_ids:
        print(f"Warning: The following communicator IDs were specified but not found in the data: {trace_ids}", file=sys.stderr)
    print("Analysis complete.")

if __name__ == "__main__":
    main()