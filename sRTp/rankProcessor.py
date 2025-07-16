#! /usr/bin/env python3

import analysisTypes
import parse

initPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ncclCommInitRank comm {localcomm} rank {rank} nranks {size} cudaDev {cudadev} nvmlDev {nvmldev} busId {busid} commId {globalcomm} - Init START")
initCompletePattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ncclCommInitRank comm {localcomm} rank {rank} nranks {size} cudaDev {cudadev} nvmlDev {nvmldev} busId {busid} commId {globalcomm} localSize {localsize} used {usedbytes} bytes on core {core} - Init COMPLETE")
opLaunchPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO {op}: opCount {opcount} sendbuff {sendbuff} recvbuff {recvbuff} count {count} datatype {datatype} op {opnum} root {root} comm {comm} [nranks={nranks}] stream {stream} task {task} globalrank {globalrank}")
kernelLaunchPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ## [{timestamp}] [{channelid}] {hwid} ncclDevFunc_{func}_{type}_{dtype} nw {nw} bi {bi} nc {nc} root {root} busId {busid} nRanks {nranks}")
kernelEndPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ## [{timestamp}] [{channelid}] {hwid} KE busId {busid} nRanks {nranks}")

def process_rank_file(rank_file):
    """
    Process the rank file to extract NCCL communicator information.
    
    Args:
        rank_file (str): Path to the rank file.
    
    Returns:
        list: A list of localComm objects representing the communicators.
    """
    communicators = []
    
    with open(rank_file, 'r') as file:
        commOpening = None
        for line in file:
            # Comm Init Lines look like this:
            # frontier00061:695134:695340 [0] NCCL INFO ncclCommInitRank comm 0x9bf11c0 rank 0 nranks 4096 cudaDev 0 nvmlDev 4 busId d1000 commId 0xf4592124255ac8f2 - Init START
            if "ncclCommInitRank" in line and "Init START" in line:
                r = initPattern.parse(line.strip())
                comm = analysisTypes.localComm(r['node'], r['globalcomm'], r['localcomm'], r['rank'], r['size'], r['busid'], r['cudadev'], r['nvmldev'])
                commOpening = r['localcomm']
                for ecomm in communicators:
                    assert ecomm.commId != comm.commId, f"Duplicate communicator ID found: {comm.commId}"
                communicators.append(comm)
                continue
            
            # Channel lines look like this:
            # frontier00061:695135:695311 [0] NCCL INFO Channel 00/08 :    0   3   1   2   6   5   7   4   8  11   9  10  14  13  15  12  16  19  17  18
            # frontier00061:695135:695311 [0] NCCL INFO Channel 03/0 : 0[d6000] -> 2[de000] via P2P/IPC comm 0x9bbf3b0 nRanks 4096
            # frontier00061:695135:695311 [0] NCCL INFO Channel 00/0 : 4092[d6000] -> 0[d6000] [receive] via NET/AWS Libfabric/2/GDRDMA comm 0x9bbf3b0 nRanks 4096
            if commOpening and "Channel" in line:
                for comm in communicators:
                    if comm.localId == commOpening:
                        # Here we would parse the channel information and add it to the communicator
                        # For now, we just store the channel log strip
                        comm.add_channel(line.strip())
                        continue

            # Comm Init Complete Lines look like this:
            # frontier00061:695135:695311 [0] NCCL INFO ncclCommInitRank comm 0x9bbf3b0 rank 0 nranks 4096 cudaDev 0 nvmlDev 5 busId d6000 commId 0xbc8a3ad231751b7 localSize 304 used 227459488 bytes on core 13 - Init COMPLETE
            if "Init COMPLETE" in line:
                assert commOpening is not None, "Init COMPLETE line found without a preceding Init START"
                # We need to find the communicator that matches the localcomm and globalcomm
                r = initCompletePattern.parse(line.strip())
                for comm in communicators:
                    if comm.localId == r['localcomm'] and comm.commId == r['globalcomm']:
                        commOpening = None
                        # Eventually we will call a method on the comm to finalize completion and sanity check
                        # channel information and the like, but I need to talk to Arm about the channel info first.
                        continue

            # Operation starts look like this:
            # frontier00061:695135:695135 [0] NCCL INFO AllGather: opCount 2 sendbuff 0x7ffc62d00000 recvbuff 0x7ff887600000 count 131072 datatype 9 op 0 root 0 comm 0x9bbf3b0 [nranks=4096] stream 0x90013c0 task 0 globalrank 0
            if "opCount" in line and "comm" in line:
                r = opLaunchPattern.parse(line.strip())
                for comm in communicators:
                    # NCCL communicators are identified by their localId
                    if comm.localId == r['comm']:
                        # Start the operation on the communicator
                        comm.start_operation(r['op'], r['opcount'])
                        continue
            
            # Kernel launches look like this:
            # frontier00061:695135:695318 [0] NCCL INFO ## [442464.629746] [00:00:00] 000000 KL HWID 42302510 ncclDevFunc_AllGather_RING_SIMPLE_Sum_i8 nw 4 bi 0 nc 8 root 0 busId d6000 nRanks 4096
            if "ncclDevFunc_" in line and "KL" in line:
                launchLogged = False
                r = kernelLaunchPattern.parse(line.strip())
                for comm in communicators:
                    if comm.size == int(r['nranks']):
                        for operation in comm.pending_operations:
                            if operation.getOperationType() == r['func']:
                                if launchLogged:
                                    raise ValueError("Ambiguous kernel launch: multiple communicators have ops of this type and size pending.")
                                # Start the kernel operation if it matches an operation on this communicator
                                if comm.start_kernel_if_match(r['func'], r['tid'], r['channelid']):
                                    launchLogged = True
                                    continue
                if not launchLogged:
                    raise ValueError(f"Kernel launch for {r['func']} with size {r['nranks']} did not match any pending operations.")

            #kernel end lines look like this:
            # frontier00061:695135:695522 [0] NCCL INFO ## [442510.054859] [00:00:00] 000000 KE busId d6000 nRanks 4096
            if "KE" in line and "busId" in line:
                endLogged = False
                r = kernelEndPattern.parse(line.strip())
                for comm in communicators:
                    if comm.size == int(r['nranks']):
                        for operation in comm.pending_operations:
                            # The end kernel logging doesn't give optype or size, so we have to go by
                            # tid and channelid alone
                            if comm.end_kernel_if_match(r['tid'], r['channelid']):
                                if endLogged:
                                    raise ValueError("Ambiguous kernel end: multiple communicators have ops of this type and size pending.")
                                endLogged = True
                                continue




    return communicators

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Process rank files and extract NCCL communicator information.")
    parser.add_argument(
        "rank_file",
        type=str,
        help="Path to the rank file to process."
    )

    args = parser.parse_args()
    comms = process_rank_file(args.rank_file)
    print(f"Found {len(comms)} communicators in the rank file.")
    for comm in comms:
        print(f"Communicator ID: {comm.commId} \n\tLocal ID: {comm.localId} \n\tRank: {comm.localRank} \n\tSize: {comm.size} \n\tBus ID: {comm.busID} \n\tCUDA Dev: {comm.cudaDev} \n\tNVML Dev: {comm.nvmlDev}")
        #print(f"Channels: {comm.channels}")
        if len(comm.pending_operations) == 0:
            print("\tNo pending operations.")
        for pending_op in comm.pending_operations:
            print(f"\tPending Operation: {pending_op.op_type} Seq Num: {pending_op.seq_num}")
        if len(comm.completed_operations) == 0:
            print("\tNo completed operations.")
        for completed_op in comm.completed_operations:
            print(f"\tCompleted Operation: {completed_op.op_type} Seq Num: {completed_op.seq_num}")


if __name__ == "__main__":
    main()