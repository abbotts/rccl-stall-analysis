#! /usr/bin/env python3

import sRTp.ncclTypes as ncclTypes
import parse
import sys

initPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ncclCommInitRank{impl} comm {localcomm} rank {rank} nranks {size} cudaDev {cudadev} nvmlDev {nvmldev} busId {busid} commId {globalcomm} - Init START")
initCompletePattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ncclCommInitRank{impl} comm {localcomm} rank {rank} nranks {size} cudaDev {cudadev} nvmlDev {nvmldev} busId {busid} commId {globalcomm}{usage}- Init COMPLETE")
opLaunchPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO {op}: opCount {opcount} sendbuff {sendbuff} recvbuff {recvbuff} count {count} datatype {datatype} op {opnum} root {root} comm {comm} [nranks={nranks}] stream {stream} task {task} globalrank {globalrank}")
kernelLaunchPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ## [{timestamp}] [{channelid}] {hwid} ncclDevFunc_{func}_{type}_{dtype} nw {nw} bi {bi} nc {nc} root {root} busId {busid} nRanks {nranks}")
kernelEndPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO ## [{timestamp}] [{channelid}] {hwid} KE busId {busid} nRanks {nranks}")
ipcPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO Channel {channelid}/{channelnum} : {src}[{srcbusid}] -> {dst}[{dstbusid}] via P2P/IPC comm {comm} nRanks {nranks}")
ofiPattern = parse.compile("{node}:{pid}:{tid} [{unk}] NCCL INFO Channel {channelid}/{channelnum} : {src}[{srcbusid}] -> {dst}[{dstbusid}] [{direction}] via NET/{mechanism}/{ofi_details} comm {comm} nRanks {nranks}")
proxyPattern = parse.compile("{proxy} coll:{collid} comm:{comm} [{direction}] dtype:{dtype} redOp:{redop} proto:{proto}  nb:{nb} ns:{ns} p:{p} t:{t} r:{r}, d:{d}   myrank:{myrank} peer:{peer} chan:{chan} tail:{tail} recvtail:{recvtail} reg:{reg} connSz:{connsz}(retries:{retries})]")

def process_rank_file(rank_file, strict=False):
    """
    Process the rank file to extract NCCL communicator information.

    This function assumes the rank file is from Arm Patinyasakdikul's NCCL tests on Frontier and assume
    interleaving and the specific types of outputs in those files.

    Currently if it fails to parse a proxy dump line it will print a warning and continue processing the rest of the file.
    There are often multiple of these. Strict mode will raise an exception instead.

    If any non-proxy dump parse fails it will raise an exception and stop processing the file.
    
    Args:
        rank_file (str): Path to the rank file.
        strict (bool): If True, raise exceptions on parsing errors instead of skipping them.
    
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
                comm = ncclTypes.localComm(r['node'], r['globalcomm'], r['localcomm'], r['rank'], r['size'], r['busid'], r['cudadev'], r['nvmldev'])
                commOpening = r['localcomm']
                for ecomm in communicators:
                    assert ecomm.commId != comm.commId, f"Duplicate communicator ID found: {comm.commId}"
                communicators.append(comm)
                continue
            
            # Channel lines look like this:
            # frontier00061:695135:695311 [0] NCCL INFO Channel 03/0 : 0[d6000] -> 2[de000] via P2P/IPC comm 0x9bbf3b0 nRanks 4096
            # frontier00061:695135:695311 [0] NCCL INFO Channel 00/0 : 4092[d6000] -> 0[d6000] [receive] via NET/AWS Libfabric/2/GDRDMA comm 0x9bbf3b0 nRanks 4096
            if commOpening and "Channel" in line:
                for comm in communicators:
                    if comm.localId == commOpening:
                        # Rings get connected first, then trees
                        algo = "Ring"
                        if comm.rings_connected:
                            algo = "Tree"

                        # Here we would parse the channel information and add it to the communicator
                        if "P2P/IPC" in line:
                            channel_info = ipcPattern.parse(line.strip())
                            # IPC channels are always self to peer, so just grab the dst
                            comm.add_peer_to_channel(int(channel_info['channelid']), int(channel_info['dst']), 'both', 'IPC', algo=algo)
                        elif "NET/" in line:
                            channel_info = ofiPattern.parse(line.strip())
                            #print(line.strip())
                            #print(channel_info)
                            if channel_info['direction'] == 'send':
                                comm.add_peer_to_channel(int(channel_info['channelid']), int(channel_info['dst']), 'send', 'OFI', algo=algo)
                            elif channel_info['direction'] == 'receive':
                                comm.add_peer_to_channel(int(channel_info['channelid']), int(channel_info['src']), 'receive', 'OFI', algo=algo)

                        continue

            # Playing fast and loose here.. there's a risk of channels being printed for overlapping communicators, so we mark rings and trees constructed when we see
            # the message. No other way to do it at the moment.
            if "Connected all rings" in line:
                for comm in communicators:
                    if comm.localId == commOpening:
                        comm.finish_rings()
                        continue
            if "Connected all trees" in line:
                for comm in communicators:
                    if comm.localId == commOpening:
                        comm.finish_trees()
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
                opRecorded = False
                r = opLaunchPattern.parse(line.strip())
                try:
                    for comm in communicators:
                        # NCCL communicators are identified by their localId
                        if comm.localId == r['comm']:
                            # Start the operation on the communicator
                            comm.start_operation(r['op'], r['opcount'], r['count'], r['datatype'])
                            opRecorded = True
                            continue
                except Exception as e:
                    # Handle parsing errors or other exceptions
                    print(f"Error processing operation start: {line.strip()}")
                    print(e)
                    raise e
                
                if not opRecorded:
                    print(f"Bad line: {line.strip()}", file=sys.stderr)
                    print("Operation didn't start on any communicators.", file=sys.stderr)
                    for comm in communicators:
                        print(f"Communicator {comm.commId}, size {comm.size} has {len(comm.pending_operations)} pending operations.")
                    raise ValueError(f"Operation start for {r['op']} with size {r['nranks']} did not match any communicators.")
            
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
                                if comm.start_kernel_if_match(r['func'], r['tid'], r['channelid'], r['timestamp']):
                                    launchLogged = True
                                    continue
                if not launchLogged:
                    print(f"Bad line: {line}", file=sys.stderr)
                    for comm in communicators:
                        print(f"Communicator {comm.commId}, size {comm.size} has {len(comm.pending_operations)} pending operations.")
                        for op in comm.pending_operations:
                            print(f"\tPending Operation: {op.op_type} Seq Num: {op.seq_num}")
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
                            try:
                                if comm.end_kernel_if_match(r['tid'], r['channelid'], r['timestamp']):
                                    if endLogged:
                                        raise ValueError("Ambiguous kernel end: multiple communicators have ops of this type and size pending.")
                                    #print(f"Logging kernel end for {r['tid']} and channel {r['channelid']} on communicator {comm.commId}")
                                    #print(line)
                                    endLogged = True
                                    continue
                            except Exception as e:
                                print("Error ending kernel operation on line:")
                                print(line.strip())
                                raise e
                if not endLogged:
                    print(f"Bad line: {line.strip()}", file=sys.stderr)
                    for comm in communicators:
                        print(f"Communicator {comm.commId}, size {comm.size} has {len(comm.pending_operations)} pending operations.")
                        for op in comm.pending_operations:
                            print(f"\tPending Operation: {op.op_type} Seq Num: {op.seq_num}")
                    raise ValueError(f"Kernel end for tid {r['tid']} and channelid {r['channelid']} did not match any pending operations.")

            # A proxy print looks like this:
            # 0x7ff5ee771d48 [0-1|0| coll:3 comm:0x1b885440 [SEND] dtype:9 redOp:0 proto:2  nb:1048576 ns:16380 p:4436 t:4428 r:0, d:4428   myrank:6 peer:10 chan:1 tail:4428 recvtail:4428 reg:0 connSz:-1(retries:265446404)]
            if "recvtail" in line and "myrank" in line:
                r = proxyPattern.parse(line.strip())
                # Proxy print lines get broken *a lot*, so let's gracefully fail unless we've been told to be strict
                if r is None:
                    if strict:
                        raise ValueError(f"Failed to parse proxy print line: {line.strip()}")
                    else:
                        print(f"WARNING: Failed to parse proxy print line: {line.strip()}", file=sys.stderr)
                        continue
                for comm in communicators:
                    if comm.localId == r['comm']:
                        comm.add_proxy_print(int(r['peer']), int(r['chan']),
                                             r['direction'], int(r['tail']),
                                             int(r['recvtail']), int(r['retries']),
                                             int(r['collid']), int(r['dtype']),
                                             int(r['redop']), int(r['proto']),
                                             int(r['nb']), int(r['ns']),
                                             int(r['p']), int(r['t']),
                                             int(r['r']), int(r['d']),
                                             line.strip())
                        continue

    return communicators

def cleanup_file(input_path, output_path, unique_string, truncate=False):
    """Process an input file to make sure it can be read by process_rank_file.
    
    Arguments:
        input_path (str): Path to the input file to be cleaned up.
        output_path (str): Path to the output file where cleaned content will be written.
        unique_string (str): A string that should only occur at the start of a line,

    The unique string should be something like a node name that will only appear once per line (if at all).
    We'll use that to detect skipped new lines and automatically correct them.

    After automatically correcting any indentation issues, we'll try to parse following the same rules as process_rank_file.
    If a parse fails we'll try any easy corrections we can think of from past experiences.
    If those don't work, we'll print the file name, line number, and the line itself to stderr and continue processing the rest of the file.

    """
    with open(input_path, 'r') as file:
        lines = file.readlines()

    with open(output_path, 'w') as file:
        active_ops_lines = []
        stored_partial = None
        for lnum, line in enumerate(lines):
            # If there's an ACTIVE OPS print in the line we need to just
            # run forward and capture all the proxy, storing the contents of the line
            # before the active opts print, then flush the proxy buffer and then the active ops print
            #print(f"Processing line {lnum}: {line.strip()}", file=sys.stderr)
            #if stored_partial is not None:
                #print(f"Stored Partial Line: {stored_partial.strip()}", file=sys.stderr)
                #print(f"active_ops_lines: {active_ops_lines}", file=sys.stderr)
            if "ACTIVE OPS" in line:
                # If this was the start of a line then we probably didn't interrupt anything,
                # so we can just continue.
                if line.startswith("ACTIVE OPS"):
                    continue
                if stored_partial is not None:
                    print(f"Error: Found ACTIVE OPS print in line {lnum} of file {output_path} but there was a partial line stored from before. This is likely a bug in cleanup_output.py.", file=sys.stderr)
                    raise ValueError("Found ACTIVE OPS print in line but there was a partial line stored from before.")
                parts = line.split("ACTIVE OPS")
                stored_partial = parts[0]
                for part in parts[1:]:
                    active_ops_lines.append("ACTIVE OPS" + part)
                continue
            if stored_partial is not None:
                if "recvtail" in line or line.startswith("|") or line.strip() == "v" or (line.startswith("[") and line.strip().endswith("]")):
                    active_ops_lines.append(line)
                    continue
                elif line.strip() == "" and len(active_ops_lines) > 0:
                    file.writelines(active_ops_lines)
                    active_ops_lines = []
                    continue
                  #print(f"Stored Partial Line: {stored_partial}", file=sys.stderr)
                    #print("Writing stored partial line before current line", file=sys.stderr)
                #print(f"Stored partial line: {stored_partial.strip()}", file=sys.stderr)
                line = stored_partial + line
                stored_partial = None
                #print(f"Writing stored partial line before current line: {line.strip()}", file=sys.stderr)

            towrite = []
            # Check if the string "frontier" occurs more than once in the line
            if line.count(unique_string) > 1:
                # If it does, we will insert a newline before the second occurrence
                parts = line.split(unique_string)
                # Write the first part, then a newline, then the second part with "frontier" re-added
                for part in parts:
                    towrite.append(unique_string + part + "\n")
            # if line doesn't start with the unique string but contains it,
            # we will insert a newline before the first occurrence
            elif unique_string in line and not line.startswith(unique_string):
                parts = line.split(unique_string, 1)
                # Write the first part, then a newline, then the second part with "frontier" re-added
                towrite.append(parts[0] + "\n")
                towrite.append(unique_string + parts[1])
                #print(towrite)
            else:
                # line is clean, write it as is
                towrite.append(line)
            
            for il, lw in enumerate(towrite):
                def print_failure(example):
                    if lnum == len(lines) - 1:
                        towrite[il] = "<DROPPED FOR TRUNCATION>"
                        if truncate:
                            return
                        print(f"This is the last line in the file and probably truncated. Dropping", file=sys.stderr)
                    print("***********************", file=sys.stderr)
                    print(f"Error parsing line {lnum} in {input_path}:", file=sys.stderr)
                    print(f"Got: \n{line.strip()}", file=sys.stderr)
                    print(f"Expected something like: \n{example}", file=sys.stderr)

                # Comm Init Lines look like this:
                example = "frontier00061:695134:695340 [0] NCCL INFO ncclCommInitRank comm 0x9bf11c0 rank 0 nranks 4096 cudaDev 0 nvmlDev 4 busId d1000 commId 0xf4592124255ac8f2 - Init START"
                if "ncclCommInitRank" in lw and "Init START" in lw:
                    r = initPattern.parse(lw.strip())
                    if r is None:
                        print_failure(example)
                        continue
            
                # Channel lines look like this:
                # frontier00061:695135:695311 [0] NCCL INFO Channel 03/0 : 0[d6000] -> 2[de000] via P2P/IPC comm 0x9bbf3b0 nRanks 4096
                # frontier00061:695135:695311 [0] NCCL INFO Channel 00/0 : 4092[d6000] -> 0[d6000] [receive] via NET/AWS Libfabric/2/GDRDMA comm 0x9bbf3b0 nRanks 4096
                if "Channel" in line:
                    if "P2P/IPC" in lw:
                        channel_info = ipcPattern.parse(lw.strip())
                        if channel_info is None:
                            print_failure("frontier00061:695135:695311 [0] NCCL INFO Channel 03/0 : 0[d6000] -> 2[de000] via P2P/IPC comm 0x9bbf3b0 nRanks 4096")
                            continue
                    elif "NET/" in lw:
                        channel_info = ofiPattern.parse(lw.strip())
                        if channel_info is None:
                            print_failure("frontier00061:695135:695311 [0] NCCL INFO Channel 00/0 : 4092[d6000] -> 0[d6000] [receive] via NET/AWS Libfabric/2/GDRDMA comm 0x9bbf3b0 nRanks 4096")
                            continue
                
                # Comm Init Complete Lines look like this:
                # frontier00061:695135:695311 [0] NCCL INFO ncclCommInitRank comm 0x9bbf3b0 rank 0 nranks 4096 cudaDev 0 nvmlDev 5 busId d6000 commId 0xbc8a3ad231751b7 localSize 304 used 227459488 bytes on core 13 - Init COMPLETE
                if "Init COMPLETE" in line:
                    # We need to find the communicator that matches the localcomm and globalcomm
                    r = initCompletePattern.parse(lw.strip())
                    if r is None:
                        print_failure("frontier00061:695135:695311 [0] NCCL INFO ncclCommInitRank comm 0x9bbf3b0 rank 0 nranks 4096 cudaDev 0 nvmlDev 5 busId d6000 commId 0xbc8a3ad231751b7 localSize 304 used 227459488 bytes on core 13 - Init COMPLETE")
                        continue
                
                # Operation starts look like this:
                # frontier00061:695135:695135 [0] NCCL INFO AllGather: opCount 2 sendbuff 0x7ffc62d00000 recvbuff 0x7ff887600000 count 131072 datatype 9 op 0 root 0 comm 0x9bbf3b0 [nranks=4096] stream 0x90013c0 task 0 globalrank 0
                if "opCount" in lw and "comm" in lw:
                    r = opLaunchPattern.parse(lw.strip())
                    if r is None:
                        print_failure("frontier00061:695135:695135 [0] NCCL INFO AllGather: opCount 2 sendbuff 0x7ffc62d00000 recvbuff 0x7ff887600000 count 131072 datatype 9 op 0 root 0 comm 0x9bbf3b0 [nranks=4096] stream 0x90013c0 task 0 globalrank 0")
                        continue
            
                # Kernel launches look like this:
                # frontier00061:695135:695318 [0] NCCL INFO ## [442464.629746] [00:00:00] 000000 KL HWID 42302510 ncclDevFunc_AllGather_RING_SIMPLE_Sum_i8 nw 4 bi 0 nc 8 root 0 busId d6000 nRanks 4096
                if "ncclDevFunc_" in line and "KL" in line:
                    r = kernelLaunchPattern.parse(line.strip())
                    if r is None:
                        print_failure("frontier00061:695135:695318 [0] NCCL INFO ## [442464.629746] [00:00:00] 000000 KL HWID 42302510 ncclDevFunc_AllGather_RING_SIMPLE_Sum_i8 nw 4 bi 0 nc 8 root 0 busId d6000 nRanks 4096")
                        continue
                
                #kernel end lines look like this:
                # frontier00061:695135:695522 [0] NCCL INFO ## [442510.054859] [00:00:00] 000000 KE busId d6000 nRanks 4096
                if "KE" in lw and "busId" in lw:
                    r = kernelEndPattern.parse(lw.strip())
                    if r is None:
                        print_failure("frontier00061:695135:695522 [0] NCCL INFO ## [442510.054859] [00:00:00] 000000 KE busId d6000 nRanks 4096")
                        continue
                
                # A proxy print looks like this:
                # 0x7ff5ee771d48 [0-1|0| coll:3 comm:0x1b885440 [SEND] dtype:9 redOp:0 proto:2  nb:1048576 ns:16380 p:4436 t:4428 r:0, d:4428   myrank:6 peer:10 chan:1 tail:4428 recvtail:4428 reg:0 connSz:-1(retries:265446404)]
                if "recvtail" in lw and "myrank" in lw:
                    r = proxyPattern.parse(lw.strip())
                    if r is None:
                        # I have often seen a missing closing bracket in the proxy print, so we will try to fix that
                        if not lw.strip().endswith("]"):
                            towrite[il] = lw.strip() + "]" + "\n"
                        r = proxyPattern.parse(towrite[il].strip())
                    # If we still can't parse it, print an error
                    if r is None:
                        print_failure("0x7ff5ee771d48 [0-1|0| coll:3 comm:0x1b885440 [SEND] dtype:9 redOp:0 proto:2  nb:1048576 ns:16380 p:4436 t:4428 r:0, d:4428   myrank:6 peer:10 chan:1 tail:4428 recvtail:4428 reg:0 connSz:-1(retries:265446404)]")
                    continue
            file.writelines(towrite)
        return output_path
            
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
            print(f"\tCompleted Operation: {completed_op.op_type} Seq Num: {completed_op.seq_num} Duration: {completed_op.duration if completed_op.duration is not None else 'unknown'}")
        print(f"\tProxy Stalls: {comm.get_proxy_stall_count()}")


if __name__ == "__main__":
    main()