import sRTp.ncclTypes as ncclTypes
import json
import os
import subprocess

ncclDtypeDict = {
    "ncclInt8": 0,
    "ncclChar": 0,
    "ncclUint8": 1,
    "ncclInt32": 2,
    "ncclInt": 2,
    "ncclUint32": 3,
    "ncclInt64": 4,
    "ncclUint64": 5,
    "ncclFloat16": 6,
    "ncclHalf": 6,
    "ncclFloat32": 7,
    "ncclFloat": 7,
    "ncclFloat64": 8,
    "ncclDouble": 8,
    "ncclBfloat16": 9,
    } 


def process_profiler_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # We should only have one communicator per file, but sticking to this
    # format so we can have minimal diverging code patterns from ArmRankFiles
    communicators = []
    splitbase = os.path.basename(filename).split('_')
    nranks = int(splitbase[2])
    node_name = splitbase[3]
    global_rank = int(splitbase[4])
    comm_id = int(splitbase[1])
    local_rank = int(splitbase[5].split('.')[0])

    # How can I figure out size... can I defer it?

    # NB: From the profiler output we don't have any distinction between local and global comm IDs.
    comm = ncclTypes.localComm(nodeId=node_name, commId=comm_id, 
                               localId=None, localRank=local_rank,
                               size=nranks, busID=None,
                               cudaDev=None, nvmlDev=None,
                               nchannels=None)
    # Profiler output seems to offset sequence numbers by the group ID.. which is weird?
    seqNumBase = None
    lastOp = None
    for entry in data:
        if entry == {}:
            continue

        elif entry["cat"] == "COLL":
            if entry["ph"] == 'b':
                assert entry['args']['CommHash'] == comm_id, f"Communicator ID mismatch: {entry['args']['CommHash']} vs {comm_id}"
                if seqNumBase is None:
                    seqNumBase = entry['args']['SeqNum']
                seqNumBase = min(seqNumBase, entry['args']['SeqNum'])
                comm.start_operation(op_type=entry['name'],
                                     seq_num=hex(entry['args']['SeqNum'] - seqNumBase), # I made this only accepts hex strings.. that was dumb
                                     count=entry['args']['Count'],
                                     dtype=ncclDtypeDict[entry['args']['Datatype']],
                                     algorithm=entry['args']['Algorithm'],
                                     timestamp=entry['ts'])
                lastOp = comm.pending_operations[-1]
                comm.nchannels = entry['args']['nChannels']
            elif entry["ph"] == 'e':
                lastOp._end_time = entry['ts']
                if lastOp._start_time is not None:
                    lastOp.duration = (lastOp._end_time - lastOp._start_time) * 1000.0
                comm.complete_operation(lastOp)
                lastOp = None
    communicators.append(comm)
    return communicators, global_rank