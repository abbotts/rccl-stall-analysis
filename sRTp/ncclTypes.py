#! /usr/bin/env python3

import numpy as np


# This class describes a channel that we will follow for debugging.
# It needs to track information to identify the channel in output logs,
# such as the channel ID, the NCCL communicator it is associated with,
# whether it is run or key, and what it's monitoring.

# frontier00061:695135:695522 [0] NCCL INFO ## [442509.637772] [00:00:00] 000000 KL HWID 42302610 ncclDevFunc_ReduceScatter_RING_SIMPLE_Sum_bf16 nw 4 bi 0 nc 8 root 0 busId d6000 nRanks 4096
# Decoding: [00:00:00] - [rank on communicator:blockid:threadid] - in RCCL, blockId and threadId are the same [and will map directly to channels for this run, but not always]
# Look at channel informtion preceeding this line to see what the channel is doing
# frontier00061:695135:695521 [0] NCCL INFO Connected all rings
# Each channel represents one ring

class ProxyStall:
    """A class representing a proxy stall in the channel.
    
    This simple little class holds information about stalled communication in a channel across the fabric
    We'll associate these with specific operations, not with specific channels, so we can trace the communication
    in an operation that is stalled.
    """
    def __init__(self, channel_id: str, peer: int, direction: str, tail: int, recvtail: int, retries: int, collid: int, dtype: int, redop: int, protocol: int, nb: int, ns: int, p: int, t: int, r: int, d: int, contextstr: str):
        self.channel_id = channel_id  # The ID of the channel where the stall occurred
        self.peer = peer # The peer we were talking to
        self.direction = direction # send or receive
        self.tail = tail # Number of messages sent
        self.recvtail = recvtail # Number of messages received
        self.retries = retries # Times we've retried this operation
        self.collid = collid # What collective we're doing (some enum)
        self.dtype = dtype # Data type of the operation (some enum)
        self.redop = redop # Operation type (some enum)
        self.protocol = protocol # Protocol used (some enum)
        self.nb = nb # Maybe number of bytes (ask Arm or check source)
        self.ns = ns # Maybe Number of sends (ask Arm or check source)
        self.p = p # Maybe total packets (ask Arm or check source)
        self.t = t # Maybe number of transmitted packets (ask Arm or check source)
        self.r = r # No clue what this is, ask Arm or check source
        self.d = d # no clue what this is, ask Arm or check source
        # This tracks if we have traced this stall or not
        self.traced = False
        self.contextstr = contextstr  # A string to help us trace this stall in the logs, if needed
    def __repr__(self):
        return self.contextstr + f"Traced: {self.traced}"

class DTStall:
    # This represents a print from the kernel associated with a channel saying it is stalled
    """A class representing a data transfer stall in the channel.

    Attributes:
        channel_id (str): The ID of the channel where the stall occurred.
        timestamp (float): The timestamp when the stall was detected.
        content (str): The content of the stall message.
    """
    def __init__(self, channel_id: str, timestamp: float, content: str):
        self.channel_id = channel_id
        self.timestamp = timestamp
        self.content = content
        # For all the cases we have now, this should be identical to the channel_id
        self.block_number = channel_id.split(':')[1]  # Extract the block number from the channel ID
    def __repr__(self):
        return f"DTStall(channel_id={self.channel_id}, timestamp={self.timestamp}, content={self.content})"
    def get_block_number(self):
        """Get the block number from the channel ID."""
        return self.block_number

# A channel can look like this:
# frontier00061:695135:695311 [0] NCCL INFO Channel 00/0 : 0[d6000] -> 3[c6000] via P2P/IPC comm 0x9bbf3b0 nRanks 4096
# frontier00061:695135:695311 [0] NCCL INFO Channel 00/0 : 4092[d6000] -> 0[d6000] [receive] via NET/AWS Libfabric/2/GDRDMA comm 0x9bbf3b0 nRanks 4096

class Peer:
    def __init__(self, peer_id: int, peer_type: str, peer_direction: str):
        '''
        A class representing a peer in a channel.         
        peer_direction can be 'send', 'receive', or 'both'
        peer_type can be 'IPC', 'OFI', or other types as needed
        '''
        self.peer_id = peer_id
        self.peer_type = peer_type
        if peer_type not in ['IPC', 'OFI']:
            raise ValueError("peer_type must be 'IPC' or 'OFI'.")
        if peer_direction not in ['send', 'receive', 'both']:
            raise ValueError("peer_direction must be 'send', 'receive', or 'both'.")
        self.peer_direction = peer_direction
    def __repr__(self):
        return f"Peer(id={self.peer_id}, type={self.peer_type}, direction={self.peer_direction})"

class Channel:
    def __init__(self, channel_num: int, comm):
        self.channel_num = channel_num  # At this low level we should know the unique number each channel has, this will be the block number
        self.comm = comm  # The local NCCL communicator this channel is associated with
        self.peers = []  # List of peers in this channel
    def __repr__(self):
        return f"Channel(num={self.channel_num}, comm={self.comm.commId}, peers={self.peers})"
    def add_peer(self, peer_id: int, direction: str, peer_type: str):
        """Add a peer to the channel.
        
        Args:
            peer_id (int): The ID of the peer.
            direction (str): 'send', 'receive', or 'both'.
            peer_type (str): The type of connection (e.g., 'IPC', 'OFI').
        """
        self.peers.append(Peer(peer_id, peer_type, direction))



# This class describes an operation
# it needs to track the operation type, sequence number, start time, end time, and completion status
# it should also track the NCCL communicator it is associated with

class Operation:
    def __init__(self, op_type:str, seq_num:int, count:int, dtype:int, comm, algorithm="Ring"):
        self.op_type = op_type
        self.seq_num = seq_num
        self.count = count
        self.dtype = dtype
        self.status = None
        self.comm = comm
        self.progress_tid = None
        self.channels = []  # List of channels this operation is monitoring
        self.duration = None  # Duration of the operation, if applicable
        self._start_time = None  # Start time of the operation, if applicable
        self._end_time = None  # End time of the operation, if applicable
        self.expect_kernels = True  # Whether this will start kernels
        if self.comm.size == 1:
            self.expect_kernels = False
        self.algorithm = algorithm  # Algorithm used for the operation, default is "Ring"
        self.proxy_stalls = []  # List of proxy stalls associated with this operation instance
        self.dt_stalls = []  # List of data transfer stalls associated with this operation instance
    def __repr__(self):
        if self.duration is not None and self.duration > 0:
            duration_str = f"{self.duration} ms"
        else:
            duration_str = "unknown duration"
        return f"Operation(type={self.op_type}, seq_num={self.seq_num}, status={self.status}, duration={duration_str})"
    def complete(self):
        """Mark the operation as complete."""
        self.status = 'completed'
    def is_complete(self):
        """Check if the operation is complete."""
        return self.status == 'completed'
    def getOperationType(self):
        """Get the type of the operation."""
        return self.op_type
    def start_kernel(self, tid:int, channel_id:str, timestamp=None):
        """Log running kernels on running channels"""
        if self.progress_tid is None:
            self.progress_tid = tid
            self.status = 'in-progress'
        elif self.progress_tid != tid:
            raise ValueError("Operation already has a progress thread ID.")
        if channel_id in self.channels:
            #raise ValueError("Channel ID already in operation channels.")
            print(f"WARNING: Channel ID {channel_id} already in operation channels on TID {tid}")
            return
        self.channels.append(channel_id)
        if timestamp is not None:
            if self._start_time is None:
                self._start_time = float(timestamp)
            else:
                self._start_time = min(self._start_time, float(timestamp))
    def end_kernel(self, tid:int, channel_id:str, timestamp=None):
        """End kernels on channels"""
        if self.progress_tid != tid:
            raise ValueError("Operation progress thread ID does not match the ending thread ID.")
        if channel_id not in self.channels:
            raise ValueError("Channel ID not found in operation channels.")
        self.channels.remove(channel_id)
        if self._start_time is not None and timestamp is not None:
            if self._end_time is None:
                self._end_time = float(timestamp)
            else:
                self._end_time = max(self._end_time, float(timestamp))
        if self.channels == [] and self.expect_kernels:
            if self._start_time is not None and self._end_time is not None:
                self.duration = (self._end_time - self._start_time) * 1000.0
            # If no channels are left, mark the operation as complete
            self.complete()
    def add_proxy_stall(self, proxy_stall: ProxyStall):
        """Add a proxy stall to the operation.
        
        Args:
            proxy_stall (ProxyStall): The proxy stall to add.
        """
        self.proxy_stalls.append(proxy_stall)

    def add_dt_stall(self, dt_stall: DTStall):
        """Add a data transfer stall to the operation.
        
        Args:
            dt_stall (DTStall): The data transfer stall to add.
        """
        self.dt_stalls.append(dt_stall)

# This type describes a NCCL communicator object
# We should track the global commId, local rank, size, busID
# Channels, and ranks those channels are monitoring
# This should a member that is an ordered list of operations issued on to this group
# Or maybe the ordered list should be two ordered lists, one for issues operations and one for completed operations
# (How can I associate a NCCL communicator with pyTorch process group? Otherwise sequencenumber/opcount won't be useful)

class localComm:
    """A class representing a local NCCL communicator.
    
    Inputs to member functions are mostly strings for now, as this class is parsed out of log files.
    For classes downstream of this, this class will convert them to the appropriate types.

    Attributes:
        nodeId (str): The node ID where the communicator is located.
        commId (str): The global communicator ID, as a hex string.
        localId (str): The local communicator ID, as a hex string.
        localRank (str): The rank of the local communicator, as base 10 string.
        size (str): The size of the communicator, as a base 10 string.
        busID (str): The bus ID of the communicator, as a hex string.
        cudaDev (str): The CUDA device associated with the communicator, as a base 10 string.
        nvmlDev (str): The NVML device associated with the communicator, as a base 10 string.
        channels (list): A list of channels this communicator is monitoring. Currently just strings.
        pending_operations (list): A list of operations that are pending. Type is Operation.
        completed_operations (list): A list of operations that have been completed. Type is Operation.

        For now we assume 8 channels per communicator, but this may change in the future.
        nchannels (int): The number of channels this communicator has, default is 8.
    """
    def __init__(self, nodeId, commId, localId, localRank, size, busID, cudaDev, nvmlDev, nchannels=8):
        self.nodeId = nodeId
        self.commId = commId
        self.localId = localId
        self.localRank = int(localRank)
        self.size = int(size)
        self.busID = busID
        self.cudaDev = cudaDev
        self.nvmlDev = nvmlDev
        self.ring_channels = []  # List of channels this communicator uses
        self.rings_connected = False  # Whether all rings are connected
        self.tree_connected = False  # Whether all tree channels are connected
        self.tree_channels = []  # List of tree channels this communicator uses
        if self.size > 1:
            self.ring_channels = [ Channel(i, self) for i in range(nchannels) ]  # Preallocate 8 channels for ring operations
            self.tree_channels = [ Channel(i, self) for i in range(nchannels) ]  # Preallocate 8 channels for tree operations
        self.pending_operations = []  # List of operations that are pending
        self.completed_operations = []  # List of operations that have been completed
        self.proxy_stall_count = 0  # Count of proxy stalls associated with this communicator
        self.unmatched_proxies = []  # List of unmatched proxy stalls

    def add_peer_to_channel(self, channel_num: int, peer_id: int, direction: str, peer_type: str, algo: str = "Ring"):
        """Add a peer to a channel in this communicator.
        
        Args:
            channel_num (int): The channel number to add the peer to.
            peer_id (int): The ID of the peer.
            direction (str): 'send' or 'receive'.
            peer_type (str): The type of connection (e.g., 'IPC', 'OFI').
        """
        if channel_num < 0:
            raise ValueError("Channel number must be non-negative.")
        if algo == "Ring":
            if channel_num >= len(self.ring_channels):
                # If we're trying to add a channel above our length then we need to expand the list
                self.ring_channels.extend([Channel(i, self) for i in range(len(self.ring_channels), channel_num + 1)])
            self.ring_channels[channel_num].add_peer(peer_id, direction, peer_type)
        elif algo == "Tree":
            if channel_num >= len(self.tree_channels):
                # If we're trying to add a channel above our length then we need to expand the list
                self.tree_channels.extend([Channel(i, self) for i in range(len(self.tree_channels), channel_num + 1)])
            self.tree_channels[channel_num].add_peer(peer_id, direction, peer_type)
        else:
            raise ValueError("Unknown algorithm.")

    def finish_rings(self):
        """Mark all ring channels as connected."""
        self.rings_connected = True

    def finish_trees(self):
        """Mark all tree channels as connected."""
        self.tree_connected = True

    def start_operation(self, op_type, seq_num, count, dtype):
        """Start an operation on this communicator."""
        # This could be a more complex structure to track operations
        self.pending_operations.append(Operation(op_type, int(seq_num, 16), int(count), int(dtype), self))
        if self.size == 1:
            # If the communicator size is 1, we can immediately complete the operation
            self.complete_operation_by_match(op_type, seq_num)
        for ip, unmatched_proxy in enumerate(self.unmatched_proxies):
            if unmatched_proxy[0] == seq_num:
                self.pending_operations[-1].add_proxy_stall(unmatched_proxy[1])
                self.unmatched_proxies.pop(ip)
            

    # For now a channel ID is a string, but once Arm can remind me what they mean, I can make this a more complex type
    def start_kernel_if_match(self, opType:str, tid:str, channel_id:str, timestamp=None):
        """If the kernel matches an operation on this communicator, start it and return true.
        Otherwise, return false.
        Inputs to this function are all strings for now, and type conversions will be done later.
        """
        kstarted = False
        for operation in self.pending_operations:
            # This assumes we will not have multiple pending operations of the same type on the same communicator
            # If we do, we will need to track the sequence number or some other identifier
            # Disabling this check. There's a fragility here now, if multiple operations are running on the same communicator at the
            # same time. But that shouldn't be possible...
            #if kstarted:
            #    raise ValueError("Multiple pending operations of the same type on the same communicator.")
            # Here I assume that there will be only one thread watching progress on a given operation
            # If there are multiple threads this will raise an exception
            if operation.getOperationType() == opType:
                operation.start_kernel(int(tid), channel_id, timestamp)
                kstarted = True
                break
            else:
                print(f"Operation {operation.getOperationType()} does not match {opType}, skipping.")
        return kstarted

    def end_kernel_if_match(self, tid, channel_id, opType:str, timestamp=None):
        """If the kernel matches an operation on this communicator, end it and return true.
        Otherwise, return false."""
        kend = False
        for operation in self.pending_operations:
            if kend:
                #raise ValueError("Multiple pending operations using the same tid and channel_id on the same communicator.")
                break
            if operation.progress_tid == int(tid) and channel_id in operation.channels and opType == operation.getOperationType():
                operation.end_kernel(int(tid), channel_id, timestamp)
                kend = True
            if operation.is_complete():
                self.complete_operation(operation)
        return kend

    def complete_operation_by_match(self, op_type, seq_num):
        """Complete an operation on this communicator."""
        # This could be a more complex structure to track operations
        for operation in self.pending_operations:
            if operation.op_type == op_type and operation.seq_num == int(seq_num, base=16):
                self.completed_operations.append(operation)
                self.pending_operations.remove(operation)

    def complete_operation(self, operation: Operation):
        """Complete an operation on this communicator."""
        if operation in self.pending_operations:
            self.completed_operations.append(operation)
            self.pending_operations.remove(operation)
        else:
            raise ValueError("Operation not found in pending operations.")

    def get_operation(self, seq_num: int) -> Operation:
        """Get an operation by its sequence number."""
        for operation in self.pending_operations + self.completed_operations:
            if operation.seq_num == seq_num:
                return operation
        raise IndexError(f"Operation with sequence number {seq_num} not found.")
    
    def add_proxy_print(self, opcountx2:int, peer: int, channel: int, direction: str, tail: int, recvtail: int, retries: int, collid: int, dtype: int, redop: int, proto: int, nb: int, ns: int, p: int, t: int, r: int, d: int, content: str) -> None:
        """Add a proxy print to the communicator.
        
        This is used to track stalls in communication across the fabric.
        """
        # I would like this to be true but without the kernel logging we may not be able to move operations to completed, so
        # let's just add prints on the last pending operation at this moment.
        #if len(self.pending_operations) != 1:
        #    raise ValueError("There should be exactly one pending operation to add a proxy print.")
        found = False
        for op in self.pending_operations:
            if op.seq_num == opcountx2/2: 
                op.add_proxy_stall(ProxyStall(channel, peer, direction, tail, recvtail, retries, collid, dtype, redop, proto, nb, ns, p, t, r, d, content))
                found = True
                break
        for op in self.completed_operations and not found:
            if op.seq_num == opcountx2/2: 
                op.add_proxy_stall(ProxyStall(channel, peer, direction, tail, recvtail, retries, collid, dtype, redop, proto, nb, ns, p, t, r, d, content))
                found = True
                break
        
        if not found:
            self.unmatched_proxies.append((opcountx2/2, ProxyStall(channel, peer, direction, tail, recvtail, retries, collid, dtype, redop, proto, nb, ns, p, t, r, d, content)))
        self.proxy_stall_count += 1
    
    def get_proxy_stall_count(self) -> int:
        """Get the count of proxy stalls associated with this communicator."""
        return self.proxy_stall_count

    def add_dt_stall(self, channel_id:str, timestamp:float, content:str) -> None:
        """Add a data transfer stall to the communicator."""
        if len(self.pending_operations) != 1:
            raise ValueError("There should be exactly one pending operation to add a data transfer stall.")
        self.pending_operations[0].add_dt_stall(DTStall(channel_id, timestamp, content))

    def get_operations(self) -> list:
        """Get a list of all operations (both pending and completed) associated with this communicator."""
        return self.pending_operations + self.completed_operations

    def check_sequence_consistency(self) -> bool:
        """ Check if the sequence number of an operation is the same as its place in the op array."""
        for i, op in enumerate(self.get_operations()):
            if op.seq_num != i:
                return False
        return True

# We need a class that represents a communicator across the entire system
# This class should track the individual local communicators and the mapping between the global rank
# and their local rank in the communicator.
# It should be able to match completed and pending operations across all the communicators.
class globalComm:
    """A class representing a global NCCL communicator.
    
    This class is a container for multiple local communicators and provides methods to manage operations across them.
    """
    def __init__(self, commId: str, size: int):
        self.commId = commId
        # A list of local communicators preallocated to size
        self.local_communicators = [None] * size  # Ordered list of localComm objects
        # I suspect we'll go back and forth between local and global ranks a lot, so let's keep
        # a map for both directions
        self.local_to_global_rank_map = {}  # Maps local rank to global rank
        self.global_to_local_rank_map = {}  # Maps global rank to local rank
        self.size = size  # Size of the communicator

    def add_local_communicator(self, local_comm: localComm, global_rank: int):
        """Add a local communicator to the global communicator."""
        self.local_communicators[local_comm.localRank] = local_comm
        # Update the global rank map
        self.local_to_global_rank_map[local_comm.localRank] = global_rank
        self.global_to_local_rank_map[global_rank] = local_comm.localRank

    def get_local_communicator(self, local_rank: int) -> localComm:
        """Get the local communicator for a given local rank."""
        return self.local_communicators[local_rank]
    
    def comm_record_complete(self) -> bool:
        """Returns True if there is a local comm object for every rank in the communicator.
        
        If this is False then either a communicator is missing from a log or it was called too soon.
        """
        return None not in self.local_communicators

    def get_pending_opcounts(self) -> np.ndarray:
        """Get the  number of pending operations across all local communicators as a numpy array."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        return np.array([len(comm.pending_operations) for comm in self.local_communicators])

    def get_completed_opcounts(self) -> np.ndarray:
        """Get the number of completed operations across all local communicators as a numpy array."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        return np.array([len(comm.completed_operations) for comm in self.local_communicators])
    
    def get_opcounts(self) -> np.ndarray:
        """Get the number of pending and completed operations across all local communicators as a numpy array."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        pending_counts = self.get_pending_opcounts()
        completed_counts = self.get_completed_opcounts()
        return pending_counts + completed_counts
    
    def same_completed_opcounts(self) -> bool:
        """Check if all local communicators have the same number of completed operations."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        completed_counts = self.get_completed_opcounts()
        return np.all(completed_counts == completed_counts[0])

    def same_opcounts(self) -> bool:
        """Check if all local communicators have the same number of pending and completed operations."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        opcounts = self.get_opcounts()
        return np.all(opcounts == opcounts[0])

    def get_completed_durations(self, fillMissing: bool = False) -> np.ndarray:
        """Get the durations of completed operations across all local communicators as a numpy array.
        Time units are in milliseconds.
        
        If fillMissing is True, it will fill missing durations with NaN.
        """

        if not fillMissing and not self.same_completed_opcounts():
            raise ValueError("Not all local communicators have the same number of completed operations. Try fillMissing=True.")

        rval = np.empty((len(self.local_communicators), max(self.get_completed_opcounts())))
        rval.fill(np.nan)  # Fill with NaN for missing durations

        for i, comm in enumerate(self.local_communicators):
            for j, op in enumerate(comm.completed_operations):
                if op.duration is not None:
                    rval[i, j] = op.duration
        
        return rval
    
    def get_completed_operations(self) -> list:
        """Get a list of all completed operations across all local communicators."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        all_completed_ops = [ [ op.op_type for op in comm.completed_operations] for comm in self.local_communicators]
        all_equal = True
        for oplist in all_completed_ops:
            if oplist != all_completed_ops[0]:
                all_equal = False
                break
        if not all_equal:
            raise ValueError("Not all local communicators have the same completed operations.")

        return [ op for op in self.local_communicators[0].completed_operations ]

    def get_pending_operations(self) -> list:
        """Get a list of all pending operations across all local communicators."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        all_pending_ops = [ [ op.op_type for op in comm.pending_operations] for comm in self.local_communicators]
        all_equal = True
        for oplist in all_pending_ops:
            if oplist != all_pending_ops[0]:
                all_equal = False
                break
        if not all_equal:
            raise ValueError("Not all local communicators have the same pending operations.")

        return [ op for op in self.local_communicators[0].pending_operations ]
    
    def get_operations(self, allowUneven: bool = False) -> list:
        """Get a list of all operations (pending and completed) across all local communicators."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        all_ops = [ comm.completed_operations + comm.pending_operations for comm in self.local_communicators ]
        all_ops_types = [ [ op.getOperationType() for op in ops ] for ops in all_ops ]
        all_equal = True
        for oplist in all_ops_types:
            if oplist != all_ops_types[0]:
                all_equal = False
                break
        if not all_equal:
            if allowUneven:
                print("Warning: Not all local communicators have the same operations. Returning operations from the first communicator.")
            else:
                raise ValueError("Not all local communicators have the same operations.")

        return [ op for op in all_ops[0] ]

    def check_consistency(self) -> bool:
        """Check if all local communicators have the same pending and completed operations, and that the completed operations have the same sequence numbers, counts, and dtypes."""
        consistent = True
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        total_counts = self.get_opcounts()
        if not np.all(total_counts == total_counts[0]):
            consistent = False
            print("Operation counts are not consistent across local communicators.")
            unique, counts = np.unique(total_counts, return_counts=True)
            for i, unique in enumerate(unique):
                print(f"Operation count {unique} appears {counts[i]} times.")
                if counts[i] < 10:
                    print(f"\tLocal ranks: {np.argwhere(total_counts == unique).flatten()}")
                    print(f"\tGlobal ranks: {[self.local_to_global_rank_map[r] for r in np.argwhere(total_counts == unique).flatten()]}")

        pending_counts = self.get_pending_opcounts()
        if not np.all(pending_counts == pending_counts[0]):
            consistent = False
            print("Pending operation counts are not consistent across local communicators.")
            unique, counts, = np.unique(pending_counts, return_counts=True)
            for i, unique in enumerate(unique):
                print(f"Pending operation count {unique} appears {counts[i]} times.")
                if counts[i] < 10:
                    print(f"\tLocal ranks: {np.argwhere(pending_counts == unique).flatten()}")
                    print(f"\tGlobal ranks: {[self.local_to_global_rank_map[r] for r in np.argwhere(pending_counts == unique).flatten()]}")

        if not self.same_completed_opcounts():
            consistent = False
            print("Completed operation counts are not consistent across local communicators.")
            unique, counts = np.unique(self.get_completed_opcounts(), return_counts=True)
            for i, unique in enumerate(unique):
                print(f"Completed operation count {unique} appears {counts[i]} times.")
                if counts[i] < 10:
                    print(f"\tLocal ranks: {np.argwhere(self.get_completed_opcounts() == unique).flatten()}")
                    print(f"\tGlobal ranks: {[self.local_to_global_rank_map[r] for r in np.argwhere(self.get_completed_opcounts() == unique).flatten()]}")

        for comm in self.local_communicators:
            if not comm.check_sequence_consistency():
                consistent = False
            for i, op in enumerate(comm.completed_operations):
                if i >= len(self.local_communicators[0].completed_operations):
                    consistent = False
                    print(f"Completed operation index {i} out of range in communicator {comm.commId} at rank {comm.localRank}.")
                    continue
                if op.seq_num != self.local_communicators[0].completed_operations[i].seq_num:
                    consistent = False
                    print(f"Completed operation sequence number mismatch in communicator {comm.commId} at rank {comm.localRan}.")
                if op.count != self.local_communicators[0].completed_operations[i].count:
                    consistent = False
                    print(f"Completed operation count mismatch in communicator {comm.commId} at rank {comm.localRan}.")
                if op.dtype != self.local_communicators[0].completed_operations[i].dtype:
                    consistent = False
                    print(f"Completed operation dtype mismatch in communicator {comm.commId} at rank {comm.localRan}.")
            for i, op in enumerate(comm.pending_operations):
                if i >= len(self.local_communicators[0].pending_operations):
                    consistent = False
                    print(f"Pending operation index {i} out of range in communicator {comm.commId} at rank {comm.localRank}.")
                    continue
                if op.seq_num != self.local_communicators[0].pending_operations[i].seq_num:
                    consistent = False
                    print(f"Pending operation sequence number mismatch in communicator {comm.commId} at rank {comm.localRank}.")
                if op.count != self.local_communicators[0].pending_operations[i].count:
                    consistent = False
                    print(f"Pending operation count mismatch in communicator {comm.commId} at rank {comm.localRank}.")
                if op.dtype != self.local_communicators[0].pending_operations[i].dtype:
                    consistent = False
                    print(f"Pending operation dtype mismatch in communicator {comm.commId} at rank {comm.localRank}.")
        return consistent
    
    def get_proxy_stall_counts_on_completed_operations(self) -> np.ndarray:
        """Get the total count of proxy stalls across all local communicators on completed operations."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        rval = np.zeros((len(self.local_communicators), max(self.get_completed_opcounts())))

        for i, comm in enumerate(self.local_communicators):
            for j, op in enumerate(comm.completed_operations):
                rval[i, j] = len(op.proxy_stalls)
        return rval

    def get_proxy_stall_counts_on_operations(self) -> np.ndarray:
        """Get the total count of proxy stalls across all local communicators on operations."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        rval = np.zeros((len(self.local_communicators), max(self.get_opcounts())))

        for i, comm in enumerate(self.local_communicators):
            for j, op in enumerate(comm.completed_operations + comm.pending_operations):
                rval[i, j] = len(op.proxy_stalls)
        return rval

    def get_proxy_stall_counts(self) -> np.ndarray:
        """Get the total count of proxy stalls across all local communicators."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        rval = np.zeros(len(self.local_communicators))

        for i, comm in enumerate(self.local_communicators):
            rval[i] = comm.get_proxy_stall_count()
        return rval
    

    def trace_proxy_stalls(self, algo: str = "Ring", allowUneven: bool = False) -> None:
        """Trace proxy stalls across all local communicators."""
        if allowUneven:
            print("Warning: Tracing proxy stalls with allowUneven=True is a bit unpredictable and may miss stalls.")
        uncounted_stalls = self.get_proxy_stall_counts_on_operations()
        if(uncounted_stalls.sum() == 0):
            print("No proxy stalls to trace.")
            return
        all_ops = self.get_operations(allowUneven=allowUneven)

        last_uncounted_stalls = 0
        while uncounted_stalls.sum() > 0:
            if last_uncounted_stalls == uncounted_stalls.sum():
                print("No progress made in tracing proxy stalls, exiting to avoid infinite loop.")
                break
            last_uncounted_stalls = uncounted_stalls.sum()
            print(f"Uncounted stalls remaining: {uncounted_stalls.sum()}")
            # Find the first uncounted stall
            #print(np.argwhere(uncounted_stalls))
            starting_rank, op_index = np.argwhere(uncounted_stalls)[0]
            #print(uncounted_stalls)
            print(f"Tracing proxy stall in communicator {self.local_communicators[starting_rank].commId} at local rank {starting_rank}, operation index {op_index}.")
            

            current_rank = None
            last_rank = None # Track this to make sure we go in the right direction for IPC
            last_step = None # Track if our last step was an IPC or OFI step
            last_stall = None # Track the last stall we found, so we can match info on the other side
            tracing_channel = None  # This will hold the channel we are tracing through
            # if we loop back around, we need to stop
            searching = True
            steps = 0 # Track the number of steps we have taken
            while searching:
                #print(f"Searched {steps} / {self.size} ranks for stalls.", end='\r')
                # Python doesn't have a do-while loop, so we need to use a while loop with a break
                # that will terminate on this condition *after* doing the proxy search
                if current_rank == starting_rank:
                    searching = False

                # We should expect to search the proxies unless we both come in on an IPC step
                search_proxy = True
                # This will hold the stalls we are searching for on this step
                unique_stalls = []

               # For first iteration we don't need to search for a channel and skip straight to stall searching
                if current_rank is None:
                    current_rank = starting_rank
                    for stall in reversed(self.local_communicators[current_rank].get_operations()[op_index].proxy_stalls):
                        if stall.traced is False:
                            unique_stalls.append(stall)
                            break
                    last_rank = current_rank
                    tracing_channel = unique_stalls[0].channel_id

                    if stall.direction == 'RECV':
                        # Everything is setup to follow the send direction and process recieves when we
                        # arrive on a rank, so if we are a recieve then we should just immediately move on
                        # to the send direction and our stall will be traced when we arrive back here.
                        for peer in self.local_communicators[last_rank].ring_channels[tracing_channel].peers:
                            if peer.peer_id != stall.peer:
                                last_step = peer.peer_type
                                current_rank = peer.peer_id
                                unique_stalls.remove(stall)
                                if last_step == 'IPC':
                                    search_proxy = False
                                break
                    else:
                        last_stall = unique_stalls[0]
                        current_rank = unique_stalls[0].peer
                        last_step = 'OFI'

                    print(f"Stall on channel {tracing_channel} starting on rank {last_rank} (Global {self.local_to_global_rank_map[last_rank]}).")
                else:
                    # If last_step was IPC, we won't search proxy output unless our next step is OFI
                    # So turn the search off for now, and if we find a channel out over OFI we'll check the proxy
                    if last_step == 'IPC':
                        search_proxy = False
                    else:
                        # If we came in off OFI, we should see if we have a stall from it
                        try:
                            for stall in reversed(self.local_communicators[current_rank].get_operations()[op_index].proxy_stalls):
                                if stall.peer == last_rank and stall.traced is False and stall.channel_id == tracing_channel:
                                    unique_stalls.append(stall)
                                    #assert stall.channel_id == tracing_channel, f"Stall channel ID {stall.channel_id} does not match tracing channel {tracing_channel} on rank {current_rank}."
                                    if last_stall is None:
                                        print("**********************************")
                                        print(f"Rank {current_rank} (Global {self.local_to_global_rank_map[current_rank]}) found stall on channel {stall.channel_id} from rank {last_rank} (Global {self.local_to_global_rank_map[last_rank]}), but there was no stall on the sender side to match.")
                                        print(f"Global communicator {self.commId} operation {all_ops[op_index].op_type} seq_num {all_ops[op_index].seq_num}.")
                                        for lproxy in self.local_communicators[last_rank].get_operation(op_index).proxy_stalls:
                                            print(lproxy)
                                        continue
                                    if stall.recvtail != last_stall.recvtail:
                                        print(f"Mismatch in recvtail proxy output from rank {last_rank} (Global {self.local_to_global_rank_map[last_rank]}) on rank {current_rank} (Global {self.local_to_global_rank_map[current_rank]}).")
                                        print(f"Global communicator {self.commId} operation {all_ops[op_index].op_type} seq_num {all_ops[op_index].seq_num}.")
                                        #print(f"Rank {current_rank} Tail: {stall.tail}, Recvtail: {stall.recvtail}, Retries: {stall.retries}, Collid: {stall.collid}, Dtype: {stall.dtype}, Redop: {stall.redop}, Protocol: {stall.protocol}, Nb: {stall.nb}, Ns: {stall.ns}, P: {stall.p}, T: {stall.t}, R: {stall.r}, D: {stall.d}")
                                        #print(f"Rank {last_rank} Tail: {last_stall.tail}, Recvtail: {last_stall.recvtail}, Retries: {last_stall.retries}, Collid: {last_stall.collid}, Dtype: {last_stall.dtype}, Redop: {last_stall.redop}, Protocol: {last_stall.protocol}, Nb: {last_stall.nb}, Ns: {last_stall.ns}, P: {last_stall.p}, T: {last_stall.t}, R: {last_stall.r}, D: {last_stall.d}")
                                        print(last_stall)
                                        print(stall)
                                    break
                        except IndexError:
                            search_proxy = False
                            print(f"!!!!WARNING!!!:Operation with index {op_index} does not exist on rank {current_rank} (Global {self.local_to_global_rank_map[current_rank]}).")
                            print(f"This explains the stall from rank {last_rank} (Global {self.local_to_global_rank_map[last_rank]}) to rank {current_rank} (Global {self.local_to_global_rank_map[current_rank]}).")
                            print(last_stall)
                            print(f"Skipping and moving along channel.")

                    channel_list = self.local_communicators[current_rank].ring_channels if algo == "Ring" else self.local_communicators[current_rank].tree_channels

                    # This is a bunch of logic for searching channels based on peers
                    # I don't think we need this, and we're going to try just tracing the channel instead
                    if False:
                        # Search channels to find the one we came in on
                        # We need to look for a channel that has two peers, otherwise it can't be traced
                        for channel in channel_list:
                            if len(channel.peers) < 2:
                                continue
                            for peer in channel.peers:
                                if peer.peer_id == last_rank and peer.peer_type == last_step:
                                    # We found the peer we came from, so now pick our channel
                                    local_channel = channel
                                    break
                            if local_channel is not None:
                                break
                        if local_channel is None:
                            raise ValueError(f"Could not find channel on rank {current_rank} for {last_step} step from rank {last_rank} of type {last_step}.")
                        for stall in unique_stalls:
                            # If we had a proxy stall matching the last peer, its channel ID should match the one we found by searching channels for the peer
                            if stall.channel_id != local_channel.channel_num:
                                raise ValueError(f"Stall channel ID {stall.channel_id} does not match local channel {local_channel.channel_num} on rank {current_rank}.")
                    
                    # This logic to figure out where we're going is valid even if we're using a fixed channel
                    local_channel = channel_list[tracing_channel]              
                    for peer in local_channel.peers:
                        #print(local_channel)

                        # As long as we're tracing in the send direction, the only time
                        # the rank we came from should be logged as a peer is if it's an OFI receive peer
                        if peer.peer_id == last_rank:
                            assert peer.peer_type == "OFI" and peer.peer_direction == 'receive', f"Peer {peer.peer_id} on rank {current_rank} is not an OFI receive peer, but is the last rank we came from."
                            continue

                        # If the next step on our channel is an IPC step, we should just take it and leave the proxy output
                        # for a different ring
                        if peer.peer_type == 'IPC':
                            last_rank = current_rank
                            current_rank = peer.peer_id
                            last_step = 'IPC'
                            last_stall = None
                            break
                        elif peer.peer_type == 'OFI':
                            # If the next step is an OFI step, we should check if we have a stall to trace
                            search_proxy = True
                            last_rank = current_rank
                            last_step = 'OFI'
                            current_rank = peer.peer_id
                            last_stall = None
                            try:
                                for stall in reversed(self.local_communicators[last_rank].get_operation(op_index).proxy_stalls):
                                    if stall.peer == current_rank and stall.traced is False and stall.channel_id == tracing_channel:
                                        unique_stalls.append(stall)
                                        last_stall = stall
                                        break
                            except IndexError:
                                # We don't have this operation
                                search_proxy = False
                                last_stall = ProxyStall(
                                    channel_id=tracing_channel,
                                    peer=current_rank,
                                    direction='send',
                                    tail=0,
                                    recvtail=0,
                                    retries=0,
                                    collid=0,
                                    dtype=0,
                                    redop=0,
                                    protocol=0,
                                    nb=0,
                                    ns=0,
                                    p=0,
                                    t=0,
                                    r=0,
                                    d=0,
                                    contextstr="Fake proxy stall! No operation found for this rank!"
                                )
                                pass
                            break
                
                #print(f"Stepping to rank {current_rank} (Global {self.local_to_global_rank_map[current_rank]}) from rank {last_rank} (Global {self.local_to_global_rank_map[last_rank]}).")
                if not search_proxy:
                    continue

                # At this point, "last_rank" is the rank we're currently on, and "current_rank" is the rank we're going to next
                for stall in self.local_communicators[last_rank].get_operation(op_index).proxy_stalls:
                    if stall.traced:
                        continue
                    for ustall in unique_stalls:
                        if stall.peer == ustall.peer and stall.direction == ustall.direction and stall.channel_id == ustall.channel_id:
                            # This stall is redundant. Collapse into the last stall
                            stall.traced = True
                            uncounted_stalls[last_rank, op_index] -= 1
                            if stall.tail != ustall.tail:
                                print(f"Rank {last_rank} tail changed from {ustall.tail} to {stall.tail}. It's not stalled, just slow.")
                            break