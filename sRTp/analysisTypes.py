#! /usr/bin/env python3

import numpy as np

# This class descripts an operational
# it needs to track the operation type, sequence number, start time, end time, and completion status
# it should also track the NCCL communicator it is associated with

class Operation:
    def __init__(self, op_type:str, seq_num:int, comm):
        self.op_type = op_type
        self.seq_num = seq_num
        self.status = None
        self.comm = comm
        self.progress_tid = None
        self.channels = []  # List of channels this operation is monitoring
        self.duration = None  # Duration of the operation, if applicable
        self._start_time = None  # Start time of the operation, if applicable
    def __repr__(self):
        if self.duration > 0:
            duration_str = f"{self.duration * 1000} ms"
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
            raise ValueError("Channel ID already in operation channels.")
        self.channels.append(channel_id)
        if self._start_time is None and timestamp is not None:
            self._start_time = float(timestamp)
    def end_kernel(self, tid:int, channel_id:str, timestamp=None):
        """End kernels on channels"""
        if self.progress_tid != tid:
            raise ValueError("Operation progress thread ID does not match the ending thread ID.")
        if channel_id not in self.channels:
            raise ValueError("Channel ID not found in operation channels.")
        self.channels.remove(channel_id)
        if self.channels == []:
            if self._start_time is not None and timestamp is not None:
                self.duration = float(timestamp) - self._start_time
            # If no channels are left, mark the operation as complete
            self.complete()
        

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
    """
    def __init__(self, nodeId, commId, localId, localRank, size, busID, cudaDev, nvmlDev):
        self.nodeId = nodeId
        self.commId = commId
        self.localId = localId
        self.localRank = int(localRank)
        self.size = int(size)
        self.busID = busID
        self.cudaDev = cudaDev
        self.nvmlDev = nvmlDev
        self.channels = []  # List of channels this communicator is monitoring
        self.pending_operations = []  # List of operations that are pending
        self.completed_operations = []  # List of operations that have been completed

    # Right now a channel is just a string, but it should become a type in the future
    # I just don't know what the strings actually mean (need to bug )
    def add_channel(self, channel: str):
        """Add a channel to the communicator."""
        self.channels.append(channel)

    def start_operation(self, op_type, seq_num):
        """Start an operation on this communicator."""
        # This could be a more complex structure to track operations
        self.pending_operations.append(Operation(op_type, int(seq_num, 16), self))
        if self.size == 1:
            # If the communicator size is 1, we can immediately complete the operation
            self.complete_operation_by_match(op_type, seq_num)

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
            if kstarted:
                raise ValueError("Multiple pending operations of the same type on the same communicator.")
            # Here I assume that there will be only one thread watching progress on a given operation
            # If there are multiple threads this will raise an exception
            if operation.getOperationType() == opType:
                operation.start_kernel(int(tid), channel_id, timestamp)
                kstarted = True
            else:
                print(f"Operation {operation.getOperationType()} does not match {opType}, skipping.")
        return kstarted

    def end_kernel_if_match(self, tid, channel_id, timestamp=None):
        """If the kernel matches an operation on this communicator, end it and return true.
        Otherwise, return false."""
        kend = False
        for operation in self.pending_operations:
            if kend:
                raise ValueError("Multiple pending operations using the same tid and channel_id on the same communicator.")
            if operation.progress_tid == int(tid) and channel_id in operation.channels:
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

    def add_local_communicator(self, local_comm: localComm, global_rank: int):
        """Add a local communicator to the global communicator."""
        self.local_communicators[local_comm.localId] = local_comm
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
    
    def same_completed_opcounts(self) -> bool:
        """Check if all local communicators have the same number of completed operations."""
        if not self.comm_record_complete():
            raise ValueError("Communicator is not fully populated with local communicators.")
        completed_counts = self.get_completed_opcounts()
        return np.all(completed_counts == completed_counts[0])

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
                    rval[i, j] = op.duration * 1000