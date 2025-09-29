# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# Load the pickled with all the processed data
global_comms = pickle.load(open("f1024_run3_sockets_size8192.pickle", "rb"))

# %%
# There should only be one communicator here
print(f"Found {len(global_comms)} communicators")

# %%
comms_by_len = {}
for comm_id, comm in global_comms.items():
    if comm.size not in comms_by_len:
        comms_by_len[comm.size] = []
    comms_by_len[comm.size].append(comm_id)

# %%
comms_by_len.keys()

# %%
#global_comms
# There should only be one communiator here
assert len(comms_by_len[8192]) == 1
comhash = comms_by_len[8192][0]

# %%
# Get start and stop times for each operation in each local communicator
starts = np.array([ [ op._start_time for op in comm.completed_operations ] for comm in global_comms[comhash].local_communicators ])
stops = np.array([ [ op._end_time for op in comm.completed_operations ] for comm in global_comms[comhash].local_communicators ])
#durations = np.array([ [ op._end_time - op._start_time for op in comm.completed_operations ] for comm in global_comms[comhash].local_communicators ])

# %%
# Normalize times to start at zero and convert to milliseconds
tstart = min(starts.flatten())
starts = starts - tstart
stops = stops - tstart
starts = starts * 1e3
stops = stops * 1e3
durations = stops - starts

# %%
assert starts.shape == stops.shape

# %%
#np.max(durations, axis=0)
# get an index array of where durations > 0.1 ms
nontrivial = durations > 0.1

# %%
# Make the boxplot that is a decent first pass at visualizing the data
fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(durations)
ax.set_xlabel("Operation Number")
ax.set_ylabel("Duration (ms)")
ax.set_title("Boxplot of Operation Durations for Reduce-Scatter Size 8192")
#ax.boxplot(stops)

# %% [markdown]
# # This histogram isn't very useful .. skip it for now
# ````
# fig, ax = plt.subplots(figsize=(12, 6))
# threshold = 10
# indexes = durations > threshold
# print(indexes.shape)
# counts, edges, bars = ax.hist(durations, bins=50, range=(10,durations.max()), log=True)
# ax.set_xlabel("Count")
# ax.set_ylabel("Duration (ms)")
# ax.set_title("Histogram of Operation Durations in Opnumber 28 for Reduce-Scatter Size 8192")
# #ax.bar_label(bars, fmt="%.0f")
# #ax.boxplot(stops)
# ````

# %%
max_op = np.unravel_index(np.argmax(durations), durations.shape)[1]

# %%
counts, edges = np.histogram(durations[:, max_op], bins=50)
duration_histo = np.zeros((durations.shape[1], 50), dtype=int)
for row in range(durations.shape[1]):
    duration_histo[row, :], _ = np.histogram(durations[:, row], bins=edges)


# %%
fig = plt.figure( figsize=(20, 20))
ax = fig.add_subplot(111)

p = ax.pcolormesh(np.arange(durations.shape[1]), edges[:-1], duration_histo.T, norm='log')
ax.set_xlabel("Operation Number")
ax.set_ylabel("Duration (ms)")
ax.set_title("Distribution of Operation Durations for Reduce-Scatter Size 8192")
plt.colorbar(p)

# %%
#truestarts = np.min(starts, axis=0)
#trueends = np.max(stops, axis=0)



# %%
max_op

# %%
# This was Mark's suggestion to sort ranks by their duration during the longest op. It's an intersting plot, but I think less useful than just plotting the start and stop times of each op for each rank, which I do next.
import seaborn as sns
colors = sns.color_palette("colorblind")

sorted_by_maxop_start = np.argsort(starts[:, max_op])
nonzero_index = durations[sorted_by_maxop_start, :] > 10
print(nonzero_index.shape)

fig, ax = plt.subplots(figsize=(120, 60))
for op in range(durations.shape[1]):
    ax.plot(starts[sorted_by_maxop_start, op][nonzero_index[:, op]], ls='-', color=colors[op % len(colors)])
    ax.plot(stops[sorted_by_maxop_start, op][nonzero_index[:, op]], ls='--', color=colors[op % len(colors)])

ax.set_yticks(np.arange(0,np.max(stops), 24000))
plt.grid()
ax.set_xlabel("Rank (sorted by start time of longest op)")
ax.set_ylabel("Time (ms)")

# %%


# %%
#import seaborn as sns
#colors = sns.color_palette("bright", n_colors=10)
# Plot start and stop times for each op for each rank, unsorted
nonzero_index = durations > 10
print(nonzero_index.shape)

fig, ax = plt.subplots(figsize=(120, 60))
for op in range(durations.shape[1]):
    ax.plot(np.arange(nonzero_index.shape[0])[nonzero_index[:,op]], starts[nonzero_index[:, op], op], ls='-', color=colors[op % len(colors)])
    ax.plot(np.arange(nonzero_index.shape[0])[nonzero_index[:,op]], stops[nonzero_index[:, op], op], ls='--', color=colors[op % len(colors)])
    max_rank = np.argmax(durations[:, op])
    ax.scatter(max_rank, starts[max_rank, op], s=500, color='red', marker='o', edgecolor='black')

ax.set_yticks(np.arange(0,np.max(stops), 25000))
plt.grid()
ax.set_xlabel("Rank (unsorted)")
ax.set_ylabel("Time (ms)")

# %%
# Save at super high resolution
fig.savefig("rccl_timeline_unsorted_gridlines_run3_sockets.pdf", dpi=300)

# %% [markdown]
# * STOP HERE *
# 
# Everything past this point is dinking around

exit(0)
# %%
maxranks = np.argmax(durations, axis=0)
for opnum, rank in enumerate(maxranks):
    print(f"Operation {opnum} max duration on rank {rank}: {durations[rank, opnum]:.2f} ms (start {global_comms[comhash].local_communicators[rank].completed_operations[opnum]._start_time:.2f} s, stop {global_comms[comhash].local_communicators[rank].completed_operations[opnum]._end_time:.2f} s)")

# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(starts)
ax.set_xlabel("Operation Number")
ax.set_ylabel("Start Time (ms)")
ax.set_title("Boxplot of Operation Start Times for Reduce-Scatter Size 8192")
plt.savefig("reduce_scatter_starts_size8192.pdf")
#ax.boxplot(stops)

# %%
truestarts.shape, trueends.shape

# %%
overlaps = trueends[:-1] - truestarts[1:]

# %%


# %%
plt.bar(np.arange(overlaps.shape[0]) + 1, overlaps)
plt.xlabel("Operation Number N")
plt.ylabel("Overlap between operation N and operation N-1 (ms)")

# %%
import plotly.figure_factory as ff
import pandas as pd
df = pd.DataFrame({
    "Start": truestarts,
    "Finish": trueends,
    "Task": np.arange(truestarts.shape[0])
   #"Task": "ReduceScatter"
})


# %%


fig = ff.create_gantt(df, show_colorbar=False, showgrid_x=True, showgrid_y=True, index_col='Task', group_tasks=True)
fig.update_layout(title="RCCL Operations Timeline (size 8192)", xaxis_type='linear', xaxis_title="Time (ms)", yaxis_title="Rank")
fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig.show()
df

# %%
fig, ax = plt.subplots(figsize=(200,200),dpi=400)
ax.eventplot(starts)
ax.eventplot(stops, colors='C1')

# %%
fig.savefig("high_res_event.pdf",dpi=400)

# %%
starts

# %%
comm = global_comms[comms_by_len[48][0]]
durations = comm.get_completed_durations(fillMissing=True)

# %%
correct_stall_counts = np.zeros_like(durations)
channel = None
for lc in lcomm:
    if lc.get_proxy_stall_count() > 0:
        for op in lc.completed_operations + lc.pending_operations:
            for proxy_stall in op.proxy_stalls:
                if channel is None:
                    channel = proxy_stall.channel_id
                if channel != proxy_stall.channel_id:
                    continue
                seq_num = int(proxy_stall.contextstr.split("|")[1])//2
                #print(lc.localRank)
                #print(seq_num)
                correct_stall_counts[lc.localRank, seq_num] += 1

# %%
np.argwhere(correct_stall_counts)

# %%
plt.imshow(correct_stall_counts, aspect='auto', interpolation='nearest', cmap='gist_yarg')
ax = plt.gca()
ax.set_xlabel("Sequence Number")
ax.set_ylabel("Local Rank")
ax.set_title("Correct Stall Counts")
plt.colorbar(ax.images[0], ax=ax, label='Proxy Print Counts')

# %%
np.argwhere(correct_stall_counts)

# %%
print(f"Max duration: {np.nanmax(durations[:,:200])}")
print(f"Min duration: {np.nanmin(durations)}")
print(f"Mean duration: {np.nanmean(durations)}")

# %%
np.unravel_index(np.nanargmax(durations), durations.shape)

# %%
ops = comm.local_communicators[2215].get_operations()

# %%
completed_ops = comm.local_communicators[2215].completed_operations
pending_ops = comm.local_communicators[2215].pending_operations

# %%
comm.local_to_global_rank_map[2215]

# %%
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
durations.shape
fig = plt.figure(figsize=(10,400),dpi=400)
ax = plt.axes()
opmin = 190
opmax = 200
norm = plt.Normalize(vmin=np.nanmin(durations[:,opmin:opmax]), vmax=np.nanmax(durations[:,opmin:opmax]))
ax.imshow(durations[:,opmin:opmax],aspect='auto', interpolation='nearest', norm=norm,cmap='gist_yarg')
ax.set_title("Operation Durations")
ax.set_xlabel("Sequence number")
ax.set_xticks(np.arange(opmax - opmin), labels=np.arange(opmin, opmax))
ax.set_ylabel("Rank")
plt.colorbar(ax.images[0], ax=ax, label='Duration (s)')
fig.savefig("high_res_slowdown.pdf",dpi=400)

# %%
np.unravel_index(np.nanargmax(durations[:,opmin:opmax]), durations[:,opmin:opmax].shape)

# %%
comm.local_to_global_rank_map[2572]

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
for opnum in range(opmin, opmax):
    ax.bar(np.arange(durations.shape[0]), durations[:,opnum], zs=opnum, zdir='y', alpha=1.0)

# %%
all_op_durations = np.array([op.duration for op in lcomm[0].completed_operations])

# %%
plt.plot(all_op_durations)

# %%
allgathers = [ op for op in lcomm[0].completed_operations if op.op_type == "AllGather" ]
allgather_durations = np.array([ [ op.seq_num, op.duration ] for op in allgathers])
plt.plot(allgather_durations[:, 0], allgather_durations[:, 1])

# %%
reduce_scatters = [ op for op in lcomm[0].completed_operations if op.op_type == "ReduceScatter" ]
reduce_scatter_durations = np.array([ [ op.seq_num, op.duration ] for op in reduce_scatters])
plt.plot(reduce_scatter_durations[:, 0], reduce_scatter_durations[:, 1])

# %%
fig = plt.figure()
ax = plt.gca()
ax.plot(allgather_durations[:, 0], allgather_durations[:, 1], label='AllGather')
ax.plot(reduce_scatter_durations[:, 0], reduce_scatter_durations[:, 1], label='ReduceScatter')
ax.set_xlabel('Sequence Number')
ax.set_ylabel('Duration (ms)')
ax.legend()
for op in lcomm[0].completed_operations:
    if len(op.proxy_stalls) > 0:
        print(f"Operation {op.seq_num} has stalls: {op.proxy_stalls}")
        ax.axvline(x=op.seq_num, color='red', linestyle='--')

# %%



# %%
opcounts = stall_comm.get_pending_opcounts()
for i, count in enumerate(opcounts):
    if count > 0:
        print(f"Local {i} (global {stall_comm.local_to_global_rank_map[i]}) has {count} pending operations.")
        print(stall_comm.get_local_communicator(i))
        print(stall_comm.get_local_communicator(i).pending_operations)

# %%
uncounted_stalls = stall_comm.get_proxy_stall_counts_on_operations()
uncounted_stalls

# %%
stall_comm.trace_proxy_stalls(allowUneven=True)

# %%
print(global_comms['0xc0a6e4e99db3f284'].local_communicators[0].completed_operations)

# %%
global_comms['0xc0a6e4e99db3f284'].get_completed_durations()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook")
%matplotlib inline

# %%
def plot_op_times(comm, opstart=0, opend=None):
    completed_ops = comm.get_completed_durations()
    figure, ax = plt.subplots(figsize=(12, 6))
    for op in range(completed_ops.shape[1])[opstart:opend]:
        ax.plot(np.arange(completed_ops.shape[0]), completed_ops[:,op], label=f"op {op}, {comm.local_communicators[0].completed_operations[op].op_type}")
    ax.set_xlabel("rank")
    ax.set_ylabel("duration (ms)")
    ax.legend()

# %% [markdown]
# 

# %%
timeout_comm = global_comms['0xc0a6e4e99db3f284']

# %%
durations = timeout_comm.get_completed_durations()

# %%
durations.min()

# %%
timeout_comm.local_to_global_rank_map[durations.argmin()]

# %%
for i in range(len(durations)):
    if durations[i] < 500000:
        print(f"Rank {i} has a duration of {durations[i]} ms")
        print(f"Global Rank {timeout_comm.local_to_global_rank_map[i]}")

# %%
plot_op_times(timeout_comm)

# %%
other_half = global_comms['0x6b35aecc33f079ef']

# %%
other_half.get_completed_durations().min() // 1000.0

# %%
plot_op_times(other_half)

# %%
ag1 = global_comms['0xbc8a3ad231751b7']

# %%
plot_op_times(ag1,opstart=9, opend=10)

# %%
global_comms_1 = pickle.load(open("test_1_output.pickle", "rb"))

# %%
global_comms_6 = pickle.load(open("test_6.pickle", "rb"))
comms_by_len = {}
for comm_id, comm in global_comms_6.items():
    if comm.size not in comms_by_len:
        comms_by_len[comm.size] = []
    comms_by_len[comm.size].append(comm_id)

for comm in comms_by_len[4096]:
    print(comm)
    print(f"Consistent: {global_comms_6[comm].check_consistency()}")
    print(f"{global_comms_6[comm].get_completed_opcounts()[0]} Completed Operations:")
    durations = global_comms_6[comm].get_completed_durations()
    #print(global_comms[comm].get_completed_operations())
    for iop, op in enumerate(global_comms_6[comm].get_completed_operations()):
        print(f"Operation {iop}: {op.op_type} count: {op.count} dtype: {op.dtype} Times: max {durations[:, iop].max()} ms min {durations[:, iop].min()} ms mean {durations[:, iop].mean()} ms")
    print("pending opcounts:")
    print(global_comms_6[comm].get_pending_opcounts())

# %%
reduce_comm = global_comms_6['0x4ed83dba7f3d55a1']
plot_op_times(reduce_comm, opstart=0, opend=10)

# %%
len(comms_by_len[4096])

# %%
global_comms_working = pickle.load(open("working_1.pickle", "rb"))
comms_by_len = {}
for comm_id, comm in global_comms_working.items():
    if comm.size not in comms_by_len:
        comms_by_len[comm.size] = []
    comms_by_len[comm.size].append(comm_id)

for comm in comms_by_len[4096]:
    print(comm)
    print(f"Consistent: {global_comms_working[comm].check_consistency()}")
    print(f"{global_comms_working[comm].get_completed_opcounts()[0]} Completed Operations:")
    durations = global_comms_working[comm].get_completed_durations()
    #print(global_comms[comm].get_completed_operations())
    for iop, op in enumerate(global_comms_working[comm].get_completed_operations()):
        print(f"Operation {iop}: {op.op_type} count: {op.count} dtype: {op.dtype} Times: max {durations[:, iop].max()} ms min {durations[:, iop].min()} ms mean {durations[:, iop].mean()} ms")
    print("pending opcounts:")
    print(global_comms_working[comm].get_pending_opcounts())

# %%
reduce_comm = global_comms_working['0x430365659d9cd9ef']
plot_op_times(reduce_comm, opstart=0, opend=10)

# %%



