# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

options = parser.parse_args()

# %%
# Load the pickled with all the processed data
global_comms = pickle.load(open(options.input, "rb"))

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

fig.savefig(f"rccl_opduration_boxplot_{options.output}.png")

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
fig.savefig(f"rccl_opduration_histo_{options.output}.png")
# %%
#truestarts = np.min(starts, axis=0)
#trueends = np.max(stops, axis=0)



# %%
max_op

# %%
# This was Mark's suggestion to sort ranks by their duration during the longest op. It's an intersting plot, but I think less useful than just plotting the start and stop times of each op for each rank, which I do next.
import seaborn as sns
colors = sns.color_palette("colorblind")

#sorted_by_maxop_start = np.argsort(starts[:, max_op])
#nonzero_index = durations[sorted_by_maxop_start, :] > 10
#print(nonzero_index.shape)

#fig, ax = plt.subplots(figsize=(120, 60))
#for op in range(durations.shape[1]):
#    ax.plot(starts[sorted_by_maxop_start, op][nonzero_index[:, op]], ls='-', color=colors[op % len(colors)])
#    ax.plot(stops[sorted_by_maxop_start, op][nonzero_index[:, op]], ls='--', color=colors[op % len(colors)])

#ax.set_yticks(np.arange(0,np.max(stops), 24000))
#plt.grid()
#ax.set_xlabel("Rank (sorted by start time of longest op)")
#ax.set_ylabel("Time (ms)")

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
fig.savefig(f"rccl_timeline_unsorted_gridlines_{options.output}.pdf", dpi=300)

# %% [markdown]
# * STOP HERE *
# 
# Everything past this point is dinking around

exit(0)