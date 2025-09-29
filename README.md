# RCCL/NCCL Log Analysis Tools

The project will attempt to parse log files from multiple processes across multiple nodes and read the NCCL/RCCL logging and proxy dump information to assemble a representation of the communication. It can then attempt to identify places where the communication has stalled and trace them to the origin.

## Prerequisites

You need a reasonably modern python with these extra packages installed:

- numpy
- parse
- matplotlib (required only for stall_explorer)
- plotly (required only for stall_explorer)
- seaborn (required only for stall explorer)
- dragonhpc (recommended on a cluster that supports it)

On Frontier `module load cray-python` should have numpy but you'll need to install parse through pip. I suggest just activating the venv you use for other testing and loading it into there.

## Workflow for profiler logs

Profiler output is much more structured and easy to parse. After collecting data, just run:

```
python profiler_logs_to_sRTp.py -d <path_to_data>  -o <output-pickle>
```

Note that this assumes you used my varient of the profiler plugin that does better file names, which you probably don't. I'll get that up soon.

## Workflow for log parsing (deprecated)


The basic workflow is as follows. Details on each step are below.

For a directory `DIR` containing per-rank RCCL output using the `rank-<number>.out` syntax favored by Arm, you'll do:

```
python cleanup_output.py --unique frontier DIR
python arm_logs_to_sRTp.py -d DIR-cleaned -o DIR_output.pickle
python first_pass_analysis.py -i DIR_output.pickle
< look at output and identify communicator IDs (64bit hex values) of interest>
python first_pass_analysis.py -c <comID1> <comID2> -i DIR_output.pickle
```

You will likely want to redirect the last step to a file, as the proxy tracing can be very verbose.

### Cleanup

Many processes writing different streams to a file can cause some collisions, dangling lines, and other things that will screw up the parser. `cleanup_output.py` will do its best to repair lines so the parser will succeed. If it can't repair a line, it will print a warning with the file name so you can try to repair it manually.

Proxy dump lines are most likely to be clobbered beyond automatic repair. You can *probably* ignore them because there are usually many proxy dump timeouts for one failure on a stuck channel. The analysis will ignore parsing failures in proxy dump lines for that reason.

Other warnings will probably need to be fixed.

### Parsing to a pickle

`arm_logs_to_sRTp.py` reads cleaned log files and and parses them into python objects for analysis. It will then save those objects in a python pickle. Pickles are portable from system to system, so you could parse the logs on Frontier and move the pickle to a laptop for analysis. The only requirement is that the object data definitions in the local module need to be the same as the ones in the pickle. The implementation details of methods can change, but the attributes need to be the same.

If this step fails then you probably need to ping me (Steve) with your logs. I've tried to handle everything in this that could cause a failure in the cleanup script, so something must have slipped through.

### Analyzing

My prefered way to analyze the data is load the pickle into an ipython/jupyter notebook and feel my way through the data. You can see an example of that in this repository.

That can be a little difficult without understanding how I've organized the modules, and the documentation isn't the best, so I've created a first pass analysis script.

`first_pass_analysis.py` will take the data in the pickle and tell you how many communicators are in it, their size, and then some basic information about the biggest communicators. You should then identify the communicators that are of most interest to you, either because they timed out or didn't complete, and then run the script again with those ID's as arguments to `-c`. I suggest you capture the output, as the ring tracing can be quite verbose.

*WARNING*: I think I've covered a lot of edge cases, but I discovered one I missed: I've handled cases where the operation counts are inconsistent across a communicator (e.g. some ranks record 5 operations and some record 6), except I always assumed that the first rank would have the maximum number of operations. In at least one case I've found this isn't true and the script breaks, so I'll fix that after vacation.

## Understanding the output

I've committed an example of output to this repository as `example_output.txt`. Here's some basic things to look for.

In picking a communicator to trace we're looking either for a timeout or unmatched operations. We follow the send path, so often what you're looking for will be at the end.

### Timeouts

Timeouts normally show up as completions that take a very long time, like this:

```
Completed operations:
        Operation 0: ReduceScatter count: 131072 dtype: 9 Times: max 600776.3800000539 ms min 600136.8870000006 ms mean 600492.9080419925 ms
```

When you trace the communicator it will give you output like this:

```
Mismatch in recvtail proxy output from rank 3000 (Global 6000) on rank 3004 (Global 6008).
Global communicator 0xc0a6e4e99db3f284 operation ReduceScatter seq_num 0.
0x7ff5ee771d48 [0-1|0| coll:3 comm:0x38d8e460 [SEND] dtype:9 redOp:0 proto:2  nb:1048576 ns:16380 p:44 t:44 r:0, d:36   myrank:3000 peer:3004 chan:1 tail:42 recvtail:44 reg:0 connSz:0(retries:1117146690)]Traced: True
0x7ff5ee771d48 [0-1|0| coll:3 comm:0x1b8b6c80 [RECV] dtype:9 redOp:0 proto:2  nb:1048576 ns:16380 p:36 t:36 r:36, d:28   myrank:3004 peer:3000 chan:1 tail:0 recvtail:36 reg:0 connSz:32768(retries:1803778500)]Traced: False
```

### Unmatched Operations

Sometimes there will be a different number of operations on different ranks of the communicator. That looks like this:

```
Communicator 0x688750e7f55f9410 has size 4096.
Operation counts are not consistent across local communicators.
Operation count 30 appears 1624 times.
Operation count 31 appears 2472 times.
Completed operation counts are not consistent across local communicators.
Completed operation count 30 appears 1624 times.
Completed operation count 31 appears 2472 times.
Consistent: False
```
This may indicate some hang or desync, or maybe it's an application problem. I'm not really sure yet. You can still trace the communicators, but you'll see a mix of output like:
```
!!!!WARNING!!!:Operation with index 18 does not exist on rank 3004 (Global 6008).
This explains the stall from rank 3000 (Global 6000) to rank 3004 (Global 6008).
0x7ff5e697ca78 [0-2|36| coll:3 comm:0x1b8d3080 [SEND] dtype:9 redOp:0 proto:2  nb:1048576 ns:16380 p:8 t:8 r:16380, d:0   myrank:3000 peer:3004 chan:1 tail:294846 recvtail:294848 reg:0 connSz:0(retries:667387229)]Traced: True
Skipping and moving along channel.
```
That will tell you when the hole in operation counts starts. It seems likely that indicates some stall when communicating with the other half of the job.