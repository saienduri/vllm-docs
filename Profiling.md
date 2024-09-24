**v0.1, 9/24/2024**

# Profiling using PyTorch Profiler

## Analyze execution time

Add the necessary imports:
```python
from torch.profiler import profile, record_function, ProfilerActivity
```


Define a trace handler function to export the trace in a format suitable for trace viewers like Perfetto or Chrome trace viewer:
```python
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace(“/path/to/traces/trace.json”)
```

Wrap the training loop in this profiling context to capture the trace. Make sure that everything you want to capture (dataloading, communication, checkpointing, etc) is included within this context. Step forward the profiler at the end of each training iteration.
```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=5,),
    on_trace_ready=trace_handler
) as p:

  for batch in dataloader:
    train_step(batch)
    p.step() 



```
More information on how to set the schedule parameters to efficiently analyze long running training jobs can be found here: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs

The profiler trace will be saved to `trace.json`. Open this file in a trace viewer like Perfetto to visualize the trace.


### Examples

#### 1. Abnormally long multi-node communication kernels

While scaling the RegNet+GPT workload to multiple nodes, we observed that some model configurations showed very large performance degradation. Here we show an example of how we used the profiler to give us clues that helped debug this issue.

First, we isolated just the GPT part and measured its multi-node scaling. We observed that the performance drop was small and the NCCL/RCCL communication kernels took ~1ms each:
 
![image](https://github.com/user-attachments/assets/7ff839d4-43ab-4d7a-9a33-50e23d6919b4)

Next we profiled the combined RegNet+GPT workload that showed very poor multinode scaling. The trace showed us that the communication kernels for the convolutional RegNet did not cause a major bottleneck and that the GPU utilization for this network was good. This ruled out any inefficiencies in the RegNet being the cause of the poor scaling. However, we noticed that the same communication kernels took around 500x longer in the combined RegNet+GPT case than in the isolated GPT case:

![image](https://github.com/user-attachments/assets/80e80820-9936-47d7-84ca-22d51695f901)

From this observation, we hypothesized that running the model with close to full GPU memory utilization causes smaller communication buffers to be allocated, resulting in greater transmission delay. 

Based on this hypothesis, we tested the RegNet+GPT workload with a batch size that is 1 smaller than the maximum possible batch size. With this reduced batch size, we see the communication bottleneck alleviated and more reasonable multi-node scaling.

#### 2. Dataloader bottleneck

In the RegNet+GPT workload, the random data generation and transfer to GPU memory takes up a significant amount of time and is not overlapped with computation, leaving the GPU idle. This bottleneck can be identified using the CPU and GPU trace visualization as shown in the example screenshot below:

![image](https://github.com/user-attachments/assets/60f84181-a5ac-4d8c-bb8e-b4ef3481f112)




## Analyze memory consumption
 
To analyze memory usage during the training process, we can use PyTorch's CUDA memory management APIs to record memory events and generate memory snapshots. This provides detailed information about memory allocations and frees, which can be visualized for better understanding and optimization.

### Recording Memory History
First, we start by recording the memory allocation and free events over the course of the training loop. By setting up the recording mechanism before the training loop, we can capture memory-related events like allocations, deallocations, and re-use of memory blocks.
 
Example code snippet to start and stop memory history recording:
 
```python
def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return
logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )
def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return
logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)
```

In this function, we specify `MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT` to limit the number of memory events to capture, ensuring we don’t overwhelm the system with excessive data for long-running jobs. 
 
### Exporting Memory Snapshot

To visualize memory usage, we can export memory snapshots at any point during or after the training. These snapshots contain detailed information about memory allocations and can be saved to disk in a format that can later be analyzed using external tools or custom scripts.
 
```python
def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return
# Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"
try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return
```

### Analysing memory snapshot pickle files
 
This method generates a memory snapshot and saves it as a `.pickle` file, which can be visualized by dragging and dropping into the user interface provided by PyTorch at PyTorch Memory Visualization. This tool allows users to adjust the level of detail by filtering out smaller memory events to simplify the view.
 
 
### Example of a Memory Profile:

![image](https://github.com/user-attachments/assets/9622984b-c5c7-4128-9a8b-1274677762b4)

The image above illustrates a memory profile. The upward slope represents the forward pass where the activations are allocated, highlighting the increasing memory usage as more activations are stored. Conversely, the downward slope indicates the backward pass where gradients are computed and activations are deallocated, freeing up memory. The horizontal lines spanning the full timeline represent static memory, such as model parameters and optimizer state.
Hovering over the user interface provides detailed information about memory allocations, including the memory address and the specific code trace that triggered the allocation. It also shows the size of each allocation, helping developers understand and optimize memory usage during model training.

