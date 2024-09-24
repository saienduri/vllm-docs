# rocDecode Testing using powderluv docker

* Steps to run a basic videodecode C++ sample with AMD HW Video decoder using rocDecode library.

### Step 1: Pull and run the docker
> [!NOTE]
usage: -v {LOCAL_HOST_DIRECTORY_PATH}:{DOCKER_DIRECTORY_PATH}

```
sudo docker pull powderluv/vllm_dev_channel:ROCm6.2_hipblaslt0.10.0_pytorch2.5_vllm0.6.1_cython_09192024

sudo docker run -v <Map any data (videos) from bare-metal if required> -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 126G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined powderluv/vllm_dev_channel:ROCm6.2_hipblaslt0.10.0_pytorch2.5_vllm0.6.1_cython_09192024
```

## Sample 1:

### Step 2: Inside the docker - performance scripts (please follow this initially to see performance numbers)

```
apt update
apt install rocdecode rocdecode-dev rocdecode-test
mkdir rocdecode-sample
cd rocdecode-sample/
cmake /opt/rocm/share/rocdecode/samples/videoDecodePerf/
make -j
./videodecodeperf -i /opt/rocm/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4 -t 4
```

### Other details:

* To get a list of all the parameters that can be passed to the sample, there is a help option.

```
./videodecodeperf -h
Options:
-i Input File Path - required
-t Number of threads (>= 1) - optional; default: 4
-d Device ID (>= 0)  - optional; default: 0
-z force_zero_latency (force_zero_latency, Decoded frames will be flushed out for display immediately); optional;
```


## Sample 2:

### Step 2: Inside the docker - script to actaully get the frame and do operations on it

```
apt update
apt install rocdecode rocdecode-dev rocdecode-test
mkdir rocdecode-sample
cd rocdecode-sample/
cmake /opt/rocm/share/rocdecode/samples/videoDecode/
make -j
./videodecode -i /opt/rocm/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
```

### Other details:

* The above steps shows how to run the videoDecode sample on the docker with a sample video provided by AMD. Any other HEVC/AVC video can be used on rocm 6.2 docker.

* To get a list of all the parameters that can be passed to the smaple, there is a help option.

```
./videodecode -h
Options:
-i Input File Path - required
-o Output File Path - dumps output if requested; optional
-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0
-f Number of decoded frames - specify the number of pictures to be decoded; optional
-z force_zero_latency (force_zero_latency, Decoded frames will be flushed out for display immediately); optional;
-disp_delay -specify the number of frames to be delayed for display; optional;
-sei extract SEI messages; optional;
-md5 generate MD5 message digest on the decoded YUV image sequence; optional;
-md5_check MD5 File Path - generate MD5 message digest on the decoded YUV image sequence and compare to the reference MD5 string in a file; optional;
-crop crop rectangle for output (not used when using interopped decoded frame); optional; default: 0
-m output_surface_memory_type - decoded surface memory; optional; default - 0 [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]
-seek_criteria - Demux seek criteria & value - optional; default - 0,0; [0: no seek; 1: SEEK_CRITERIA_FRAME_NUM, frame number; 2: SEEK_CRITERIA_TIME_STAMP, frame number (time calculated internally)]
-seek_mode - Seek to previous key frame or exact - optional; default - 0[0: SEEK_MODE_PREV_KEY_FRAME; 1: SEEK_MODE_EXACT_FRAME]
```

* There are other samples available inside `/opt/rocm/share/rocdecode/samples/` with a help menu for each sample.

* Details for all these samples can be found in the main GitHub repo for rocDecode for rocm 6.2: https://github.com/ROCm/rocDecode/tree/release/rocm-rel-6.2/samples