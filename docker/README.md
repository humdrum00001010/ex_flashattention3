# RunPod image

Build the exact current experiment and Nx/EXLA overlay:

```sh
NX_WORKTREE=/path/to/nx-worktree \
IMAGE=docker.io/your-namespace/fa3-tp-nx:v0.1.0-cuda13-xla010 \
PUSH=1 \
./docker/build-push.sh
```

The image targets `linux/amd64`, stores its payload under `/opt/fa3-tp`, and
starts an SSH daemon using RunPod's injected `PUBLIC_KEY`. RunPod may safely
mount persistent storage at `/workspace` without hiding the application.

Create a private NVIDIA Pod template with:

- image: `docker.io/your-namespace/fa3-tp-nx:v0.1.0-cuda13-xla010`
- container disk: at least 50 GB
- TCP ports: `22/tcp`
- volume mount: `/workspace`
- entrypoint/start command: leave empty

With `runpodctl` 1.14, create the required two-GPU H100 SXM Pod from that
image with:

```sh
runpodctl create pod \
  --name fa3-tp-nx \
  --imageName docker.io/your-namespace/fa3-tp-nx:v0.1.0-cuda13-xla010 \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --gpuCount 2 \
  --secureCloud \
  --containerDiskSize 50 \
  --volumeSize 1 \
  --volumePath /workspace \
  --mem 64 \
  --vcpu 16 \
  --ports "22/tcp" \
  --startSSH
```

After SSH connects:

```sh
fa3-test
fa3-bench
```

Both commands create a fresh HLO directory, enable XLA custom-call command
buffers, and use the baked `libfa3_xla.so`.
