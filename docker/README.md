# Easy Setup with Docker and DevContainer for phyai

To simplify developers' experience with phyai, we provide ready-to-use Dockerfile and DevContainer configurations.

## 1. Using Dockerfile

Run these commands from the provided source directory. The build uses that local source as its context.

```bash
# NVIDIA GPU: choose the Dockerfile for your CUDA version.
docker build -f docker/Dockerfile.torch211cu13 -t phyai:torch211-cu13 .
docker run -it --gpus all --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name phyai_torch211cu13_dev phyai:torch211-cu13 bash
```

## 2. Using DevContainer

To set up with VS Code Dev Containers:

1. Install prerequisites:
    - Docker
    - VS Code
    - Dev Containers extension

2. Open the provided source directory in VS Code:

    ```shell
    code .
    ```

3. When prompted:

    "Folder contains a Dev Container configuration file. Reopen in container?"
    Click Reopen in Container

    (Alternatively: Press F1 → "Dev Containers: Reopen in Container")

The container will automatically build and launch with:

* All dependencies pre-installed
* Correct environment configuration
* Shared memory and security settings applied
