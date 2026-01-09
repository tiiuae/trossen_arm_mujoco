# ALOHA Simulation on Linux

This folder contains Docker configuration files to run the ALOHA simulation environment on Linux with NVIDIA GPU support.

## Prerequisites

1. **Docker** - [Install here](https://docs.docker.com/engine/install/ubuntu/)
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **NVIDIA Docker Runtime** - Required for GPU acceleration
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **NVIDIA GPU Drivers** - Ensure you have the latest NVIDIA drivers installed
   ```bash
   nvidia-smi  # Verify GPU is detected
   ```

4. **Git** - For cloning/managing the repository

## Building the Docker Image

```bash
cd /path/to/trossen_arm_mujoco/old_aloha
docker build --pull --rm -f Dockerfile -t interbotix_ros_simulation:latest .
```

This builds the Docker image with ROS 2 Humble and Interbotix dependencies. The build may take 10-15 minutes.

## Running with Docker Compose

1. **Allow X11 forwarding:**
   ```bash
   xhost +local:docker
   ```

2. **Start the container:**
   ```bash
   cd old_aloha
   docker-compose up -d
   ```

3. **Attach to the container:**
   ```bash
   docker-compose exec simulation bash
   ```

4. **Verify GPU is available (inside container):**
   ```bash
   nvidia-smi
   ```

## Running Gazebo Simulation

Inside the container:

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash
source /workspace/interbotix_ws/install/setup.bash

# Launch Gazebo with your robot
ros2 launch interbotix_xsarm_gazebo xsarm_gazebo.launch.py robot_model:=wx250s
```

## Troubleshooting

### "Cannot connect to X server"
- Run `xhost +local:docker` before starting the container
- Ensure `DISPLAY` environment variable is set correctly
- Check if X11 is running: `echo $DISPLAY`

### "nvidia-smi not found" or GPU not detected
- Verify NVIDIA drivers are installed: `nvidia-smi` (on host)
- Ensure nvidia-docker2 is installed and configured
- Restart Docker daemon: `sudo systemctl restart docker`
- Check Docker can access GPU: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

### "Address already in use"
- A previous container instance is still running
- Stop it with: `docker-compose down`

### Building fails with CMake errors
- The Dockerfile includes `ros2-control-test-assets` dependency
- Ensure you have stable internet connection during build
- Try rebuilding: `docker-compose build --no-cache`

### Gazebo crashes or renders poorly
- Ensure NVIDIA GPU drivers are up to date
- Check GPU is being used: `nvidia-smi` inside the container
- Try reducing graphics quality in Gazebo settings

## Container Management

**Stop the container:**
```bash
docker-compose down
```

**View running containers:**
```bash
docker ps
```

**View container logs:**
```bash
docker-compose logs -f simulation
```

**Restart the container:**
```bash
docker-compose restart
```

**Remove the image:**
```bash
docker rmi interbotix_ros_simulation:latest
```

## File Sharing

- The `worlds/` directory is shared between host and container
- Place Gazebo world files here to access them in the container at `/root/gazebo_worlds`

## Environment Variables

The `docker-compose.yml` file sets:
- `DISPLAY=${DISPLAY}` - For X11 forwarding from host
- `NVIDIA_VISIBLE_DEVICES=all` - Makes all GPUs available
- `NVIDIA_DRIVER_CAPABILITIES=all` - Enables all NVIDIA capabilities
- `network_mode: host` - Uses host networking for better performance

## GPU Acceleration

This setup uses NVIDIA GPU acceleration for:
- **Gazebo rendering** - Smooth 3D visualization
- **Physics simulation** - Faster computation
- **ROS 2 perception** - Hardware-accelerated image processing

The container is configured with:
- NVIDIA Docker runtime
- Direct GPU access via `--gpus all`
- All NVIDIA driver capabilities enabled

## Notes

- The image is built with `--platform=linux/amd64`
- ROS 2 Humble is used in this configuration
- The container runs in privileged mode with host networking
- GPU acceleration is enabled by default
- Gazebo worlds are mounted to `/root/gazebo_worlds` in the container
