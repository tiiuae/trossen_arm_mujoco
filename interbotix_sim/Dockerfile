FROM --platform=linux/amd64 osrf/ros:humble-desktop-full



# Install base packages
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    rosdep update

RUN apt-get install -y git-all curl nano net-tools
RUN apt-get install -y build-essential cmake

# === Install required ROS 2 control dependencies ===
RUN apt-get install -y ros-humble-ros2-control-test-assets

# === Install X11 utilities for GUI forwarding ===
RUN apt-get update && apt-get install -y x11-apps xauth
# Set DISPLAY variable (will be overridden at runtime)
ENV DISPLAY=:0

# === Install interbotix packages ===
# Install Interbotix dependencies
WORKDIR /workspace
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
RUN chmod +x xsarm_amd64_install.sh

# Install Interbotix packages (this creates /workspace/interbotix_ws)
RUN ./xsarm_amd64_install.sh -d humble -p /workspace/interbotix_ws -n

# Clean up the installation script
RUN rm -f /workspace/xsarm_amd64_install.sh

# Copy the python scripts into the container
COPY robot-controllers/ /workspace/robot-controllers/

# Write to bashrc to source the interbotix workspace
RUN echo "" >> ~/.bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/interbotix_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
