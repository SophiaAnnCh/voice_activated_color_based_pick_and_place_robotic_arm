# Voice-Activated Color-Based Pick-and-Place Robotic Arm

## Installation and Setup

This guide explains how to set up the project on a fresh Ubuntu system.

### 1. Prerequisites

Before getting started, ensure you have:

- **Ubuntu 22.04 LTS**
- **ROS 2 Humble**
- An active internet connection to download dependencies and speech-recognition models
- A working microphone for voice commands
- Sufficient system resources to run Gazebo and RViz

### 2. Clone the Repository

Clone the project and navigate to the repository directory:

```bash
git clone https://github.com/SophiaAnnCh/voice_activated_color_based_pick_and_place_robotic_arm.git

cd voice_activated_color_based_pick_and_place_robotic_arm
```

### 3. Set Up the ROS 2 Environment

Source the ROS 2 Humble environment:

```bash
source /opt/ros/humble/setup.bash
```

### 4. Install System Dependencies

Update the package index and install the required ROS 2 packages and development tools:

```bash
sudo apt update

sudo apt install -y \
  ros-humble-moveit \
  ros-humble-ros-gz \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-cv-bridge \
  ros-humble-tf2-ros \
  ros-humble-tf2-tools \
  ros-humble-tf-transformations \
  ros-humble-xacro \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pip \
  ffmpeg
```

### 5. Initialize rosdep and Install Dependencies

`rosdep` resolves dependencies declared in ROS package manifests.

Initialize it if it has not already been initialized:

```bash
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi

rosdep update
```

Install the dependencies declared by the packages in the workspace:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

**Note:** This command assumes the repository contains the ROS packages in a `src/` directory. If the repository is structured differently, adjust the path or workspace layout accordingly.

### 6. Build the Workspace

Build the ROS 2 workspace:

```bash
colcon build --symlink-install
```

After the build completes successfully, source the workspace overlay:

```bash
source install/setup.bash
```

To make the environment available in a new terminal, source both environments again:

```bash
source /opt/ros/humble/setup.bash
source ~/voice_activated_color_based_pick_and_place_robotic_arm/install/setup.bash
```

Replace the workspace path in the final command if you cloned the repository into a different directory.

### 7. Verify the Installation

After building and sourcing the workspace, verify that the expected ROS 2 packages are available:

```bash
ros2 pkg list | grep -E 'panda_(bringup|controller|description|moveit|nlp|vision)|pymoveit2'
```

The expected output is:

```text
panda_bringup
panda_controller
panda_description
panda_moveit
panda_nlp
panda_vision
pymoveit2
```

These packages provide the main components of the project:

| Package | Purpose |
|---|---|
| `panda_bringup` | Launches the integrated pick-and-place simulation |
| `panda_controller` | Contains robot controller launch configurations |
| `panda_description` | Provides the Panda robot description and Gazebo launch configuration |
| `panda_moveit` | Provides MoveIt 2 planning and robot configuration |
| `panda_nlp` | Processes voice commands using speech recognition and NLP |
| `panda_vision` | Detects colored objects and publishes their coordinates |
| `pymoveit2` | Provides the Python MoveIt 2 interface and pick-and-place script |

Next, verify that the expected executables are registered by running the following commands:

```bash
ros2 pkg executables panda_nlp
ros2 pkg executables panda_vision
ros2 pkg executables panda_controller
ros2 pkg executables pymoveit2
```

Expected executable names include:

```text
panda_nlp voice_command
panda_vision color_detector
panda_controller slider_controller.py
pymoveit2 pick_and_place.py
```

Additional example executables may also appear under `pymoveit2`.

If the expected packages and executables are available, the initial workspace build and package registration checks are complete.

## 8. Launching the Project

Follow these steps to start the simulation and run the voice-guided pick-and-place system.

### Step 1: Start the Simulation and MoveIt 2

Open a terminal and run:

```bash
source /opt/ros/humble/setup.bash
source ~/voice_activated_color_based_pick_and_place_robotic_arm/install/setup.bash

ros2 launch panda_bringup pick_and_place.launch.py
```

This launch file starts:
- Gazebo simulation with the Panda robotic arm.
- MoveIt 2 for motion planning.
- The color detection node.

Wait for Gazebo, MoveIt 2, and the robot controllers to finish initializing before continuing.

### Step 2: Start the Voice Command Node

Open a **second terminal** and run:

```bash
source /opt/ros/humble/setup.bash
source ~/voice_activated_color_based_pick_and_place_robotic_arm/install/setup.bash

ros2 run panda_nlp voice_command
```

Wait until the voice-command node reports that it is ready.

### Step 3: Start the Pick-and-Place Node

Open a **third terminal** and run:

```bash
source /opt/ros/humble/setup.bash
source ~/voice_activated_color_based_pick_and_place_robotic_arm/install/setup.bash

ros2 run pymoveit2 pick_and_place.py
```

Wait for the node to initialize and confirm that it is ready to receive commands.

### Step 4: Verify the Running System

Open another terminal and check the active ROS 2 nodes:

```bash
source /opt/ros/humble/setup.bash
source ~/voice_activated_color_based_pick_and_place_robotic_arm/install/setup.bash

ros2 node list
```

Check that the camera and vision topics are available:

```bash
ros2 topic list | grep -E 'camera/image_raw|color_coordinates'
```

Verify that the robot controllers are active:

```bash
ros2 control list_controllers
```

The arm controller, gripper controller, and joint-state broadcaster should report an `active` state.

### Step 5: Issue a Voice Command

Once all nodes are running, activate recording in the voice-command terminal according to its prompt.

For example, say:

> "Pick up the red ball."

The intended workflow is:

1. The voice node transcribes the spoken command.
2. The NLP module extracts the requested action and object color.
3. The vision node detects the corresponding colored object and publishes its coordinates.
4. The pick-and-place node uses MoveIt 2 to plan and execute the grasping and placement sequence.

Ensure the target object is visible to the simulated camera and that its coordinates are valid before executing robot motion.






