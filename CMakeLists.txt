cmake_minimum_required(VERSION 3.0.2)
project(MSG-Loc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress PCL developer warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  nav_msgs
  message_filters
  tf2_ros
  visualization_msgs
  message_generation
  rospy
  cv_bridge
  image_geometry
  image_transport
  geometry_msgs
)

find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  
  # === [여기가 추가되어야 할 핵심 포인트] ===
  ${gtsam_quadrics_INCLUDE_DIR}     # 시스템(make install)에 설치된 경우
  /root/workspace/gtsam-quadrics    # 워크스페이스에 로컬 빌드만 해둔 경우 (절대경로)
)

# Link directories
link_directories(
  ${GTSAM_LIBRARY_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  # === [여기에 라이브러리 파일(.so)이 생성된 경로 추가] ===
  /root/workspace/gtsam-quadrics/build
)


generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   nav_msgs
   geometry_msgs
)

# Catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime roscpp std_msgs sensor_msgs nav_msgs geometry_msgs
)

# Catkin Python scripts
# catkin_install_python(PROGRAMS
#   src/quadric_script/detection_storage_gdino_lvis.py
#   src/quadric_script/detection_storage_gdino_coco.py
#   src/quadric_script/detection_storage_yolo_lvis.py
#   src/quadric_script/detection_storage_yolo_coco.py
#   src/quadric_script/quadricSLAM.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Add executables and link libraries

add_executable(msgloc_node src/msgloc.cpp)
target_link_libraries(msgloc_node
  ${catkin_LIBRARIES}
  gtsam
  gtsam_quadrics
  OpenMP::OpenMP_CXX
)
