cmake_minimum_required(VERSION 3.8)
project(sonar_3d_reconstruction)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python executable scripts (install(PROGRAMS) automatically sets executable permissions)
install(PROGRAMS
  scripts/octree_sonar_mapper_node.py
  scripts/3d_mapper_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules (non-executable Python files)
install(FILES
  scripts/octree_mapper.py
  scripts/3d_mapper.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install rviz configuration files
install(DIRECTORY
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

# Install header files (if any C++ headers exist)
install(DIRECTORY
  include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
