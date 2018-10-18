cmake_minimum_required(VERSION 3.0)

find_package(PythonLibs REQUIRED EXACT 3.5)
include_directories(${PYTHON_INCLUDE_DIRS})

find_path(PYBIND11_INCLUDE_DIR pybind11/numpy.h
  HINTS "${PYBIND11_INCLUDE_DIR}" "${PYTHON_INCLUDE_DIRS}")
if(PYBIND11_INCLUDE_DIR)
  include_directories(${PYBIND11_INCLUDE_DIR})
else()
  message(FATAL "Cannot find PYBIND11 include files. Please install/download PyBind11 at https://github.com/pybind/pybind11")
endif()

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable (fetalnav
  fetalnav.cxx
  )

target_link_libraries(fetalnav
  ${ITK_LIBRARIES}
  ${PYTHON_LIBRARIES}
  )
