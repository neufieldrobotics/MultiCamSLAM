option(BUILD_DemoORB  "Build demo application with ORB features" ON)

#set(HDRS
#        ${PROJECT_SOURCE_DIR}/src/modules/loopclosure_analysis/DLoopDetector.h
#        ${PROJECT_SOURCE_DIR}/src/modules/loopclosure_analysis/TemplatedLoopDetector.h)
set(DLib_DIR "/home/auv/software/DLib/build")
find_package(DLib REQUIRED)
message("Dlib LIBS" ${DLib_LIBS})
message("Dlib includes" ${DLib_INCLUDE_DIRS})
include_directories( ${PROJECT_SOURCE_DIR}/src/modules/loopclosure_analysis)
include_directories( ${DLib_INCLUDE_DIRS})

if(BUILD_DemoORB)
    add_executable(demo_orb  ${PROJECT_SOURCE_DIR}/src/modules/loopclosure_analysis/demo_orb.cpp )
    target_link_libraries(demo_orb  ${DLib_LIBS} ${LIBS_LF} ${LIBS_SLAM} ${catkin_LIBRARIES} /usr/lib/x86_64-linux-gnu/libpython2.7.so)
    #target_include_directories(demo_orb PUBLIC ${DLib_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS})
    #set_target_properties(demo_orb PROPERTIES CXX_STANDARD 11)
endif(BUILD_DemoORB)