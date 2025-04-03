cmake_minimum_required(VERSION 3.15)
project(GuitarToneEmulatorTests)

# Include the main project's CMakeLists.txt
include(${CMAKE_SOURCE_DIR}/../CMakeLists.txt)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(GTest REQUIRED)

# Add source files
set(SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/../Source/Features/FeatureExtractor.cpp
    ${CMAKE_SOURCE_DIR}/../Source/Neural/ToneMatcher.cpp
)

# Add test executable
add_executable(GuitarToneEmulatorTests 
    FeatureExtractorTest.cpp
    ToneMatcherTest.cpp
)

# Link with the main plugin target to get access to all the source files
target_link_libraries(GuitarToneEmulatorTests 
    PRIVATE
    GuitarToneEmulator
    gtest_main
    gtest
    juce::juce_audio_basics
    juce::juce_audio_formats
    juce::juce_core
)

# Add compile definitions
target_compile_definitions(GuitarToneEmulatorTests
    PRIVATE
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
    JUCE_STANDALONE_APPLICATION=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_STRICT_REFCOUNTEDPOINTER=1
    JUCE_ENABLE_SCREEN_CAPTURE=0
    JUCE_ENABLE_LIVE_CONSTANT_EDITOR=0
) 