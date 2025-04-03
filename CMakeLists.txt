cmake_minimum_required(VERSION 3.15)
project(GuitarToneEmulator VERSION 1.0.0)

# Set C++ standard before anything else
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set source directory
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source")

# Add source directory to include path
include_directories(${SOURCE_DIR})

# macOS specific settings
if(APPLE)
    # Set minimum macOS version to 10.13 to maintain compatibility
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version" FORCE)
    
    # Add optimization flags
    add_compile_options(-O3)
    
    # Add AVX2 only for Intel Macs
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        add_compile_options(-mavx2)
    endif()
    
    # Add JUCE macOS compatibility flags
    add_compile_definitions(
        JUCE_MAC_OLD_SCREEN_CAPTURE=1
        JUCE_DISABLE_ASSERTIONS=1
        JUCE_CHECK_MEMORY_LEAKS=0
        JUCE_SILENCE_XCODE_15_LINKER_WARNING=1
        JUCE_DISABLE_CAUTIOUS_MAC_WARNINGS=1
        JUCE_DISABLE_CAUTIOUS_MAC_DEPRECATION_WARNINGS=1
        JUCE_DISABLE_CAUTIOUS_MAC_OSX_WARNINGS=1
        JUCE_DISABLE_CAUTIOUS_MAC_OSX_DEPRECATION_WARNINGS=1
        JUCE_DISABLE_CAUTIOUS_MAC_OSX_AVAILABILITY_WARNINGS=1
        JUCE_DISABLE_CAUTIOUS_MAC_SCREEN_CAPTURE_WARNINGS=1
        JUCE_DISABLE_CAUTIOUS_MAC_DISPLAY_LINK_WARNINGS=1
    )
endif()

# Fetch JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG develop
    GIT_SHALLOW 1
)

# Configure JUCE options before making it available
set(JUCE_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
set(JUCE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(JUCE_ENABLE_MODULE_SOURCE_GROUPS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JUCE)

# Fetch RTNeural
FetchContent_Declare(
    rtneural
    GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
    GIT_TAG main
    GIT_SHALLOW 1
)
FetchContent_MakeAvailable(rtneural)

# Fetch Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW 1
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# Fetch Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW 1
)
FetchContent_MakeAvailable(googletest)

# Create plugin
juce_add_plugin(GuitarToneEmulator
    VERSION "1.0.0"
    COMPANY_NAME "YourCompany"
    BUNDLE_ID "com.yourcompany.guitartoneemulator"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE Ycom
    PLUGIN_CODE Gtem
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Guitar Tone Emulator")

# Add source files using absolute paths
target_sources(GuitarToneEmulator
    PRIVATE
        "${SOURCE_DIR}/PluginProcessor.cpp"
        "${SOURCE_DIR}/PluginEditor.cpp"
        "${SOURCE_DIR}/Features/FeatureExtractor.cpp"
        "${SOURCE_DIR}/Neural/RTNeuralWrapper.cpp"
        "${SOURCE_DIR}/Neural/ToneMatcher.cpp"
        "${SOURCE_DIR}/Visualization/ToneVisualizer.cpp")

# Add header files
target_include_directories(GuitarToneEmulator
    PRIVATE
        "${SOURCE_DIR}"
        "${SOURCE_DIR}/Features"
        "${SOURCE_DIR}/Neural"
        "${SOURCE_DIR}/Visualization"
        "${rtneural_SOURCE_DIR}"
        "${rtneural_SOURCE_DIR}/RTNeural"
        "${eigen_SOURCE_DIR}")

# Link dependencies
target_link_libraries(GuitarToneEmulator
    PRIVATE
        RTNeural
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics)

# Add compile definitions
target_compile_definitions(GuitarToneEmulator
    PRIVATE
        # Disable unnecessary features
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=1
        JUCE_REPORT_APP_USAGE=1
        JUCE_STRICT_REFCOUNTEDPOINTER=1
        # Disable screen capture and other problematic features
        JUCE_ENABLE_SCREEN_CAPTURE=0
        JUCE_ENABLE_LIVE_CONSTANT_EDITOR=0
        # Neural network settings
        EIGEN_MPL2_ONLY=1
        RTNEURAL_USE_EIGEN=1)

# Create test executable
add_executable(GuitarToneEmulatorTests
    "${CMAKE_CURRENT_SOURCE_DIR}/Test/FeatureExtractorTest.cpp")

# Add test include directories
target_include_directories(GuitarToneEmulatorTests
    PRIVATE
        "${SOURCE_DIR}"
        "${SOURCE_DIR}/Features"
        "${SOURCE_DIR}/Neural"
        "${SOURCE_DIR}/Visualization"
        "${rtneural_SOURCE_DIR}"
        "${rtneural_SOURCE_DIR}/RTNeural"
        "${eigen_SOURCE_DIR}")

# Link test executable
target_link_libraries(GuitarToneEmulatorTests
    PRIVATE
        GuitarToneEmulator
        GTest::gtest_main
        GTest::gtest
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_core)

# Enable testing
enable_testing()
add_test(NAME GuitarToneEmulatorTests COMMAND GuitarToneEmulatorTests)

# Set test working directory
set_tests_properties(GuitarToneEmulatorTests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# Set output directories
set_target_properties(GuitarToneEmulator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Create ProcessAudio executable
add_executable(ProcessAudio
    "${SOURCE_DIR}/ProcessAudio.cpp")

# Add include directories
target_include_directories(ProcessAudio
    PRIVATE
        "${SOURCE_DIR}"
        "${SOURCE_DIR}/Features"
        "${SOURCE_DIR}/Neural"
        "${SOURCE_DIR}/Visualization"
        "${rtneural_SOURCE_DIR}"
        "${rtneural_SOURCE_DIR}/RTNeural"
        "${eigen_SOURCE_DIR}")

# Link ProcessAudio
target_link_libraries(ProcessAudio
    PRIVATE
        GuitarToneEmulator
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_core)

# Set output directory
set_target_properties(ProcessAudio PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_sources(ProcessAudio PRIVATE
    Source/ProcessAudio.cpp
    Source/Audio/AudioProcessor.cpp
    Source/Features/FeatureExtractor.cpp
    Source/Neural/NeuralProcessor.cpp
)

target_include_directories(ProcessAudio PRIVATE
    Source
    Source/Audio
    Source/Features
    Source/Neural
) 