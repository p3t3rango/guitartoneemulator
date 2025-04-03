#include <gtest/gtest.h>
#include <JuceHeader.h>
#include "../Source/Features/FeatureExtractor.h"
#include "../Source/Neural/ToneMatcher.h"
#include "TestToneGenerator.h"
#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <memory>
#include <iostream>

using namespace GuitarToneEmulator;

class FeatureExtractorTest : public ::testing::Test {
protected:
    void SetUp() override {
        featureExtractor = std::make_unique<FeatureExtractor>();
        featureExtractor->prepare(44100.0, 2048);
        toneMatcher = std::make_unique<ToneMatcher>();
        toneMatcher->prepareToPlay(44100.0, 2048);
        
        // Use existing audio files
        testDir = juce::File::getCurrentWorkingDirectory().getChildFile("Test");
        referenceFile = testDir.getChildFile("reference.wav");
        inputFile = testDir.getChildFile("input.wav");
        outputFile = testDir.getChildFile("output.wav");
        
        // Verify files exist
        if (!referenceFile.existsAsFile() || !inputFile.existsAsFile()) {
            std::cout << "\nError: Please ensure reference.wav and input.wav exist in the current directory." << std::endl;
            std::cout << "Expected paths:" << std::endl;
            std::cout << "Reference: " << referenceFile.getFullPathName() << std::endl;
            std::cout << "Input: " << inputFile.getFullPathName() << std::endl;
        }
    }

    void TearDown() override {
        featureExtractor.reset();
        toneMatcher.reset();
    }

    std::unique_ptr<FeatureExtractor> featureExtractor;
    std::unique_ptr<ToneMatcher> toneMatcher;
    juce::File testDir;
    juce::File referenceFile;
    juce::File inputFile;
    juce::File outputFile;
};

TEST_F(FeatureExtractorTest, MFCCComputation) {
    // Create a test signal (sine wave with harmonics)
    juce::AudioBuffer<float> buffer(1, 2048);
    const float fundamental = 440.0f;
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float t = static_cast<float>(i) / 44100.0f;
        // Fundamental + first two harmonics
        buffer.setSample(0, i, 0.5f * std::sin(2.0f * M_PI * fundamental * t) +
                               0.3f * std::sin(4.0f * M_PI * fundamental * t) +
                               0.2f * std::sin(6.0f * M_PI * fundamental * t));
    }

    // Process the buffer and get features
    featureExtractor->processBlock(buffer);
    const auto& features = featureExtractor->analyzeTone(buffer);

    // Verify features
    EXPECT_GT(features.spectral.harmonicAmplitudes.size(), 0) << "No harmonic amplitudes found";
    EXPECT_GT(features.spectral.harmonicPhases.size(), 0) << "No harmonic phases found";
    
    if (features.spectral.harmonicAmplitudes.size() > 0) {
        std::cout << "\nHarmonic Analysis Results:" << std::endl;
        std::cout << "Number of harmonics: " << features.spectral.harmonicAmplitudes.size() << std::endl;
        std::cout << "First harmonic amplitude: " << features.spectral.harmonicAmplitudes[0] << std::endl;
        std::cout << "First harmonic phase: " << features.spectral.harmonicPhases[0] << std::endl;
    }
}

TEST_F(FeatureExtractorTest, OnsetDetection) {
    // Create a test signal with clear onsets
    juce::AudioBuffer<float> buffer(1, 4096);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        if (i < 2048) {
            buffer.setSample(0, i, std::sin(2.0f * M_PI * 440.0f * i / 44100.0f));
        } else {
            buffer.setSample(0, i, std::sin(2.0f * M_PI * 880.0f * i / 44100.0f));
        }
    }

    // Process the buffer
    featureExtractor->processBlock(buffer);

    // Get temporal features
    const auto& features = featureExtractor->analyzeTone(buffer);

    // Verify onset detection
    EXPECT_GT(features.temporal.transients.size(), 0);
    EXPECT_LT(features.temporal.transients[0].position, 2048.0f / 44100.0f); // First onset should be before the change
}

TEST_F(FeatureExtractorTest, SpectralFeatures) {
    // Create a test signal with multiple frequency components
    juce::AudioBuffer<float> buffer(1, 2048);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float sample = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / 44100.0f) +
                      0.3f * std::sin(2.0f * M_PI * 880.0f * i / 44100.0f) +
                      0.2f * std::sin(2.0f * M_PI * 1320.0f * i / 44100.0f);
        buffer.setSample(0, i, sample);
    }

    // Process the buffer
    featureExtractor->processBlock(buffer);

    // Get spectral features
    const auto& features = featureExtractor->analyzeTone(buffer);

    // Verify spectral features
    EXPECT_GT(features.spectral.centroid, 0.0f);
    EXPECT_GT(features.spectral.spread, 0.0f);
    EXPECT_GE(features.spectral.flatness, 0.0f);
    EXPECT_LE(features.spectral.flatness, 1.0f);
}

TEST_F(FeatureExtractorTest, DynamicFeatures) {
    // Create a test signal with varying amplitude
    juce::AudioBuffer<float> buffer(1, 2048);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float amplitude = 0.5f + 0.5f * std::sin(2.0f * M_PI * 2.0f * i / 2048.0f);
        buffer.setSample(0, i, amplitude * std::sin(2.0f * M_PI * 440.0f * i / 44100.0f));
    }

    // Process the buffer
    featureExtractor->processBlock(buffer);

    // Get dynamic features
    const auto& features = featureExtractor->analyzeTone(buffer);

    // Verify dynamic features
    EXPECT_GT(features.temporal.rms, 0.0f);
    EXPECT_GT(features.temporal.zeroCrossingRate, 0);
}

TEST_F(FeatureExtractorTest, TemporalFeatures) {
    // Create a test signal with varying zero-crossing rate
    juce::AudioBuffer<float> buffer(1, 2048);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float frequency = 440.0f + 220.0f * std::sin(2.0f * M_PI * 1.0f * i / 2048.0f);
        buffer.setSample(0, i, std::sin(2.0f * M_PI * frequency * i / 44100.0f));
    }

    // Process the buffer
    featureExtractor->processBlock(buffer);

    // Get temporal features
    const auto& features = featureExtractor->analyzeTone(buffer);

    // Verify temporal features
    EXPECT_GT(features.temporal.zeroCrossingRate, 0);
    EXPECT_GT(features.temporal.rms, 0.0f);
    EXPECT_LT(features.temporal.rms, 1.0f);
}

TEST_F(FeatureExtractorTest, ToneMatching) {
    // Print test setup information
    std::cout << "\n=== Test Setup ===" << std::endl;
    std::cout << "Test directory: " << testDir.getFullPathName() << std::endl;
    std::cout << "Reference file: " << referenceFile.getFullPathName() << std::endl;
    std::cout << "Input file: " << inputFile.getFullPathName() << std::endl;
    std::cout << "Output file: " << outputFile.getFullPathName() << std::endl;

    // Load reference and input audio files
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();
    
    std::unique_ptr<juce::AudioFormatReader> referenceReader(formatManager.createReaderFor(referenceFile));
    std::unique_ptr<juce::AudioFormatReader> inputReader(formatManager.createReaderFor(inputFile));
    
    ASSERT_NE(referenceReader, nullptr) << "Failed to load reference file";
    ASSERT_NE(inputReader, nullptr) << "Failed to load input file";
    
    const double sampleRate = referenceReader->sampleRate;
    const int numChannels = referenceReader->numChannels;
    
    std::cout << "\n=== Audio Analysis ===" << std::endl;
    std::cout << "Sample rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "Reference duration: " << referenceReader->lengthInSamples / sampleRate << " seconds" << std::endl;
    std::cout << "Input duration: " << inputReader->lengthInSamples / sampleRate << " seconds" << std::endl;

    // Load audio into buffers
    juce::AudioBuffer<float> referenceBuffer(numChannels, static_cast<int>(referenceReader->lengthInSamples));
    referenceReader->read(&referenceBuffer, 0, referenceBuffer.getNumSamples(), 0, true, true);

    // Only load first 5 seconds of input audio for testing
    const int maxInputSamples = static_cast<int>(5.0 * sampleRate);
    juce::AudioBuffer<float> inputBuffer(numChannels, maxInputSamples);
    inputReader->read(&inputBuffer, 0, maxInputSamples, 0, true, true);

    // Analyze reference tone features
    auto referenceFeatures = featureExtractor->analyzeTone(referenceBuffer);
    std::cout << "\n=== Reference Tone Features ===" << std::endl;
    std::cout << "Spectral centroid: " << referenceFeatures.spectral.centroid << std::endl;
    std::cout << "Harmonic content: " << referenceFeatures.spectral.harmonicContent << std::endl;
    std::cout << "Brightness: " << referenceFeatures.brightness << std::endl;
    std::cout << "Warmth: " << referenceFeatures.warmth << std::endl;
    std::cout << "Number of harmonics: " << referenceFeatures.spectral.harmonicAmplitudes.size() << std::endl;

    // Start tone matching training with the buffer
    std::cout << "\n=== Training Phase ===" << std::endl;
    toneMatcher->startTraining(referenceBuffer);
    ASSERT_TRUE(toneMatcher->isTraining()) << "Training failed to start";

    // Process training in blocks
    const int blockSize = 2048;
    juce::AudioBuffer<float> block(referenceBuffer.getNumChannels(), blockSize);
    
    for (int offset = 0; offset < referenceBuffer.getNumSamples(); offset += blockSize) {
        int numSamples = std::min(blockSize, referenceBuffer.getNumSamples() - offset);
        block.setSize(referenceBuffer.getNumChannels(), numSamples, true, true, true);
        
        for (int ch = 0; ch < referenceBuffer.getNumChannels(); ++ch) {
            block.copyFrom(ch, 0, referenceBuffer, ch, offset, numSamples);
        }
        
        toneMatcher->processTraining(block);
    }

    // Stop training
    toneMatcher->stopTraining();
    ASSERT_FALSE(toneMatcher->isTraining()) << "Training failed to stop";

    // Process input audio through tone matcher
    std::cout << "\n=== Processing Phase ===" << std::endl;
    juce::AudioBuffer<float> processedBuffer(inputBuffer.getNumChannels(), inputBuffer.getNumSamples());
    processedBuffer.makeCopyOf(inputBuffer);

    for (int offset = 0; offset < processedBuffer.getNumSamples(); offset += blockSize) {
        int numSamples = std::min(blockSize, processedBuffer.getNumSamples() - offset);
        block.setSize(processedBuffer.getNumChannels(), numSamples, true, true, true);
        
        for (int ch = 0; ch < processedBuffer.getNumChannels(); ++ch) {
            block.copyFrom(ch, 0, processedBuffer, ch, offset, numSamples);
        }
        
        toneMatcher->processBlock(block);
        
        for (int ch = 0; ch < processedBuffer.getNumChannels(); ++ch) {
            processedBuffer.copyFrom(ch, offset, block, ch, 0, numSamples);
        }
    }

    // Analyze processed features
    auto processedFeatures = featureExtractor->analyzeTone(processedBuffer);
    std::cout << "\n=== Processed Tone Features ===" << std::endl;
    std::cout << "Spectral centroid: " << processedFeatures.spectral.centroid << std::endl;
    std::cout << "Harmonic content: " << processedFeatures.spectral.harmonicContent << std::endl;
    std::cout << "Brightness: " << processedFeatures.brightness << std::endl;
    std::cout << "Warmth: " << processedFeatures.warmth << std::endl;
    std::cout << "Number of harmonics: " << processedFeatures.spectral.harmonicAmplitudes.size() << std::endl;

    // Calculate feature differences
    float centroidDiff = std::abs(processedFeatures.spectral.centroid - referenceFeatures.spectral.centroid);
    float harmonicDiff = std::abs(processedFeatures.spectral.harmonicContent - referenceFeatures.spectral.harmonicContent);
    float brightnessDiff = std::abs(processedFeatures.brightness - referenceFeatures.brightness);
    float warmthDiff = std::abs(processedFeatures.warmth - referenceFeatures.warmth);

    std::cout << "\n=== Feature Differences ===" << std::endl;
    std::cout << "Centroid difference: " << centroidDiff << std::endl;
    std::cout << "Harmonic content difference: " << harmonicDiff << std::endl;
    std::cout << "Brightness difference: " << brightnessDiff << std::endl;
    std::cout << "Warmth difference: " << warmthDiff << std::endl;

    // Save processed audio
    std::unique_ptr<juce::AudioFormatWriter> writer;
    if (auto fileStream = std::unique_ptr<juce::FileOutputStream>(outputFile.createOutputStream())) {
        writer.reset(formatManager.findFormatForFileExtension("wav")->createWriterFor(
            fileStream.release(), sampleRate, processedBuffer.getNumChannels(), 16, {}, 0));
    }
    
    ASSERT_NE(writer, nullptr) << "Failed to create output file writer";
    writer->writeFromAudioSampleBuffer(processedBuffer, 0, processedBuffer.getNumSamples());

    // Verify the results
    const float maxFeatureDiff = 0.3f; // Maximum allowed difference in features
    EXPECT_LT(centroidDiff, maxFeatureDiff) << "Spectral centroid difference too large";
    EXPECT_LT(harmonicDiff, maxFeatureDiff) << "Harmonic content difference too large";
    EXPECT_LT(brightnessDiff, maxFeatureDiff) << "Brightness difference too large";
    EXPECT_LT(warmthDiff, maxFeatureDiff) << "Warmth difference too large";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 