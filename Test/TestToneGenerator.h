#pragma once

#include <JuceHeader.h>
#include <cmath>
#include <vector>

class TestToneGenerator {
public:
    static void generateTestTone(const juce::String& outputPath, 
                               double sampleRate = 44100.0,
                               double duration = 2.0,
                               bool isReference = false) {
        // Create an audio file
        juce::WavAudioFormat wavFormat;
        std::unique_ptr<juce::AudioFormatWriter> writer;
        juce::File outputFile(outputPath);
        
        if (auto fileStream = std::unique_ptr<juce::FileOutputStream>(outputFile.createOutputStream())) {
            writer.reset(wavFormat.createWriterFor(fileStream.release(), sampleRate, 1, 16, {}, 0));
        }
        
        if (!writer)
            return;
            
        const int numSamples = static_cast<int>(sampleRate * duration);
        juce::AudioBuffer<float> buffer(1, numSamples);
        
        // Generate a guitar-like tone
        float fundamental = 110.0f; // A2 note
        std::vector<float> harmonicAmplitudes = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};
        
        for (int sample = 0; sample < numSamples; ++sample) {
            float time = static_cast<float>(sample) / sampleRate;
            float value = 0.0f;
            
            // Add harmonics
            for (size_t i = 0; i < harmonicAmplitudes.size(); ++i) {
                float freq = fundamental * (i + 1);
                value += harmonicAmplitudes[i] * std::sin(2.0f * juce::MathConstants<float>::pi * freq * time);
            }
            
            // Apply envelope
            float envelope = calculateEnvelope(time, duration);
            
            // Add some characteristic guitar features
            if (isReference) {
                // Reference tone: warm, rich harmonics
                value = std::tanh(value * 2.0f); // Soft clipping
                value *= 0.8f; // Slightly lower volume
            } else {
                // Input tone: brighter, more distorted
                value = std::tanh(value * 4.0f); // More aggressive clipping
                value = applyBrightness(value, 1.5f);
            }
            
            value *= envelope * 0.8f; // Apply envelope and prevent clipping
            buffer.setSample(0, sample, value);
        }
        
        // Write the buffer to file
        writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
    }
    
private:
    static float calculateEnvelope(float time, float duration) {
        const float attackTime = 0.01f;
        const float decayTime = 0.1f;
        const float sustainLevel = 0.7f;
        const float releaseTime = 0.5f;
        
        if (time < attackTime) {
            return time / attackTime;
        } else if (time < attackTime + decayTime) {
            float decayPhase = (time - attackTime) / decayTime;
            return 1.0f - (1.0f - sustainLevel) * decayPhase;
        } else if (time < duration - releaseTime) {
            return sustainLevel;
        } else {
            float releasePhase = (time - (duration - releaseTime)) / releaseTime;
            return sustainLevel * (1.0f - releasePhase);
        }
    }
    
    static float applyBrightness(float input, float amount) {
        // Simple high-frequency enhancement
        float squared = input * input;
        float enhanced = input + (squared - input) * amount;
        return enhanced * 0.8f; // Normalize volume
    }
}; 