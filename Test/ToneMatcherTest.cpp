#include <JuceHeader.h>
#include "../Source/Neural/ToneMatcher.h"

class ToneMatcherTest : public juce::UnitTest
{
public:
    ToneMatcherTest() : juce::UnitTest("ToneMatcher Test") {}

    void runTest() override
    {
        beginTest("Basic ToneMatcher Test");
        
        ToneMatcher matcher;
        matcher.prepare(44100.0, 512);
        
        // Create test buffer
        juce::AudioBuffer<float> buffer(1, 512);
        buffer.clear();
        
        // Generate sine wave
        float frequency = 440.0f; // A4 note
        float sampleRate = 44100.0f;
        for (int i = 0; i < buffer.getNumSamples(); ++i)
        {
            float time = static_cast<float>(i) / sampleRate;
            buffer.setSample(0, i, std::sin(2.0f * juce::MathConstants<float>::pi * frequency * time));
        }
        
        // Process buffer
        matcher.processBlock(buffer);
        
        // Test match amount
        matcher.setMatchAmount(0.5f);
        expect(matcher.getMatchAmount() == 0.5f);
        
        // Test training state
        expect(!matcher.isTraining());
    }
};

static ToneMatcherTest toneMatcherTest; 