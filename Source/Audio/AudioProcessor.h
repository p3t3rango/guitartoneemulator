#pragma once
#include <JuceHeader.h>
#include "../Features/FeatureExtractor.h"
#include "../Neural/NeuralProcessor.h"

namespace GuitarToneEmulator {

class AudioProcessor {
public:
    AudioProcessor();
    
    void prepare(double sampleRate, int blockSize);
    void processBlock(juce::AudioBuffer<float>& buffer);

private:
    void preProcess(juce::AudioBuffer<float>& buffer);
    void postProcess(juce::AudioBuffer<float>& buffer);
    
    FeatureExtractor featureExtractor;
    NeuralProcessor neuralProcessor;
    
    juce::dsp::ProcessorChain<
        juce::dsp::Gain<float>,
        juce::dsp::IIR::Filter<float>,
        juce::dsp::Compressor<float>
    > processingChain;
    
    double sampleRate = 44100.0;
    int blockSize = 2048;
    
    // Thread safety
    juce::SpinLock processLock;
    std::atomic<bool> isProcessing{false};
};

} // namespace GuitarToneEmulator 