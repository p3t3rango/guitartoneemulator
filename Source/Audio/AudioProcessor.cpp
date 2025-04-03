#include "AudioProcessor.h"

namespace GuitarToneEmulator {

AudioProcessor::AudioProcessor() {
    // Initialize processing chain
    auto& gain = processingChain.get<0>();
    gain.setGainDecibels(0.0f);
    
    auto& filter = processingChain.get<1>();
    filter.setType(juce::dsp::IIR::Filter<float>::Type::lowpass);
    filter.setCutoffFrequency(20000.0f);
    
    auto& compressor = processingChain.get<2>();
    compressor.setThreshold(-20.0f);
    compressor.setRatio(4.0f);
    compressor.setAttack(5.0f);
    compressor.setRelease(100.0f);
}

void AudioProcessor::processBlock(juce::AudioBuffer<float>& buffer) {
    const juce::SpinLock::ScopedLockType lock(processLock);
    if (isProcessing.exchange(true)) return;
    
    // Extract features
    auto features = featureExtractor.extractFeatures(buffer);
    
    // Process through neural network
    std::vector<float> neuralOutput(buffer.getNumSamples());
    neuralProcessor.processFeatures(features, neuralOutput);
    
    // Apply neural output to audio
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
        float* channelData = buffer.getWritePointer(channel);
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
            // Blend original with processed signal
            channelData[sample] = 0.7f * channelData[sample] + 0.3f * neuralOutput[sample];
        }
    }
    
    // Process through chain
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    processingChain.process(context);
    
    isProcessing.store(false);
}

void AudioProcessor::prepare(double newSampleRate, int newBlockSize) {
    sampleRate = newSampleRate;
    blockSize = newBlockSize;
    
    // Prepare DSP chain
    juce::dsp::ProcessSpec spec{
        sampleRate,
        static_cast<uint32>(blockSize),
        static_cast<uint32>(2) // stereo
    };
    
    processingChain.prepare(spec);
    
    // Initialize processors
    featureExtractor.prepare(sampleRate, blockSize);
    
    NeuralProcessor::ProcessingConfig config;
    config.sampleRate = sampleRate;
    config.blockSize = blockSize;
    config.inputSize = 13;  // Total number of features
    config.hiddenSize = 32;
    config.outputSize = blockSize;
    neuralProcessor.prepare(config);
}

} // namespace GuitarToneEmulator 