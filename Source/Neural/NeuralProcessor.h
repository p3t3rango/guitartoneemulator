#pragma once
#include <JuceHeader.h>
#include "../Features/FeatureExtractor.h"
#include <vector>
#include <memory>

namespace GuitarToneEmulator {

class NeuralProcessor {
public:
    struct ProcessingConfig {
        double sampleRate;
        int blockSize;
        int inputSize;
        int hiddenSize;
        int outputSize;
    };

    class NetworkLayer {
    public:
        NetworkLayer(int inputSize = 1, int outputSize = 1) 
            : inSize(inputSize)
            , outSize(outputSize)
            , weights(inputSize * outputSize)
            , bias(outputSize) {}

        void process(const float* input, float* output, int numSamples);

    private:
        int inSize;
        int outSize;
        std::vector<float> weights;
        std::vector<float> bias;
    };

    NeuralProcessor() 
        : inputLayer(1, 1)
        , hiddenLayer(1, 1)
        , outputLayer(1, 1)
        , fifo(1024) {}
    
    void prepare(const ProcessingConfig& config);
    
    void processFeatures(const FeatureExtractor::Features& features,
                        std::vector<float>& output);

private:
    NetworkLayer inputLayer;
    NetworkLayer hiddenLayer;
    NetworkLayer outputLayer;
    
    ProcessingConfig config;
    
    // Lock-free ring buffer for real-time processing
    juce::AbstractFifo fifo;
    std::vector<float> fifoData;
};

} // namespace GuitarToneEmulator 