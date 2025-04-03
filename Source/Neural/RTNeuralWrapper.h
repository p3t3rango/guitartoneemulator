#pragma once

#include <JuceHeader.h>
#include <RTNeural/RTNeural.h>
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace GuitarToneEmulator {

struct KnobState {
    std::string name;
    float value = 0.0f;
    float min = 0.0f;
    float max = 1.0f;
    bool isEnabled = true;
};

struct NetworkConfig {
    int inputSize = 1;
    int hiddenSize = 64;
    int outputSize = 1;
    int numLayers = 8;
    int receptiveField = 1024;
    float learningRate = 0.001f;
    bool useSkipConnections = true;
    bool useMultiband = false;
    int numBands = 3;
    
    struct {
        bool enabled = false;
        int numFilters = 8;
        int kernelSize = 16;
    } convolution;
};

class RTNeuralWrapper {
public:
    RTNeuralWrapper();
    ~RTNeuralWrapper();
    
    bool initialize(const NetworkConfig& config);
    void prepare();
    void reset();
    
    float processSample(float input);
    void train(float input, float target);
    
    void setKnobState(const std::string& name, float value);
    KnobState getKnobState(const std::string& name) const;
    
    void saveWeights(juce::OutputStream& stream);
    void loadWeights(juce::InputStream& stream);
    void loadBiases(juce::InputStream& stream);
    
    // Load/save model
    bool loadModel(const juce::File& modelFile);
    bool saveModel(const juce::File& modelFile);
    
    // State management
    void setState(const std::vector<float>& hiddenState);
    std::vector<float> getState() const;
    
    // NAM model support
    bool loadNAMModel(const juce::File& modelFile);
    bool saveNAMModel(const juce::File& modelFile);
    
    // Performance optimization
    void optimizeForBlockSize(int blockSize);
    void setUseSIMD(bool useSIMD);

    float process(float input) {
        // Early return if not properly initialized
        if (!initialized) {
            return input;
        }

        // Initialize buffers if needed
        ensureBuffersAllocated();
        
        // Clear buffers
        std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
        
        // Process input
        inputBuffer[0] = input;
        
        try {
            // Forward pass through layers
            if (inputLayer) {
                inputLayer->forward(inputBuffer.data(), inputProcessed.data());
            }
            
            if (outputLayer) {
                outputLayer->forward(inputProcessed.data(), output.data());
            }
            
            return output[0];
        }
        catch (const std::exception& e) {
            // Log error and return input unchanged
            juce::Logger::writeToLog("RTNeuralWrapper process error: " + juce::String(e.what()));
            return input;
        }
    }

    void process(const float* input, float* output, int numSamples) {
        if (!initialized) return;

        for (int i = 0; i < numSamples; ++i) {
            // Process through network
            float currentInput = input[i];
            
            // Forward through layers
            if (inputLayer) {
                inputLayer->forward(&currentInput, inputProcessed.data());
            }
            
            if (outputLayer) {
                outputLayer->forward(inputProcessed.data(), output + i);
            }
        }
    }

    void setWeights(const std::vector<float>& weights) {
        if (!initialized) return;
        
        // Convert weights to 2D vector format
        if (inputLayer) {
            std::vector<std::vector<float>> inputWeights(config.inputSize, std::vector<float>(config.hiddenSize));
            for (int i = 0; i < config.inputSize; ++i) {
                for (int j = 0; j < config.hiddenSize; ++j) {
                    inputWeights[i][j] = weights[i * config.hiddenSize + j];
                }
            }
            inputLayer->setWeights(inputWeights);
        }
        
        if (outputLayer) {
            std::vector<std::vector<float>> outputWeights(config.hiddenSize, std::vector<float>(config.outputSize));
            int offset = config.inputSize * config.hiddenSize;
            for (int i = 0; i < config.hiddenSize; ++i) {
                for (int j = 0; j < config.outputSize; ++j) {
                    outputWeights[i][j] = weights[offset + i * config.outputSize + j];
                }
            }
            outputLayer->setWeights(outputWeights);
        }
    }

    void setBiases(const std::vector<float>& biases) {
        if (!initialized) return;
        
        // Convert biases to vector format
        if (inputLayer) {
            std::vector<float> inputBiases(config.hiddenSize);
            std::copy(biases.begin(), biases.begin() + config.hiddenSize, inputBiases.begin());
            inputLayer->setBias(inputBiases.data());
        }
        
        if (outputLayer) {
            std::vector<float> outputBiases(config.outputSize);
            std::copy(biases.begin() + config.hiddenSize, biases.end(), outputBiases.begin());
            outputLayer->setBias(outputBiases.data());
        }
    }

private:
    struct DilatedLayer {
        std::unique_ptr<RTNeural::Dense<float>> dense1;
        std::unique_ptr<RTNeural::Dense<float>> dense2;
        std::unique_ptr<RTNeural::Activation<float>> activation;
        std::vector<float> buffer;
        std::vector<float> skipConnection;
        int dilation;
    };
    
    struct BandProcessor {
        std::vector<float> lowpassState;
        std::vector<float> highpassState;
        std::vector<float> bandpassState;
        std::vector<float> lowpassCoeffs;
        std::vector<float> highpassCoeffs;
        std::vector<float> bandpassCoeffs;
    };
    
    struct ConvolutionLayer {
        std::vector<std::vector<float>> kernels;
        std::vector<float> biases;
        std::vector<float> state;
    };
    
    void processMultiBand(float input, std::vector<float>& bandOutputs);
    void applyConvolution(const std::vector<float>& input, std::vector<float>& output);
    void updateFrequencyResponse();
    
    float processLowpass(float input, std::vector<float>& state);
    float processHighpass(float input, std::vector<float>& state);
    float processBandpass(float input, std::vector<float>& state, int band);
    
    NetworkConfig config;
    std::unique_ptr<RTNeural::Dense<float>> inputLayer;
    std::vector<std::unique_ptr<DilatedLayer>> layers;
    std::unique_ptr<RTNeural::Dense<float>> outputLayer;
    
    std::vector<float> inputBuffer;
    std::vector<float> inputProcessed;
    std::vector<float> output;
    std::vector<float> hiddenState;
    std::vector<float> outputBuffer;
    std::vector<float> trainingBuffer;
    std::vector<float> freqResponse;
    
    std::vector<BandProcessor> bandProcessors;
    std::vector<ConvolutionLayer> convLayers;
    
    std::map<std::string, KnobState> knobStates;
    
    int bufferPos = 0;
    bool initialized = false;

    // Helper functions
    void initializeLayers();
    void cleanupLayers();
    void updateWeights(const std::vector<float>& gradients, float learningRate);
    bool parseNAMModel(const juce::String& modelData);
    juce::String generateNAMModel() const;
    void ensureBuffersAllocated();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RTNeuralWrapper)
};

} // namespace GuitarToneEmulator 