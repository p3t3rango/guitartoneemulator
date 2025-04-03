#include "NeuralProcessor.h"

namespace GuitarToneEmulator {

void NeuralProcessor::NetworkLayer::process(const float* input, float* output, int numSamples) {
    // Simple feed-forward processing
    for (int i = 0; i < outSize; ++i) {
        float sum = bias[i];
        for (int j = 0; j < inSize; ++j) {
            sum += input[j] * weights[j * outSize + i];
        }
        // ReLU activation
        output[i] = std::max(0.0f, sum);
    }
}

void NeuralProcessor::prepare(const ProcessingConfig& newConfig) {
    config = newConfig;
    inputLayer = NetworkLayer(config.inputSize, config.hiddenSize);
    hiddenLayer = NetworkLayer(config.hiddenSize, config.outputSize);
}

void NeuralProcessor::processFeatures(const FeatureExtractor::Features& features, std::vector<float>& output) {
    // Convert features to vector for processing
    std::vector<float> featureVector;
    featureVector.reserve(config.inputSize);
    
    // Add spectral features
    featureVector.insert(featureVector.end(), features.spectral.begin(), features.spectral.end());
    
    // Add temporal features
    featureVector.insert(featureVector.end(), features.temporal.begin(), features.temporal.end());
    
    // Add modulation features
    featureVector.insert(featureVector.end(), features.modulation.begin(), features.modulation.end());
    
    // Ensure output vector is the right size
    output.resize(config.outputSize);
    
    // Process through network layers
    std::vector<float> hiddenOutput(config.hiddenSize);
    inputLayer.process(featureVector.data(), hiddenOutput.data(), featureVector.size());
    hiddenLayer.process(hiddenOutput.data(), output.data(), hiddenOutput.size());
}

} // namespace GuitarToneEmulator 