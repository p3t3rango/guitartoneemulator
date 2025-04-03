#include "ToneMatcher.h"
#include "../Features/FeatureExtractor.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace GuitarToneEmulator {

ToneMatcher::ToneMatcher()
    : training(false)
    , trainingSteps(0)
    , currentLoss(0.0f)
    , sampleRate(44100.0)
    , blockSize(512)
{
    // Initialize configuration
    config.lstm.numLayers = 2;
    config.lstm.hiddenSize = 64;
    config.lstm.useAttention = true;
    
    config.conv.numLayers = 2;
    config.conv.filterSize = 3;
    config.conv.dilationRates = {1, 2, 4, 8};
    
    config.multiband.numBands = 8;
    config.multiband.minFreq = 20.0f;
    config.multiband.maxFreq = 20000.0f;
    
    config.learningRate = 0.001f;
    config.batchSize = 32;
    
    // Calculate input size based on features
    const int baseFeatures = 13;  // 5 spectral + 5 temporal + 3 modulation
    const int harmonicFeatures = 20;  // 10 harmonics * 2 (amplitude + phase)
    config.inputFeatureSize = baseFeatures + harmonicFeatures;
    config.outputSize = config.inputFeatureSize;
    
    // Initialize normalization parameters with reasonable defaults
    normParams.mean.resize(config.inputFeatureSize, 0.0f);
    normParams.std.resize(config.inputFeatureSize, 1.0f);
    
    // Set specific means and stds for different feature types
    // Spectral features (0-4)
    for (int i = 0; i < 5; ++i)
    {
        normParams.mean[i] = 0.0f;
        normParams.std[i] = 1000.0f;  // Typical range for spectral features
    }
    
    // Temporal features (5-9)
    for (int i = 5; i < 10; ++i)
    {
        normParams.mean[i] = 0.0f;
        normParams.std[i] = 1.0f;  // Most temporal features are already normalized
    }
    
    // Modulation features (10-12)
    for (int i = 10; i < 13; ++i)
    {
        normParams.mean[i] = 0.0f;
        normParams.std[i] = 1.0f;
    }
    
    // Harmonic features (13-32)
    for (int i = 13; i < config.inputFeatureSize; i += 2)
    {
        // Amplitude
        normParams.mean[i] = 0.0f;
        normParams.std[i] = 100.0f;  // Typical range for harmonic amplitudes
        
        // Phase
        normParams.mean[i + 1] = 0.0f;
        normParams.std[i + 1] = juce::MathConstants<float>::pi;  // Phases are in [-π, π]
    }
    
    // Initialize network
    initializeNetwork();
    
    // Create feature extractor
    featureExtractor = std::make_unique<FeatureExtractor>();
    
    // Initialize random number generator
    std::random_device rd;
    gen.seed(rd());
    
    // Initialize LSTM layers
    lstmLayers.resize(config.lstm.numLayers);
    for (auto& layer : lstmLayers) {
        layer.hiddenState.resize(config.lstm.hiddenSize);
        layer.cellState.resize(config.lstm.hiddenSize);
        std::fill(layer.hiddenState.begin(), layer.hiddenState.end(), 0.0f);
        std::fill(layer.cellState.begin(), layer.cellState.end(), 0.0f);
    }
}

void ToneMatcher::prepareToPlay(double newSampleRate, int newBlockSize)
{
    sampleRate = newSampleRate;
    blockSize = newBlockSize;
    
    // Create and prepare feature extractor
    featureExtractor = std::make_unique<FeatureExtractor>();
    featureExtractor->prepare(sampleRate, blockSize);
    
    // Prepare audio processing components
    audioProcessing.pickPositionDelay.prepare({sampleRate, static_cast<uint32>(blockSize), 1});
    audioProcessing.pickPositionDelay.setDelay(0.0f);
    
    audioProcessing.lowShelf.setCoefficients(juce::IIRCoefficients::makeLowShelf(sampleRate, 500.0f, 1.0f, 1.0f));
    audioProcessing.highShelf.setCoefficients(juce::IIRCoefficients::makeHighShelf(sampleRate, 2000.0f, 1.0f, 1.0f));
    
    // Configure multiband filters
    const float bandsPerOctave = std::log2(config.multiband.maxFreq / config.multiband.minFreq);
    const float freqRatio = std::pow(2.0f, bandsPerOctave / config.multiband.numBands);
    
    multibandProcessor.filters.resize(config.multiband.numBands);
    multibandProcessor.weights.resize(config.multiband.numBands, 1.0f / config.multiband.numBands);
    
    float freq = config.multiband.minFreq;
    for (size_t i = 0; i < config.multiband.numBands; ++i)
    {
        multibandProcessor.filters[i].setCoefficients(
            juce::IIRCoefficients::makeBandPass(sampleRate, freq, freq / 4.0f)
        );
        freq *= freqRatio;
    }
    
    // Reset network states
    resetStates();
}

void ToneMatcher::processBlock(juce::AudioBuffer<float>& buffer)
{
    if (!featureExtractor || buffer.getNumSamples() == 0) {
        return;
    }
    
    // Process spectral features
    processSpectral(buffer);
    
    // Apply final gain adjustment
    buffer.applyGain(audioProcessing.outputGain);
}

void ToneMatcher::processTraining(const juce::AudioBuffer<float>& buffer)
{
    if (!training || !featureExtractor)
        return;

    try
    {
        // Extract features from the current buffer
        featureExtractor->processBlock(buffer);
        
        // Get current features
        const float* data = buffer.getReadPointer(0);
        std::vector<float> inputVector = featuresToVector(featureExtractor->analyzeTone(buffer));
        normalizeFeatures(inputVector);
        
        // Train the network
        if (!referenceVector.empty() && inputVector.size() == config.inputFeatureSize)
        {
            train(inputVector, referenceVector);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in processTraining: " << e.what() << std::endl;
        stopTraining();
    }
}

void ToneMatcher::processBatch()
{
    if (batchQueue.empty())
        return;
        
    std::vector<float> batchLoss;
    
    while (!batchQueue.empty() && batchLoss.size() < config.batchSize)
    {
        auto& batch = batchQueue.front();
        std::vector<float> output = batch.inputFeatures[0];
        
        try {
            // Forward pass through network layers
            if (config.conv.numLayers > 0)
            {
                std::vector<float> tempBuffer(output.size());
                processConvolutionalLayer(output, tempBuffer);
                output = tempBuffer;
            }
            
            if (config.lstm.numLayers > 0)
            {
                std::vector<float> tempBuffer(output.size());
                processLSTMLayer(output, tempBuffer);
                output = tempBuffer;
            }
            
            if (config.multiband.numBands > 0)
            {
                std::vector<float> tempBuffer(output.size());
                processMultibandLayer(output, tempBuffer);
                output = tempBuffer;
            }
            
            // Calculate losses
            float spectralLoss = calculateSpectralLoss(output, batch.targetFeatures[0]);
            float temporalLoss = calculateTemporalLoss(output, batch.targetFeatures[0]);
            float totalLoss = spectralLoss + temporalLoss;
            
            batchLoss.push_back(totalLoss);
            
            // Calculate gradients and update weights
            std::vector<float> gradients(output.size());
            for (size_t i = 0; i < output.size(); ++i)
            {
                gradients[i] = output[i] - batch.targetFeatures[0][i];
            }
            
            updateNetworkWeights(gradients);
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing batch: " << e.what() << std::endl;
        }
        
        batchQueue.pop();
    }
    
    // Update training statistics
    if (!batchLoss.empty())
    {
        currentLoss = std::accumulate(batchLoss.begin(), batchLoss.end(), 0.0f) / batchLoss.size();
        trainingSteps++;
        
        std::cout << "Training step " << trainingSteps 
                  << ", Loss: " << currentLoss << std::endl;
    }
}

void ToneMatcher::processConvolutionalLayer(const std::vector<float>& input, std::vector<float>& output)
{
    output.resize(input.size());
    
    // Apply convolution with dilation
    for (size_t i = 0; i < convLayers.size(); ++i)
    {
        auto& layer = convLayers[i];
        const int dilation = config.conv.dilationRates[i % config.conv.dilationRates.size()];
        
        // Process through convolutional layer
        std::vector<float> residual = input;
        for (size_t j = 0; j < layer.filters.size(); ++j)
        {
            const auto& filter = layer.filters[j];
            const auto& bias = layer.biases[j];
            
            // Apply convolution
            for (size_t k = 0; k < input.size(); ++k)
            {
                float sum = bias;
                for (size_t m = 0; m < filter.size(); ++m)
                {
                    const size_t idx = k + m * dilation;
                    if (idx < input.size())
                        sum += filter[m] * input[idx];
                }
                output[k] = sigmoid(sum);
            }
        }
        
        // Add residual connection if enabled
        if (i < config.conv.numLayers - 1)  // Don't add residual on last layer
        {
            for (size_t j = 0; j < output.size(); ++j)
                output[j] += residual[j];
        }
    }
}

void ToneMatcher::processAttention(const std::vector<float>& input, std::vector<float>& output, AttentionLayer& attention)
{
    const int seqLength = input.size() / config.lstm.hiddenSize;
    const int hiddenSize = config.lstm.hiddenSize;
    
    // Reshape input to sequence
    std::vector<std::vector<float>> sequence(seqLength, std::vector<float>(hiddenSize));
    for (int t = 0; t < seqLength; ++t)
    {
        for (int h = 0; h < hiddenSize; ++h)
        {
            sequence[t][h] = input[t * hiddenSize + h];
        }
    }
    
    // Calculate query, key, and value matrices
    std::vector<std::vector<float>> query(seqLength, std::vector<float>(hiddenSize));
    std::vector<std::vector<float>> key(seqLength, std::vector<float>(hiddenSize));
    std::vector<std::vector<float>> value(seqLength, std::vector<float>(hiddenSize));
    
    // Linear transformations
    for (int t = 0; t < seqLength; ++t)
    {
        for (int h = 0; h < hiddenSize; ++h)
        {
            float q = 0.0f, k = 0.0f, v = 0.0f;
            for (int i = 0; i < hiddenSize; ++i)
            {
                q += sequence[t][i] * attention.queryWeights[h][i];
                k += sequence[t][i] * attention.keyWeights[h][i];
                v += sequence[t][i] * attention.valueWeights[h][i];
            }
            query[t][h] = q + attention.biases[h];
            key[t][h] = k + attention.biases[h + hiddenSize];
            value[t][h] = v + attention.biases[h + 2 * hiddenSize];
        }
    }
    
    // Calculate attention scores
    std::vector<std::vector<float>> scores(seqLength, std::vector<float>(seqLength));
    float maxScore = -std::numeric_limits<float>::infinity();
    
    for (int t1 = 0; t1 < seqLength; ++t1)
    {
        for (int t2 = 0; t2 < seqLength; ++t2)
        {
            float score = 0.0f;
            for (int h = 0; h < hiddenSize; ++h)
            {
                score += query[t1][h] * key[t2][h];
            }
            score *= attention.scalingFactor;
            scores[t1][t2] = score;
            maxScore = std::max(maxScore, score);
        }
    }
    
    // Apply softmax
    std::vector<std::vector<float>> attention_weights(seqLength, std::vector<float>(seqLength));
    for (int t1 = 0; t1 < seqLength; ++t1)
    {
        float sum = 0.0f;
        for (int t2 = 0; t2 < seqLength; ++t2)
        {
            attention_weights[t1][t2] = std::exp(scores[t1][t2] - maxScore);
            sum += attention_weights[t1][t2];
        }
        for (int t2 = 0; t2 < seqLength; ++t2)
        {
            attention_weights[t1][t2] /= sum;
        }
    }
    
    // Calculate weighted sum
    std::vector<std::vector<float>> context(seqLength, std::vector<float>(hiddenSize));
    for (int t1 = 0; t1 < seqLength; ++t1)
    {
        for (int h = 0; h < hiddenSize; ++h)
        {
            float sum = 0.0f;
            for (int t2 = 0; t2 < seqLength; ++t2)
            {
                sum += attention_weights[t1][t2] * value[t2][h];
            }
            context[t1][h] = sum;
        }
    }
    
    // Flatten output
    output.resize(seqLength * hiddenSize);
    for (int t = 0; t < seqLength; ++t)
    {
        for (int h = 0; h < hiddenSize; ++h)
        {
            output[t * hiddenSize + h] = context[t][h];
        }
    }
}

void ToneMatcher::processLSTMLayer(const std::vector<float>& input, std::vector<float>& output)
{
    try {
        if (input.empty()) {
            std::cerr << "Error: Empty input vector in processLSTMLayer" << std::endl;
            return;
        }

        output.resize(input.size());
        std::vector<float> layerInput = input;

        for (auto& layer : lstmLayers)
        {
            std::vector<float> layerOutput(layerInput.size());
            layer.processLSTM(layerInput.data(), layerOutput.data(), layerInput.size());
            layerInput = layerOutput;
        }

        output = layerInput;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in processLSTMLayer: " << e.what() << std::endl;
        output.resize(input.size(), 0.0f);  // Fill with zeros in case of error
    }
}

void ToneMatcher::processMultibandLayer(const std::vector<float>& input, std::vector<float>& output)
{
    const int numBands = config.multiband.numBands;
    output.resize(input.size());
    std::fill(output.begin(), output.end(), 0.0f);
    
    // Process each frequency band
    std::vector<float> bandOutput(input.size());
    for (int band = 0; band < numBands; ++band)
    {
        // Apply bandpass filter
        auto& filter = multibandProcessor.filters[band];
        bandOutput = input;
        filter.processSamples(bandOutput.data(), bandOutput.size());
        
        // Apply band weight and accumulate
        const float weight = multibandProcessor.weights[band];
        for (size_t i = 0; i < output.size(); ++i)
            output[i] += weight * bandOutput[i];
    }
}

float ToneMatcher::calculateSpectralLoss(const std::vector<float>& predicted, const std::vector<float>& target)
{
    float loss = 0.0f;
    
    // Calculate weighted MSE for spectral features
    for (size_t i = 0; i < predicted.size(); ++i)
    {
        float diff = predicted[i] - target[i];
        float weight = 1.0f;
        
        // Give higher weight to important spectral features
        if (i < 5) // First 5 features are spectral
            weight = 2.0f;
            
        loss += weight * diff * diff;
    }
    
    return loss / predicted.size();
}

float ToneMatcher::calculateTemporalLoss(const std::vector<float>& predicted, const std::vector<float>& target)
{
    float loss = 0.0f;
    
    // Calculate weighted MSE for temporal features
    for (size_t i = 5; i < predicted.size(); ++i)
    {
        float diff = predicted[i] - target[i];
        float weight = 1.0f;
        
        // Give higher weight to envelope features
        if (i >= 8 && i <= 11) // ADSR features
            weight = 1.5f;
            
        loss += weight * diff * diff;
    }
    
    return loss / (predicted.size() - 5);
}

void ToneMatcher::updateNetworkWeights(const std::vector<float>& gradients)
{
    const float learningRate = config.learningRate;
    size_t offset = 0;
    
    // Update convolutional layer weights
    for (auto& layer : convLayers)
    {
        for (auto& filter : layer.filters)
        {
            for (float& weight : filter)
            {
                weight -= learningRate * gradients[offset];
                ++offset;
            }
        }
        
        for (float& bias : layer.biases)
        {
            bias -= learningRate * gradients[offset];
            ++offset;
        }
    }
    
    // Update LSTM layer weights
    for (auto& layer : lstmLayers)
    {
        // Update input weights and biases
        for (float& weight : layer.inputWeights)
        {
            weight -= learningRate * gradients[offset];
            ++offset;
        }
        
        for (float& bias : layer.inputBias)
        {
            bias -= learningRate * gradients[offset];
            ++offset;
        }
        
        // Update forget weights and biases
        for (float& weight : layer.forgetWeights)
        {
            weight -= learningRate * gradients[offset];
            ++offset;
        }
        
        for (float& bias : layer.forgetBias)
        {
            bias -= learningRate * gradients[offset];
            ++offset;
        }
        
        // Update cell weights and biases
        for (float& weight : layer.cellWeights)
        {
            weight -= learningRate * gradients[offset];
            ++offset;
        }
        
        for (float& bias : layer.cellBias)
        {
            bias -= learningRate * gradients[offset];
            ++offset;
        }
        
        // Update output weights and biases
        for (float& weight : layer.outputWeights)
        {
            weight -= learningRate * gradients[offset];
            ++offset;
        }
        
        for (float& bias : layer.outputBias)
        {
            bias -= learningRate * gradients[offset];
            ++offset;
        }
        
        // Update recurrent weights
        for (float& weight : layer.recurrentWeights)
        {
            weight -= learningRate * gradients[offset];
            ++offset;
        }
    }
    
    // Update multiband weights
    for (float& weight : multibandProcessor.weights)
    {
        weight -= learningRate * gradients[offset];
        ++offset;
    }
}

void ToneMatcher::startTraining(const juce::AudioBuffer<float>& referenceBuffer)
{
    if (!featureExtractor)
        return;

    try
    {
        // Extract features from reference buffer
        featureExtractor->processBlock(referenceBuffer);
        
        // Pre-allocate the reference vector with the expected size
        referenceVector.clear();
        referenceVector.reserve(config.inputFeatureSize);
        
        // Convert to vector and normalize
        referenceVector = featuresToVector(featureExtractor->analyzeTone(referenceBuffer));
        if (referenceVector.size() != config.inputFeatureSize)
        {
            std::cerr << "Error: Feature vector size mismatch" << std::endl;
            return;
        }
        normalizeFeatures(referenceVector);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in startTraining: " << e.what() << std::endl;
    }
}

void ToneMatcher::stopTraining()
{
    training = false;
    trainingSteps = 0;
    currentLoss = 0.0f;
    
    // Reset all states
    resetStates();
}

void ToneMatcher::resetTraining()
{
    // Reset LSTM states
    if (config.lstm.numLayers > 0)
    {
        for (auto& layer : lstmLayers)
        {
            std::fill(layer.hiddenState.begin(), layer.hiddenState.end(), 0.0f);
            std::fill(layer.cellState.begin(), layer.cellState.end(), 0.0f);
        }
    }

    // Reset multiband filters
    if (config.multiband.numBands > 0)
    {
        for (auto& filter : multibandProcessor.filters)
        {
            filter.reset();
        }
    }
}

void ToneMatcher::saveModel(const juce::String& path)
{
    // TODO: Implement model saving
}

void ToneMatcher::loadModel(const juce::String& path)
{
    // TODO: Implement model loading
}

void ToneMatcher::initializeNetwork()
{
    // Initialize LSTM layers
    lstmLayers.resize(config.lstm.numLayers);
    for (auto& layer : lstmLayers)
    {
        const size_t stateSize = config.lstm.hiddenSize;
        layer.hiddenState.resize(stateSize, 0.0f);
        layer.cellState.resize(stateSize, 0.0f);
        
        // Initialize weights (8 weight matrices: input, hidden for each gate)
        const size_t weightsSize = stateSize * 8 * stateSize;
        layer.weights.resize(weightsSize);
        
        // Xavier initialization for weights
        const float scale = std::sqrt(2.0f / (stateSize + stateSize));
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& w : layer.weights)
        {
            w = dist(gen);
        }
        
        // Initialize biases
        layer.biases.resize(4, 0.0f);  // One bias for each gate
    }
    
    // Initialize convolutional layers
    convLayers.resize(config.conv.numLayers);
    for (size_t i = 0; i < config.conv.numLayers; ++i)
    {
        auto& layer = convLayers[i];
        layer.dilation = config.conv.dilationRates[i % config.conv.dilationRates.size()];
        
        // Initialize weights
        const size_t filterSize = config.conv.filterSize;
        layer.weights.resize(filterSize);
        
        // Xavier initialization for weights
        const float scale = std::sqrt(2.0f / (filterSize + filterSize));
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& w : layer.weights)
        {
            w = dist(gen);
        }
        
        // Initialize biases
        layer.biases.resize(config.inputFeatureSize, 0.0f);
    }
    
    // Initialize multiband processor
    multibandProcessor.filters.resize(config.multiband.numBands);
    multibandProcessor.weights.resize(config.multiband.numBands, 1.0f / config.multiband.numBands);
    
    // Initialize normalization parameters
    normParams.mean.resize(config.inputFeatureSize, 0.0f);
    normParams.std.resize(config.inputFeatureSize, 1.0f);
}

void ToneMatcher::reset()
{
    // Reset audio processing
    audioProcessing.pickPositionDelay.reset();
    audioProcessing.lowShelf.reset();
    audioProcessing.highShelf.reset();
    
    // Reset LSTM states
    if (config.lstm.numLayers > 0)
    {
        for (auto& layer : lstmLayers)
        {
            std::fill(layer.hiddenState.begin(), layer.hiddenState.end(), 0.0f);
            std::fill(layer.cellState.begin(), layer.cellState.end(), 0.0f);
        }
    }

    // Reset multiband filters
    if (config.multiband.numBands > 0)
    {
        for (auto& filter : multibandProcessor.filters)
        {
            filter.reset();
        }
    }
}

std::vector<float> ToneMatcher::featuresToVector(const ToneFeatures& features) const
{
    std::vector<float> vector;
    vector.reserve(config.inputFeatureSize);  // Pre-allocate to avoid reallocations
    
    // Add spectral features with safety checks
    vector.push_back(std::isfinite(features.spectral.centroid) ? features.spectral.centroid : 0.0f);
    vector.push_back(std::isfinite(features.spectral.spread) ? features.spectral.spread : 0.0f);
    vector.push_back(std::isfinite(features.spectral.flatness) ? features.spectral.flatness : 0.0f);
    vector.push_back(std::isfinite(features.spectral.harmonicContent) ? features.spectral.harmonicContent : 0.0f);
    vector.push_back(std::isfinite(features.spectral.resonance) ? features.spectral.resonance : 0.0f);
    
    // Add temporal features with safety checks
    vector.push_back(std::isfinite(features.temporal.rms) ? features.temporal.rms : 0.0f);
    vector.push_back(std::isfinite(features.temporal.zeroCrossingRate) ? features.temporal.zeroCrossingRate : 0.0f);
    vector.push_back(std::isfinite(features.temporal.attackTime) ? features.temporal.attackTime : 0.0f);
    vector.push_back(std::isfinite(features.temporal.decayTime) ? features.temporal.decayTime : 0.0f);
    vector.push_back(std::isfinite(features.temporal.sustainLevel) ? features.temporal.sustainLevel : 0.0f);
    
    // Add modulation features with safety checks
    vector.push_back(std::isfinite(features.modulation.index) ? features.modulation.index : 0.0f);
    vector.push_back(std::isfinite(features.modulation.frequency) ? features.modulation.frequency : 0.0f);
    vector.push_back(std::isfinite(features.modulation.depth) ? features.modulation.depth : 0.0f);
    
    // Add harmonic features with safety checks
    const int maxHarmonics = std::min(10, static_cast<int>(config.inputFeatureSize - vector.size()) / 2);
    
    if (features.spectral.harmonicAmplitudes.size() != features.spectral.harmonicPhases.size())
    {
        std::cerr << "Warning: Harmonic amplitudes and phases size mismatch" << std::endl;
    }
    
    for (int i = 0; i < maxHarmonics && vector.size() < config.inputFeatureSize - 1; ++i)
    {
        // Add amplitude
        if (i < features.spectral.harmonicAmplitudes.size() && 
            std::isfinite(features.spectral.harmonicAmplitudes[i]))
        {
            vector.push_back(features.spectral.harmonicAmplitudes[i]);
        }
        else
        {
            vector.push_back(0.0f);
        }
        
        // Add phase
        if (i < features.spectral.harmonicPhases.size() && 
            std::isfinite(features.spectral.harmonicPhases[i]))
        {
            vector.push_back(features.spectral.harmonicPhases[i]);
        }
        else
        {
            vector.push_back(0.0f);
        }
    }
    
    // Ensure vector size matches config without truncating
    if (vector.size() > config.inputFeatureSize)
    {
        std::cerr << "Warning: Feature vector size (" << vector.size() 
                  << ") exceeds config size (" << config.inputFeatureSize << ")" << std::endl;
        vector.resize(config.inputFeatureSize);
    }
    else
    {
        while (vector.size() < config.inputFeatureSize)
        {
            vector.push_back(0.0f);
        }
    }
    
    return vector;
}

void ToneMatcher::vectorToFeatures(const std::vector<float>& vector, ToneFeatures& features) const
{
    if (vector.size() < config.inputFeatureSize)
    {
        std::cerr << "Error: Input vector size (" << vector.size() 
                  << ") is smaller than required size (" << config.inputFeatureSize << ")" << std::endl;
        return;
    }
    
    size_t offset = 0;
    
    // Extract spectral features
    features.spectral.centroid = vector[offset++];
    features.spectral.spread = vector[offset++];
    features.spectral.flatness = vector[offset++];
    features.spectral.harmonicContent = vector[offset++];
    features.spectral.resonance = vector[offset++];
    
    // Extract temporal features
    features.temporal.rms = vector[offset++];
    features.temporal.zeroCrossingRate = vector[offset++];
    features.temporal.attackTime = vector[offset++];
    features.temporal.decayTime = vector[offset++];
    features.temporal.sustainLevel = vector[offset++];
    
    // Extract modulation features
    features.modulation.index = vector[offset++];
    features.modulation.frequency = vector[offset++];
    features.modulation.depth = vector[offset++];
    
    // Extract harmonic features
    const int maxHarmonics = 10;
    features.spectral.harmonicAmplitudes.resize(maxHarmonics);
    features.spectral.harmonicPhases.resize(maxHarmonics);
    
    for (int i = 0; i < maxHarmonics; ++i)
    {
        if (offset + 1 < vector.size())
        {
            features.spectral.harmonicAmplitudes[i] = vector[offset++];
            features.spectral.harmonicPhases[i] = vector[offset++];
        }
        else
        {
            features.spectral.harmonicAmplitudes[i] = 0.0f;
            features.spectral.harmonicPhases[i] = 0.0f;
        }
    }
}

void ToneMatcher::normalizeFeatures(std::vector<float>& features) const
{
    if (features.empty() || normParams.mean.empty() || normParams.std.empty())
        return;
        
    const size_t size = std::min({features.size(), normParams.mean.size(), normParams.std.size()});
    for (size_t i = 0; i < size; ++i)
    {
        if (std::isfinite(features[i]) && std::isfinite(normParams.mean[i]) && std::isfinite(normParams.std[i]))
        {
            const float std = normParams.std[i] > 1e-6f ? normParams.std[i] : 1.0f;
            features[i] = (features[i] - normParams.mean[i]) / std;
        }
        else
        {
            features[i] = 0.0f;
        }
    }
}

void ToneMatcher::applyFeaturesToAudio(const std::vector<float>& features, float* audioData, int numSamples)
{
    try {
        if (features.empty() || audioData == nullptr || numSamples <= 0)
            return;
            
        // Create temporary buffer for processing
        std::vector<float> tempBuffer(numSamples);
        std::copy(audioData, audioData + numSamples, tempBuffer.begin());
        
        // Process features
        processLSTMLayer(features, tempBuffer);
        
        // Apply processed features back to audio
        for (int i = 0; i < numSamples && i < tempBuffer.size(); ++i)
        {
            audioData[i] = tempBuffer[i];
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in applyFeaturesToAudio: " << e.what() << std::endl;
    }
}

void ToneMatcher::prepare(double sampleRate)
{
    this->sampleRate = sampleRate;
    
    // Setup multiband processor
    const int numBands = config.multiband.numBands;
    multibandProcessor.filters.resize(numBands);
    multibandProcessor.weights.resize(numBands, 1.0f / numBands);
    
    for (int band = 0; band < numBands; ++band)
    {
        const float minFreq = config.multiband.minFreq;
        const float maxFreq = config.multiband.maxFreq;
        const float bandWidth = 0.5f;
        
        // Calculate center frequency using logarithmic spacing
        const float t = static_cast<float>(band) / static_cast<float>(numBands - 1);
        const float centerFreq = minFreq * std::pow(maxFreq / minFreq, t);
        
        // Setup bandpass filter
        auto& filter = multibandProcessor.filters[band];
        filter.setCoefficients(juce::IIRCoefficients::makeBandPass(sampleRate, centerFreq, bandWidth));
    }
    
    // Initialize audio processing
    audioProcessing.pickPositionDelay.prepare({sampleRate, static_cast<uint32>(blockSize), 1});
    audioProcessing.pickPositionDelay.setDelay(0.0f);
    
    audioProcessing.lowShelf.setCoefficients(juce::IIRCoefficients::makeLowShelf(sampleRate, 500.0f, 1.0f, 1.0f));
    audioProcessing.highShelf.setCoefficients(juce::IIRCoefficients::makeHighShelf(sampleRate, 2000.0f, 1.0f, 1.0f));
    
    // Initialize feature extractor
    if (featureExtractor != nullptr)
    {
        featureExtractor->prepare(sampleRate, blockSize);
    }
}

void ToneMatcher::process(const juce::AudioBuffer<float>& buffer)
{
    if (featureExtractor)
    {
        featureExtractor->processBlock(buffer);
    }
}

void ToneMatcher::train(const std::vector<float>& inputVector, const std::vector<float>& targetVector)
{
    if (!training || inputVector.empty() || targetVector.empty())
        return;

    if (inputVector.size() != config.inputFeatureSize || targetVector.size() != config.outputSize)
    {
        std::cerr << "Error: Input/target vector size mismatch. Expected " 
                  << config.inputFeatureSize << "/" << config.outputSize 
                  << ", got " << inputVector.size() << "/" << targetVector.size() << std::endl;
        return;
    }

    try
    {
        // Forward pass through convolutional layers
        std::vector<float> convOutput = inputVector;
        for (auto& layer : convLayers)
        {
            std::vector<float> layerOutput(convOutput.size());
            layer.processConv(convOutput.data(), layerOutput.data(), convOutput.size());
            convOutput = layerOutput;
        }

        // Forward pass through LSTM layers
        std::vector<float> lstmOutput = convOutput;
        for (size_t i = 0; i < lstmLayers.size(); ++i)
        {
            std::vector<float> layerOutput(lstmOutput.size());
            lstmLayers[i].processLSTM(lstmOutput.data(), layerOutput.data(), lstmOutput.size());
            lstmOutput = layerOutput;
        }

        // Process through multiband layers
        std::vector<float> multibandOutput(config.outputSize);
        for (size_t i = 0; i < config.multiband.numBands; ++i)
        {
            if (i < multibandProcessor.filters.size())
            {
                float bandOutput = lstmOutput[0];
                multibandProcessor.filters[i].processSamples(&bandOutput, 1);
                multibandOutput[i] = bandOutput * multibandProcessor.weights[i];
            }
        }

        // Calculate loss
        float loss = 0.0f;
        for (size_t i = 0; i < config.outputSize; ++i)
        {
            float diff = targetVector[i] - multibandOutput[i];
            loss += diff * diff;
        }
        loss /= static_cast<float>(config.outputSize);
        currentLoss = loss;

        // Backward pass and weight updates
        // Note: This is a placeholder for actual backpropagation
        // We'll implement proper backpropagation in a future update
        
        trainingSteps++;
        
        if (trainingSteps % 100 == 0)
        {
            std::cout << "Training step " << trainingSteps << ", Loss: " << loss << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during training: " << e.what() << std::endl;
        stopTraining();
    }
}

void ToneMatcher::LSTMLayer::processLSTM(const float* input, float* output, size_t size)
{
    try {
        // Ensure we have enough space in our states
        if (hiddenState.size() < size || cellState.size() < size)
        {
            hiddenState.resize(size, 0.0f);
            cellState.resize(size, 0.0f);
        }
        
        // Create temporary vectors for gate calculations
        std::vector<float> inputGate(size);
        std::vector<float> forgetGate(size);
        std::vector<float> outputGate(size);
        std::vector<float> cellInput(size);
        
        // Process each timestep
        for (size_t t = 0; t < size; ++t)
        {
            // Input gate
            inputGate[t] = 0.0f;
            for (size_t i = 0; i < size && i < weights.size() / 8; ++i)
            {
                if (t + i < size) {
                    inputGate[t] += input[i] * weights[i] + hiddenState[i] * weights[i + size];
                }
            }
            inputGate[t] = 1.0f / (1.0f + std::exp(-inputGate[t] - (t < biases.size() ? biases[0] : 0.0f)));  // sigmoid
            
            // Forget gate
            forgetGate[t] = 0.0f;
            for (size_t i = 0; i < size && i < weights.size() / 8; ++i)
            {
                if (t + i < size) {
                    forgetGate[t] += input[i] * weights[i + 2 * size] + hiddenState[i] * weights[i + 3 * size];
                }
            }
            forgetGate[t] = 1.0f / (1.0f + std::exp(-forgetGate[t] - (t < biases.size() ? biases[1] : 0.0f)));  // sigmoid
            
            // Cell input
            cellInput[t] = 0.0f;
            for (size_t i = 0; i < size && i < weights.size() / 8; ++i)
            {
                if (t + i < size) {
                    cellInput[t] += input[i] * weights[i + 4 * size] + hiddenState[i] * weights[i + 5 * size];
                }
            }
            cellInput[t] = std::tanh(cellInput[t] + (t < biases.size() ? biases[2] : 0.0f));
            
            // Update cell state
            if (t < cellState.size()) {
                cellState[t] = forgetGate[t] * cellState[t] + inputGate[t] * cellInput[t];
            }
            
            // Output gate
            outputGate[t] = 0.0f;
            for (size_t i = 0; i < size && i < weights.size() / 8; ++i)
            {
                if (t + i < size) {
                    outputGate[t] += input[i] * weights[i + 6 * size] + hiddenState[i] * weights[i + 7 * size];
                }
            }
            outputGate[t] = 1.0f / (1.0f + std::exp(-outputGate[t] - (t < biases.size() ? biases[3] : 0.0f)));  // sigmoid
            
            // Update hidden state and output
            if (t < hiddenState.size()) {
                hiddenState[t] = outputGate[t] * std::tanh(cellState[t]);
                output[t] = hiddenState[t];
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error in processLSTM: " << e.what() << std::endl;
        // Fill output with zeros in case of error
        std::fill(output, output + size, 0.0f);
    }
}

void ToneMatcher::ConvLayer::processConv(const float* input, float* output, size_t size)
{
    // Zero output buffer
    std::fill(output, output + size, 0.0f);
    
    // Apply dilated convolution
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t k = 0; k < weights.size(); ++k)
        {
            const size_t dilatedIndex = i + k * dilation;
            if (dilatedIndex < size)
            {
                output[i] += input[dilatedIndex] * weights[k];
            }
        }
        output[i] = std::tanh(output[i] + biases[i % biases.size()]);  // Add bias and apply activation
    }
}

void ToneMatcher::resetStates()
{
    // Reset all filter states
    for (auto& filter : multibandProcessor.filters)
    {
        filter.reset();
    }

    // Reset LSTM states
    lstmStates.clear();
    lstmStates.resize(lstmLayers.size());
    for (auto& state : lstmStates)
    {
        state.resize(lstmHiddenSize, 0.0f);
    }

    // Reset attention states
    attentionStates.clear();
    attentionStates.resize(attentionLayers.size());
    for (auto& state : attentionStates)
    {
        state.resize(attentionSize, 0.0f);
    }
}

ToneMatcher::~ToneMatcher()
{
    // Clean up any resources
    stopTraining();
}

std::vector<float> ToneMatcher::getCurrentFeatures(const float* audioData, int numSamples)
{
    if (!audioData || numSamples <= 0) {
        return std::vector<float>(config.inputFeatureSize, 0.0f);
    }

    // Create a temporary buffer to analyze
    juce::AudioBuffer<float> tempBuffer(1, numSamples);
    tempBuffer.copyFrom(0, 0, audioData, numSamples);
    
    // Extract features using the feature extractor
    auto features = extractFeatures(tempBuffer);
    
    // Convert to vector format
    std::vector<float> featureVector;
    featureVector.reserve(config.inputFeatureSize);
    
    // Add spectral features
    featureVector.insert(featureVector.end(), features.spectral.begin(), features.spectral.end());
    
    // Add temporal features
    featureVector.insert(featureVector.end(), features.temporal.begin(), features.temporal.end());
    
    // Add harmonic features
    featureVector.insert(featureVector.end(), features.harmonic.begin(), features.harmonic.end());
    
    // Normalize features
    normalizeFeatures(featureVector);
    
    return featureVector;
}

ToneMatcher::FeatureSet ToneMatcher::extractFeatures(const juce::AudioBuffer<float>& buffer)
{
    FeatureSet features;
    
    if (featureExtractor == nullptr || buffer.getNumSamples() == 0) {
        return features;
    }
    
    // Extract spectral features
    features.spectral = {
        featureExtractor->getSpectralCentroid(),
        featureExtractor->getSpectralSpread(),
        featureExtractor->getSpectralFlatness(),
        featureExtractor->getHarmonicContent(),
        featureExtractor->getResonance()
    };
    
    // Extract temporal features
    features.temporal = {
        featureExtractor->getRMSLevel(),
        featureExtractor->getZeroCrossingRate(),
        featureExtractor->getAttackTime(),
        featureExtractor->getDecayTime(),
        featureExtractor->getSustainLevel()
    };
    
    // Extract harmonic features
    const auto& harmonicAmps = featureExtractor->getHarmonicAmplitudes();
    const auto& harmonicPhases = featureExtractor->getHarmonicPhases();
    features.harmonic = harmonicAmps;
    features.harmonic.insert(features.harmonic.end(), harmonicPhases.begin(), harmonicPhases.end());
    
    return features;
}

void ToneMatcher::SpectralProcessor::prepare(int newFftSize)
{
    fftSize = newFftSize;
    fft = std::make_unique<juce::dsp::FFT>(std::log2(fftSize));
    window = std::make_unique<juce::dsp::WindowingFunction<float>>(fftSize, juce::dsp::WindowingFunction<float>::hann);
    
    fftBuffer.resize(fftSize * 2);  // For complex numbers
    spectrum.resize(fftSize);
    magnitudeSpectrum.resize(fftSize / 2 + 1);
    phaseSpectrum.resize(fftSize / 2 + 1);
}

void ToneMatcher::SpectralProcessor::processBlock(const float* input, float* output, int numSamples)
{
    // Copy input to FFT buffer
    std::fill(fftBuffer.begin(), fftBuffer.end(), 0.0f);
    for (int i = 0; i < std::min(numSamples, fftSize); ++i) {
        fftBuffer[i * 2] = input[i];
    }
    
    // Apply window
    window->multiplyWithWindowingTable(fftBuffer.data(), fftSize);
    
    // Perform FFT
    fft->perform(reinterpret_cast<juce::dsp::Complex<float>*>(fftBuffer.data()), 
                reinterpret_cast<juce::dsp::Complex<float>*>(spectrum.data()), false);
    
    // Extract magnitude and phase
    for (int i = 0; i <= fftSize / 2; ++i) {
        auto& bin = spectrum[i];
        magnitudeSpectrum[i] = std::abs(bin);
        phaseSpectrum[i] = std::arg(bin);
    }
}

void ToneMatcher::SpectralProcessor::modifySpectrum(const std::vector<float>& features)
{
    // Apply feature-based modifications to the spectrum
    const int numBins = fftSize / 2 + 1;
    
    // Modify magnitude spectrum based on features
    for (int i = 0; i < numBins; ++i) {
        float freq = i * 44100.0f / fftSize;  // Approximate frequency for this bin
        
        // Apply spectral shaping based on features
        float gainFactor = 1.0f;
        
        // Low frequencies (0-500 Hz)
        if (freq < 500.0f && features.size() > 0) {
            gainFactor *= std::exp(features[0]);
        }
        // Mid frequencies (500-2000 Hz)
        else if (freq < 2000.0f && features.size() > 1) {
            gainFactor *= std::exp(features[1]);
        }
        // High frequencies (2000+ Hz)
        else if (features.size() > 2) {
            gainFactor *= std::exp(features[2]);
        }
        
        // Apply gain while preserving phase
        spectrum[i] *= gainFactor;
    }
}

void ToneMatcher::processSpectral(juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    if (numSamples == 0) return;

    // Initialize spectral processor if not done
    if (spectralProcessor.fftSize == 0) {
        spectralProcessor.prepare(2048); // Use 2048 as default FFT size
    }

    // Create temporary buffers for overlap-add processing
    std::vector<float> outputAccumulator(numSamples, 0.0f);
    std::vector<float> windowAccumulator(numSamples, 0.0f);
    
    const float* inputData = buffer.getReadPointer(0);
    float* outputData = buffer.getWritePointer(0);
    
    // Process in blocks of FFT size with 75% overlap
    const int hopSize = spectralProcessor.fftSize / 4;
    const int numBlocks = (numSamples + hopSize - 1) / hopSize;
    
    for (int blockIndex = 0; blockIndex < numBlocks; ++blockIndex) {
        const int offset = blockIndex * hopSize;
        const int remainingSamples = numSamples - offset;
        const int blockSize = std::min(spectralProcessor.fftSize, remainingSamples);
        
        if (blockSize <= 0) break;
        
        // Create temporary input buffer with zero padding
        std::vector<float> blockBuffer(spectralProcessor.fftSize, 0.0f);
        for (int i = 0; i < blockSize; ++i) {
            blockBuffer[i] = inputData[offset + i];
        }
        
        // Process FFT block
        spectralProcessor.processBlock(blockBuffer.data(), nullptr, spectralProcessor.fftSize);
        
        // Get current features (using a smaller window for feature extraction)
        auto features = getCurrentFeatures(blockBuffer.data(), std::min(1024, blockSize));
        
        // Modify spectrum
        spectralProcessor.modifySpectrum(features);
        
        // Temporary buffer for reconstructed audio
        std::vector<float> reconstructedBlock(spectralProcessor.fftSize);
        reconstructAudio(spectralProcessor.spectrum, reconstructedBlock.data(), spectralProcessor.fftSize);
        
        // Overlap-add to output
        for (int i = 0; i < spectralProcessor.fftSize && (offset + i) < numSamples; ++i) {
            const float windowWeight = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / spectralProcessor.fftSize));
            outputAccumulator[offset + i] += reconstructedBlock[i] * windowWeight;
            windowAccumulator[offset + i] += windowWeight;
        }
    }
    
    // Normalize by window overlap
    for (int i = 0; i < numSamples; ++i) {
        outputData[i] = windowAccumulator[i] > 0.0f ? outputAccumulator[i] / windowAccumulator[i] : 0.0f;
    }
}

void ToneMatcher::reconstructAudio(const std::vector<std::complex<float>>& spectrum, float* output, int numSamples)
{
    if (!output || numSamples <= 0 || spectrum.empty()) return;
    
    std::vector<float> ifftBuffer(spectralProcessor.fftSize * 2, 0.0f);
    
    // Perform inverse FFT
    spectralProcessor.fft->perform(reinterpret_cast<const juce::dsp::Complex<float>*>(spectrum.data()),
                                 reinterpret_cast<juce::dsp::Complex<float>*>(ifftBuffer.data()), true);
    
    // Apply window and scale
    spectralProcessor.window->multiplyWithWindowingTable(ifftBuffer.data(), spectralProcessor.fftSize);
    const float scale = 1.0f / spectralProcessor.fftSize;
    
    // Copy to output
    for (int i = 0; i < numSamples; ++i) {
        output[i] = ifftBuffer[i * 2] * scale;  // Take real part only
    }
}

} // namespace GuitarToneEmulator 