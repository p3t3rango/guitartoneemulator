#pragma once

#include <JuceHeader.h>
#include <vector>
#include <random>
#include <queue>
#include "../Features/FeatureExtractor.h"

namespace GuitarToneEmulator {

class ToneMatcher
{
public:
    // Audio processing parameters
    struct AudioParameters {
        float inputGain = 1.0f;
        float outputGain = 1.0f;
        juce::IIRFilter lowShelf;
        juce::IIRFilter highShelf;
        std::vector<juce::IIRFilter> eqBands;
        juce::dsp::DelayLine<float> pickPositionDelay{44100};
        float dryWet = 1.0f;
    };

    // Neural network configuration
    struct NetworkConfig {
        // Basic network parameters
        int inputFeatureSize = 0;
        int outputSize = 0;
        int hiddenSize = 64;
        int numLayers = 2;
        float learningRate = 0.001f;
        int batchSize = 32;

        // LSTM configuration
        struct {
            int numLayers = 2;
            int hiddenSize = 64;
            bool useAttention = true;
        } lstm;

        // Convolutional configuration
        struct {
            int numLayers = 2;
            int filterSize = 3;
            std::vector<int> dilationRates = {1, 2, 4, 8};
        } conv;

        // Multiband configuration
        struct {
            int numBands = 8;
            float minFreq = 20.0f;
            float maxFreq = 20000.0f;
        } multiband;
    };

    // Multiband processor
    struct MultibandProcessor {
        std::vector<juce::IIRFilter> filters;
        std::vector<float> weights;
    };

    // Training batch
    struct TrainingBatch {
        std::vector<std::vector<float>> inputFeatures;
        std::vector<std::vector<float>> targetFeatures;
    };

    // Attention layer
    struct AttentionLayer {
        std::vector<std::vector<float>> queryWeights;
        std::vector<std::vector<float>> keyWeights;
        std::vector<std::vector<float>> valueWeights;
        std::vector<float> biases;
        float scalingFactor = 1.0f;
    };

    ToneMatcher();
    ~ToneMatcher();
    
    void prepareToPlay(double sampleRate, int samplesPerBlock);
    void processBlock(juce::AudioBuffer<float>& buffer);
    void reset();
    void prepare(double sampleRate);
    void process(const juce::AudioBuffer<float>& buffer);

    // Training interface
    void startTraining(const juce::AudioBuffer<float>& referenceBuffer);
    void stopTraining();
    void processTraining(const juce::AudioBuffer<float>& buffer);
    void processBatch();
    void resetTraining();
    bool isTraining() const { return training; }
    float getCurrentLoss() const { return currentLoss; }

    // Model persistence
    void saveModel(const juce::String& path);
    void loadModel(const juce::String& path);

    struct SpectralProcessor {
        void prepare(int newFftSize);
        void processBlock(const float* input, float* output, int numSamples);
        void modifySpectrum(const std::vector<float>& features);
        
        std::unique_ptr<juce::dsp::FFT> fft;
        std::unique_ptr<juce::dsp::WindowingFunction<float>> window;
        std::vector<float> fftBuffer;
        std::vector<std::complex<float>> spectrum;
        std::vector<float> magnitudeSpectrum;
        std::vector<float> phaseSpectrum;
        int fftSize = 2048;
    };

private:
    // Feature extraction
    struct FeatureSet {
        std::vector<float> spectral;    // Spectral features (centroid, spread, etc.)
        std::vector<float> temporal;     // Temporal features (RMS, ZCR, etc.)
        std::vector<float> harmonic;     // Harmonic features (pitch, harmonics)
        std::vector<std::complex<float>> spectrum; // Raw spectrum for processing
    };

    // Neural network components
    struct LSTMLayer {
        std::vector<float> hiddenState;
        std::vector<float> cellState;
        std::vector<float> weights;
        std::vector<float> biases;
        std::vector<float> inputWeights;
        std::vector<float> forgetWeights;
        std::vector<float> cellWeights;
        std::vector<float> outputWeights;
        std::vector<float> recurrentWeights;
        std::vector<float> inputBias;
        std::vector<float> forgetBias;
        std::vector<float> cellBias;
        std::vector<float> outputBias;
        
        void processLSTM(const float* input, float* output, size_t size);
        void reset();
    };

    struct ConvLayer {
        std::vector<std::vector<float>> filters;
        std::vector<float> biases;
        std::vector<float> weights;
        int kernelSize;
        int dilation;
        
        void processConv(const float* input, float* output, size_t size);
        void reset();
    };

    // Core processing methods
    FeatureSet extractFeatures(const juce::AudioBuffer<float>& buffer);
    void processFeatures(const FeatureSet& input, FeatureSet& output);
    void applyFeatures(const FeatureSet& features, juce::AudioBuffer<float>& buffer);
    void applyFeaturesToAudio(const std::vector<float>& features, float* audioData, int numSamples);
    std::vector<float> getCurrentFeatures(const float* audioData, int numSamples);
    std::vector<float> featuresToVector(const ToneFeatures& features) const;
    void vectorToFeatures(const std::vector<float>& vector, ToneFeatures& features) const;
    void normalizeFeatures(std::vector<float>& features) const;

    // Neural network processing
    void processConvolutionalLayer(const std::vector<float>& input, std::vector<float>& output);
    void processLSTMLayer(const std::vector<float>& input, std::vector<float>& output);
    void processMultibandLayer(const std::vector<float>& input, std::vector<float>& output);
    void processAttention(const std::vector<float>& input, std::vector<float>& output, AttentionLayer& attention);

    // Loss calculation
    float calculateSpectralLoss(const std::vector<float>& predicted, const std::vector<float>& target);
    float calculateTemporalLoss(const std::vector<float>& predicted, const std::vector<float>& target);

    // Neural network training
    void trainOnFeatures(const FeatureSet& input, const FeatureSet& target);
    void updateNetworkWeights(const std::vector<float>& gradients);
    void initializeNetwork();
    void resetStates();
    void train(const std::vector<float>& inputVector, const std::vector<float>& targetVector);

    // Helper functions
    float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
    float tanh(float x) const { return std::tanh(x); }

    // Audio processing chain
    void setupAudioProcessing();
    void processAudioBlock(juce::dsp::AudioBlock<float>& block);

    // Member variables
    std::unique_ptr<FeatureExtractor> featureExtractor;
    std::vector<LSTMLayer> lstmLayers;
    std::vector<ConvLayer> convLayers;
    std::vector<AttentionLayer> attentionLayers;
    AudioParameters audioProcessing;
    NetworkConfig config;
    MultibandProcessor multibandProcessor;

    double sampleRate = 44100.0;
    int blockSize = 512;
    bool training = false;
    float currentLoss = 0.0f;
    int trainingSteps = 0;
    std::mt19937 gen;  // Random number generator

    // Add normalization parameters
    struct NormalizationParams {
        std::vector<float> mean;
        std::vector<float> std;
    } normParams;

    std::vector<float> referenceVector;
    std::queue<TrainingBatch> batchQueue;
    FeatureSet referenceFeatures;
    juce::dsp::ProcessSpec processSpec;

    // LSTM state management
    std::vector<std::vector<float>> lstmStates;
    std::vector<std::vector<float>> attentionStates;
    int lstmHiddenSize = 64;
    int attentionSize = 64;

    // Spectral processing
    void processSpectral(juce::AudioBuffer<float>& buffer);
    void reconstructAudio(const std::vector<std::complex<float>>& spectrum, float* output, int numSamples);
    
    SpectralProcessor spectralProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneMatcher)
};

} // namespace GuitarToneEmulator 