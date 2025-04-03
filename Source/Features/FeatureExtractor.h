#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <array>
#include <complex>

namespace GuitarToneEmulator {

// Feature structures
struct SpectralFeatures {
    float centroid;
    float spread;
    float flatness;
    float harmonicContent;
    float resonance;
    std::vector<float> harmonicAmplitudes;
    std::vector<float> harmonicPhases;
    float inharmonicity;  // Measure of deviation from perfect harmonic series
    float spectralTilt;   // High-to-low frequency energy ratio
    float formantPeaks[3];  // First three formant frequencies
    float formantAmplitudes[3];  // Corresponding formant amplitudes
    float pickPosition;   // Estimated pick position from comb filtering
    float stringCoupling; // Measure of string-to-string interaction
    std::vector<float> frequencyBandEnergies;
    float harmonicBalance;
};

struct TemporalFeatures {
    struct Transient {
        int position;
        float strength;
        float frequency;
        float amplitude;
        float duration;
    };

    float rms;
    float zeroCrossingRate;
    float attackTime;
    float decayTime;
    float sustainLevel;
    float releaseTime;
    float dynamicRange;
    float crest;  // Peak-to-RMS ratio
    float transientSharpness;  // Rate of attack
    float sustainUnevenness;   // Variation in sustain
    float releaseShape;        // Exponential vs linear release
    float amplitude;
    float nonLinearity;        // Measure of waveshaping
    std::vector<Transient> transients;
    float pluckCharacteristics;  // Initial attack characteristics
    float palmMuteIntensity;    // Degree of palm muting
};

struct ModulationFeatures {
    float index;
    float frequency;
    float depth;
    float vibratoRate;
    float vibratoDepth;
    float tremoloRate;
    float tremoloDepth;
    float phase;
    float harmonicTremolo;     // Frequency-dependent tremolo
    float modulationSymmetry;  // Asymmetry in modulation
    float modulationCoherence; // Phase relationship between modulations
    float pickupPhase;         // Phase relationship between pickups
    float resonantModulation;  // Interaction with body resonances
};

struct ToneFeatures {
    // Tone characteristics
    float brightness;
    float warmth;
    float presence;
    float clarity;
    float fullness;

    // Detailed features
    SpectralFeatures spectral;
    TemporalFeatures temporal;
    ModulationFeatures modulation;

    float stringResonance;    // Individual string resonance
    float bodyResonance;      // Guitar body resonance
    float pickupResponse;     // Pickup frequency response
    float compressionAmount;  // Dynamic range compression
    float spatialWidth;       // Stereo image width
    float phaseCoherence;     // Phase relationships
};

class FeatureExtractor {
public:
    struct Features {
        std::vector<float> spectral;
        std::vector<float> temporal;
        std::vector<float> modulation;
    };

    FeatureExtractor();

    void prepare(double sampleRate, int blockSize);
    void processBlock(const juce::AudioBuffer<float>& buffer);
    Features extractFeatures(const juce::AudioBuffer<float>& buffer);

    // Getters for individual features
    float getRMSLevel() const { return rmsLevel; }
    float getZeroCrossingRate() const { return zeroCrossingRate; }
    float getSpectralCentroid() const { return spectralCentroid; }
    float getSpectralSpread() const { return spectralSpread; }
    float getSpectralFlatness() const { return spectralFlatness; }
    float getModulationIndex() const { return modulationIndex; }
    float getModulationFrequency() const { return modulationFrequency; }
    float getModulationDepth() const { return modulationDepth; }
    float getHarmonicContent() const { return harmonicContent; }
    float getResonance() const { return resonance; }
    float getAttackTime() const { return attackTime; }
    float getDecayTime() const { return decayTime; }
    float getSustainLevel() const { return sustainLevel; }
    float getReleaseTime() const { return releaseTime; }
    const std::vector<float>& getHarmonicAmplitudes() const { return harmonicAmplitudes; }
    const std::vector<float>& getHarmonicPhases() const { return harmonicPhases; }

    // Main analysis function
    ToneFeatures analyzeTone(const juce::AudioBuffer<float>& buffer);
    
    // Individual analysis functions
    SpectralFeatures analyzeSpectralContent(const juce::AudioBuffer<float>& buffer);
    TemporalFeatures analyzeTemporalContent(const juce::AudioBuffer<float>& buffer);
    ModulationFeatures analyzeModulationContent(const juce::AudioBuffer<float>& buffer);
    
    // Tone type classification
    enum class ToneType {
        Clean,
        Crunch,
        Lead,
        Metal,
        Bass,
        Acoustic
    };
    ToneType classifyTone(const ToneFeatures& features);

    float calculateResonance() const;

private:
    void calculateRMS(const juce::AudioBuffer<float>& buffer);
    void calculateZeroCrossingRate(const juce::AudioBuffer<float>& buffer);
    void calculateSpectralFeatures(const juce::AudioBuffer<float>& buffer);
    void calculateModulationFeatures(const juce::AudioBuffer<float>& buffer);
    void calculateEnvelopeFeatures(const juce::AudioBuffer<float>& buffer);
    void calculateHarmonicContent(const juce::AudioBuffer<float>& buffer);
    void calculateResonance(const juce::AudioBuffer<float>& buffer);
    void analyzeToneCharacteristics(ToneFeatures& features);

    // FFT and windowing
    std::unique_ptr<juce::dsp::FFT> fft;
    std::unique_ptr<juce::dsp::WindowingFunction<float>> window;
    size_t fftSize;
    std::vector<float> fftBuffer;
    std::vector<float> magnitudeSpectrum;
    std::vector<float> spectrum;
    std::vector<float> previousMagnitudeSpectrum;
    std::vector<float> previousFFT;
    std::vector<float> envelopeBuffer;

    // Feature storage
    float rmsLevel;
    float zeroCrossingRate;
    float spectralCentroid;
    float spectralSpread;
    float spectralFlatness;
    float harmonicContent;
    float resonance;
    float modulationIndex;
    float modulationFrequency;
    float modulationDepth;
    float attackTime;
    float decayTime;
    float sustainLevel;
    float releaseTime;
    std::vector<float> harmonicAmplitudes;
    std::vector<float> harmonicPhases;
    std::vector<TemporalFeatures::Transient> transients;

    // Processing parameters
    double sampleRate;
    int samplesPerBlock;
    float envelopeFollower;

    // Helper functions
    void prepareFFT();
    void performFFT(const std::vector<float>& input);
    float findFundamentalFrequency(const std::vector<float>& spectrum);
    void detectHarmonics(const std::vector<float>& spectrum, float fundamental);
    float calculateHarmonicToNoiseRatio(const std::vector<float>& spectrum);
    void updateEnvelopeFollower(float input);
    
    // Frequency band definitions
    struct FrequencyBand {
        float lowFreq;
        float highFreq;
        float resonance;
        
        FrequencyBand(float low = 0.0f, float high = 0.0f, float res = 0.0f)
            : lowFreq(low), highFreq(high), resonance(res) {}
    };
    
    std::array<FrequencyBand, 4> resonanceBands;
    std::array<std::pair<float, float>, 5> frequencyBands;

    float fundamental;

    float findFundamentalFrequencyYIN(const juce::AudioBuffer<float>& buffer);
    void analyzeHarmonicsPhaseAligned(const juce::AudioBuffer<float>& buffer, float fundamental);
    float calculateHarmonicToNoiseRatioImproved(const juce::AudioBuffer<float>& buffer);

    class SpectralProcessor {
    public:
        SpectralProcessor() = default;
        void prepare(double sampleRate, int blockSize);
        std::vector<float> process(const float* data, int numSamples);
    private:
        std::unique_ptr<juce::dsp::FFT> fft;
        std::vector<float> fftBuffer;
        int currentFftSize = 2048;
        double sampleRate = 44100.0;
    };

    class TemporalProcessor {
    public:
        TemporalProcessor() = default;
        void prepare(double sampleRate, int blockSize);
        std::vector<float> process(const float* data, int numSamples);
    private:
        double sampleRate = 44100.0;
    };

    class ModulationProcessor {
    public:
        ModulationProcessor() = default;
        void prepare(double sampleRate, int blockSize);
        std::vector<float> process(const float* data, int numSamples);
    private:
        double sampleRate = 44100.0;
    };

    SpectralProcessor spectralProcessor;
    TemporalProcessor temporalProcessor;
    ModulationProcessor modulationProcessor;
    
    double currentSampleRate = 44100.0;
    int currentBlockSize = 2048;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(FeatureExtractor)
};

} // namespace GuitarToneEmulator 