#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "JuceHeader.h"
#include "Features/FeatureExtractor.h"
#include "Neural/ToneMatcher.h"
#include "Neural/RTNeuralWrapper.h"
#include "Visualization/ToneVisualizer.h"

namespace GuitarToneEmulator {

enum class Mode {
    AmpSimulation,
    ToneMatching
};

struct AmpSimulator {
    juce::dsp::ProcessorChain<
        juce::dsp::IIR::Filter<float>,
        juce::dsp::IIR::Filter<float>,
        juce::dsp::IIR::Filter<float>
    > chain;

    void prepare(const juce::dsp::ProcessSpec& spec) {
        chain.prepare(spec);
    }

    void reset() {
        chain.reset();
    }
};

class GuitarToneEmulatorProcessor : public juce::AudioProcessor
{
public:
    GuitarToneEmulatorProcessor();
    ~GuitarToneEmulatorProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int index) override {}
    const juce::String getProgramName(int index) override { return {}; }
    void changeProgramName(int index, const juce::String& newName) override {}
    
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Mode handling
    Mode getMode() const { return currentMode; }
    void setMode(Mode mode) { currentMode = mode; }

    // Custom methods
    void loadImpulseResponse(const juce::File& irFile);
    void loadReferenceTrack(const juce::File& file);
    
    // Audio processing methods
    void processAmpSimulation(juce::AudioBuffer<float>& buffer);
    void processToneMatching(juce::AudioBuffer<float>& buffer);
    void processPreamp(juce::AudioBuffer<float>& buffer);
    void processPowerAmp(juce::AudioBuffer<float>& buffer);
    void processCabinet(juce::AudioBuffer<float>& buffer);

    // Utility methods
    static int nextPowerOfTwo(int n);

    // Parameter handling
    juce::AudioProcessorValueTreeState parameters;
    
private:
    Mode currentMode;
    AmpSimulator ampSim;
    std::unique_ptr<ToneMatcher> toneMatcher;
    std::unique_ptr<FeatureExtractor> featureExtractor;
    
    // DSP components
    juce::dsp::ProcessorDuplicator<juce::dsp::IIR::Filter<float>, juce::dsp::IIR::Coefficients<float>> toneFilter;
    juce::dsp::ProcessorDuplicator<juce::dsp::IIR::Filter<float>, juce::dsp::IIR::Coefficients<float>> lowShelfFilter;
    juce::dsp::ProcessorDuplicator<juce::dsp::IIR::Filter<float>, juce::dsp::IIR::Coefficients<float>> midPeakFilter;
    
    // Parameter handling
    juce::AudioProcessorValueTreeState::ParameterLayout createParameters();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GuitarToneEmulatorProcessor)
};

} // namespace GuitarToneEmulator

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter(); 