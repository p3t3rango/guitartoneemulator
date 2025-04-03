#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace GuitarToneEmulator {

GuitarToneEmulatorProcessor::GuitarToneEmulatorProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true))
    , currentMode(Mode::AmpSimulation)
    , parameters(*this, nullptr, "Parameters", createParameters())
    , toneFilter(juce::dsp::IIR::Coefficients<float>::makeLowPass(44100, 1000.0f))
    , lowShelfFilter(juce::dsp::IIR::Coefficients<float>::makeLowShelf(44100, 1000.0f, 0.707f, 0.0f))
    , midPeakFilter(juce::dsp::IIR::Coefficients<float>::makePeakFilter(44100, 1000.0f, 0.707f, 0.0f))
{
    // Initialize components
    toneMatcher = std::make_unique<ToneMatcher>();
    featureExtractor = std::make_unique<FeatureExtractor>();
}

void GuitarToneEmulatorProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    // Prepare DSP modules
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = samplesPerBlock;
    spec.numChannels = getTotalNumOutputChannels();

    // Prepare amp simulation components
    ampSim.chain.prepare(spec);

    // Prepare tone matching components
    if (toneMatcher) toneMatcher->prepareToPlay(sampleRate, samplesPerBlock);
    if (featureExtractor) featureExtractor->prepare(sampleRate, samplesPerBlock);
}

void GuitarToneEmulatorProcessor::releaseResources()
{
    // Release any allocated resources when playback stops
}

bool GuitarToneEmulatorProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Only support stereo in/out
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

void GuitarToneEmulatorProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Clear any output channels that don't contain input data
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // Process based on current mode
    switch (currentMode)
    {
        case Mode::AmpSimulation:
            processAmpSimulation(buffer);
            break;
        case Mode::ToneMatching:
            processToneMatching(buffer);
            break;
    }
}

void GuitarToneEmulatorProcessor::processAmpSimulation(juce::AudioBuffer<float>& buffer)
{
    processPreamp(buffer);
    processPowerAmp(buffer);
    processCabinet(buffer);
}

void GuitarToneEmulatorProcessor::processToneMatching(juce::AudioBuffer<float>& buffer)
{
    if (toneMatcher)
        toneMatcher->processBlock(buffer);
}

void GuitarToneEmulatorProcessor::processPreamp(juce::AudioBuffer<float>& buffer)
{
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    
    // Apply tone shaping
    toneFilter.process(context);
}

void GuitarToneEmulatorProcessor::processPowerAmp(juce::AudioBuffer<float>& buffer)
{
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    
    // Apply EQ
    lowShelfFilter.process(context);
    midPeakFilter.process(context);
}

void GuitarToneEmulatorProcessor::processCabinet(juce::AudioBuffer<float>& buffer)
{
    // Process through amp simulator chain
    juce::dsp::AudioBlock<float> block(buffer);
    juce::dsp::ProcessContextReplacing<float> context(block);
    ampSim.chain.process(context);
}

juce::AudioProcessorEditor* GuitarToneEmulatorProcessor::createEditor()
{
    return new GuitarToneEmulatorEditor(*this);
}

void GuitarToneEmulatorProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void GuitarToneEmulatorProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState.get() != nullptr)
        parameters.replaceState(juce::ValueTree::fromXml(*xmlState));
}

juce::AudioProcessorValueTreeState::ParameterLayout GuitarToneEmulatorProcessor::createParameters()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Add parameters for amp simulation
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "gain", "Gain", 0.0f, 1.0f, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "tone", "Tone", 0.0f, 1.0f, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "presence", "Presence", 0.0f, 1.0f, 0.5f));

    // Add parameters for tone matching
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        "toneMatchingEnabled", "Tone Matching", false));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "toneMatchAmount", "Tone Match Amount", 0.0f, 1.0f, 0.5f));

    return { params.begin(), params.end() };
}

} // namespace GuitarToneEmulator

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new GuitarToneEmulator::GuitarToneEmulatorProcessor();
} 