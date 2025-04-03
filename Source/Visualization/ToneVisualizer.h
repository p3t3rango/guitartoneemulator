#pragma once

#include <JuceHeader.h>
#include "../Features/FeatureExtractor.h"
#include <vector>

namespace GuitarToneEmulator {

class ToneVisualizer : public juce::Component,
                      public juce::Timer {
public:
    enum class VisualizationType
    {
        Waveform,
        Spectrum,
        Features,
        Comparison
    };

    ToneVisualizer()
        : fft(std::make_unique<juce::dsp::FFT>(12)), // 4096 points
          window(std::make_unique<juce::dsp::WindowingFunction<float>>(4096, juce::dsp::WindowingFunction<float>::hann)),
          audioBuffer(2, 4096),
          currentType(VisualizationType::Waveform),
          matchAmount(0.0f),
          sampleRate(44100.0),
          blockSize(512)
    {
        startTimer(33); // ~30 Hz refresh rate
    }

    ~ToneVisualizer() override {
        stopTimer();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);
        auto bounds = getLocalBounds();

        switch (currentType)
        {
            case VisualizationType::Waveform:
                drawWaveform(g, bounds);
                break;
            case VisualizationType::Spectrum:
                drawSpectrum(g, bounds);
                break;
            case VisualizationType::Features:
                drawFeatures(g, bounds);
                break;
            case VisualizationType::Comparison:
                drawComparison(g, bounds);
                break;
        }
    }

    void resized() override {
        // Update visualization layout if needed
    }

    void timerCallback() override {
        repaint();
    }

    void setSpectrum(const std::vector<float>& newSpectrum)
    {
        spectrum = newSpectrum;
        repaint();
    }

    void updateAudioBuffer(const juce::AudioBuffer<float>& buffer)
    {
        audioBuffer.makeCopyOf(buffer);
        repaint();
    }

    void setVisualizationType(VisualizationType type) { currentType = type; repaint(); }
    VisualizationType getVisualizationType() const { return currentType; }

    void setMatchAmount(float amount) { matchAmount = amount; repaint(); }
    float getMatchAmount() const { return matchAmount; }

    void drawWaveform(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawSpectrum(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawFeatures(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawComparison(juce::Graphics& g, const juce::Rectangle<int>& bounds);

private:
    std::unique_ptr<juce::dsp::FFT> fft;
    std::unique_ptr<juce::dsp::WindowingFunction<float>> window;
    juce::AudioBuffer<float> audioBuffer;
    std::vector<float> spectrum;
    VisualizationType currentType;
    float matchAmount;
    double sampleRate;
    int blockSize;

    struct Colors
    {
        static const juce::Colour background;
        static const juce::Colour grid;
        static const juce::Colour waveform;
        static const juce::Colour spectrum;
        static const juce::Colour features;
        static const juce::Colour comparison;
        static const juce::Colour reference;
        static const juce::Colour matched;
        static const juce::Colour text;
    };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneVisualizer)
};

} // namespace GuitarToneEmulator 