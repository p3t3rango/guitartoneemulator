#include "ToneVisualizer.h"

namespace GuitarToneEmulator {

// Color scheme implementation
const juce::Colour ToneVisualizer::Colors::background(0xFF1A1A1A);
const juce::Colour ToneVisualizer::Colors::grid(0xFF2A2A2A);
const juce::Colour ToneVisualizer::Colors::waveform(0xFF00FFFF);
const juce::Colour ToneVisualizer::Colors::spectrum(0xFF00FF00);
const juce::Colour ToneVisualizer::Colors::features(0xFF0000FF);
const juce::Colour ToneVisualizer::Colors::comparison(0xFFFF00FF);
const juce::Colour ToneVisualizer::Colors::reference(0xFFFF0000);
const juce::Colour ToneVisualizer::Colors::matched(0xFFFFFF00);
const juce::Colour ToneVisualizer::Colors::text(0xFFFFFFFF);

void ToneVisualizer::drawWaveform(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(Colors::waveform);
    
    const float scaleFactor = bounds.getHeight() / 2.0f;
    const float width = bounds.getWidth();
    
    juce::Path waveformPath;
    waveformPath.startNewSubPath(0, bounds.getCentreY());
    
    const float* data = audioBuffer.getReadPointer(0);
    for (int i = 0; i < audioBuffer.getNumSamples(); ++i)
    {
        const float x = i * width / audioBuffer.getNumSamples();
        const float y = bounds.getCentreY() - data[i] * scaleFactor;
        waveformPath.lineTo(x, y);
    }
    
    g.strokePath(waveformPath, juce::PathStrokeType(2.0f));
}

void ToneVisualizer::drawSpectrum(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(Colors::spectrum);
    
    const float scaleFactor = bounds.getHeight();
    const float width = bounds.getWidth();
    
    juce::Path spectrumPath;
    spectrumPath.startNewSubPath(0, bounds.getBottom());
    
    for (size_t i = 0; i < spectrum.size() / 2; ++i)
    {
        const float x = std::log10(1 + i) * width / std::log10(1 + spectrum.size() / 2);
        const float y = bounds.getBottom() - spectrum[i] * scaleFactor;
        spectrumPath.lineTo(x, y);
    }
    
    g.strokePath(spectrumPath, juce::PathStrokeType(2.0f));
}

void ToneVisualizer::drawFeatures(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(Colors::features);
    // TODO: Implement feature visualization
}

void ToneVisualizer::drawComparison(juce::Graphics& g, const juce::Rectangle<int>& bounds)
{
    g.setColour(Colors::comparison);
    // TODO: Implement comparison visualization
}

} // namespace GuitarToneEmulator 