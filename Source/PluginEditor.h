#pragma once

#include "JuceHeader.h"
#include "PluginProcessor.h"
#include "Visualization/ToneVisualizer.h"

namespace GuitarToneEmulator {

// Custom look and feel for modern UI
class CustomLookAndFeel : public juce::LookAndFeel_V4
{
public:
    CustomLookAndFeel();
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                         float sliderPosProportional, float rotaryStartAngle,
                         float rotaryEndAngle, juce::Slider& slider) override;
private:
    juce::Image knobImage;
};

class GuitarToneEmulatorEditor : public juce::AudioProcessorEditor,
                                public juce::Timer
{
public:
    GuitarToneEmulatorEditor(GuitarToneEmulatorProcessor&);
    ~GuitarToneEmulatorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void setMode(Mode mode);
    void timerCallback() override;

private:
    GuitarToneEmulatorProcessor& processor;
    
    // Look and feel
    CustomLookAndFeel lookAndFeel;
    
    // Mode buttons
    juce::TextButton ampModeButton;
    juce::TextButton toneMatchButton;
    
    // Control sliders
    juce::Slider gainSlider;
    juce::Slider toneSlider;
    juce::Slider presenceSlider;
    
    // Labels
    juce::Label gainLabel;
    juce::Label toneLabel;
    juce::Label presenceLabel;
    
    // Visualizer
    ToneVisualizer visualizer;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GuitarToneEmulatorEditor)
};

} // namespace GuitarToneEmulator 