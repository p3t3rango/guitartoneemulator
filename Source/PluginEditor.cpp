#include "PluginProcessor.h"
#include "PluginEditor.h"

namespace GuitarToneEmulator {

//==============================================================================
CustomLookAndFeel::CustomLookAndFeel()
{
    // Load custom knob image here if needed
    // knobImage = juce::ImageCache::getFromMemory(BinaryData::knob_png, BinaryData::knob_pngSize);
}

void CustomLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                       float sliderPosProportional, float rotaryStartAngle,
                                       float rotaryEndAngle, juce::Slider& slider)
{
    // Modern flat design for the knob
    auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat();
    auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
    auto toAngle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);
    auto lineWidth = radius * 0.1f;
    auto arcRadius = radius - lineWidth * 2.0f;

    // Draw knob body
    g.setColour(juce::Colour(40, 40, 40));
    g.fillEllipse(bounds.reduced(lineWidth));

    // Draw indicator arc
    juce::Path arcPath;
    arcPath.addCentredArc(bounds.getCentreX(), bounds.getCentreY(),
                         arcRadius, arcRadius,
                         0.0f, rotaryStartAngle, toAngle,
                         true);

    g.setColour(juce::Colour(97, 183, 237));
    g.strokePath(arcPath, juce::PathStrokeType(lineWidth));

    // Draw indicator dot
    juce::Path dotPath;
    auto dotRadius = radius * 0.1f;
    auto dotAngle = toAngle - juce::MathConstants<float>::halfPi;
    auto dotCentre = bounds.getCentre().getPointOnCircumference(radius * 0.8f, dotAngle);
    dotPath.addEllipse(dotCentre.getX() - dotRadius, dotCentre.getY() - dotRadius,
                      dotRadius * 2.0f, dotRadius * 2.0f);
    g.setColour(juce::Colours::white);
    g.fillPath(dotPath);
}

//==============================================================================
GuitarToneEmulatorEditor::GuitarToneEmulatorEditor(GuitarToneEmulatorProcessor& p)
    : AudioProcessorEditor(&p), processor(p)
{
    // Set up look and feel
    setLookAndFeel(&lookAndFeel);
    
    // Set up mode buttons
    ampModeButton.setButtonText("Amp Mode");
    ampModeButton.onClick = [this]() { setMode(Mode::AmpSimulation); };
    addAndMakeVisible(ampModeButton);
    
    toneMatchButton.setButtonText("Tone Match");
    toneMatchButton.onClick = [this]() { setMode(Mode::ToneMatching); };
    addAndMakeVisible(toneMatchButton);
    
    // Set up sliders
    gainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    gainSlider.setRange(-48.0f, 48.0f, 0.1f);
    gainSlider.setValue(processor.parameters.getRawParameterValue("gain")->load());
    gainSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 100, 20);
    gainSlider.onValueChange = [this] { *processor.parameters.getRawParameterValue("gain") = gainSlider.getValue(); };
    addAndMakeVisible(gainSlider);
    
    toneSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    toneSlider.setRange(0.0f, 1.0f, 0.01f);
    toneSlider.setValue(processor.parameters.getRawParameterValue("tone")->load());
    toneSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 100, 20);
    toneSlider.onValueChange = [this] { *processor.parameters.getRawParameterValue("tone") = toneSlider.getValue(); };
    addAndMakeVisible(toneSlider);
    
    presenceSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    presenceSlider.setRange(0.0f, 1.0f, 0.01f);
    presenceSlider.setValue(processor.parameters.getRawParameterValue("presence")->load());
    presenceSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 100, 20);
    presenceSlider.onValueChange = [this] { *processor.parameters.getRawParameterValue("presence") = presenceSlider.getValue(); };
    addAndMakeVisible(presenceSlider);

    // Set up labels
    gainLabel.setText("Gain", juce::dontSendNotification);
    gainLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(gainLabel);
    
    toneLabel.setText("Tone", juce::dontSendNotification);
    toneLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(toneLabel);
    
    presenceLabel.setText("Presence", juce::dontSendNotification);
    presenceLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(presenceLabel);
    
    // Set up visualizer
    addAndMakeVisible(visualizer);
    
    // Set window size
    setSize(800, 600);
    
    // Start timer for visualization updates
    startTimerHz(30); // 30 FPS
}

GuitarToneEmulatorEditor::~GuitarToneEmulatorEditor()
{
    setLookAndFeel(nullptr);
    stopTimer();
}

void GuitarToneEmulatorEditor::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    
    // Draw title
    g.setColour(juce::Colours::white);
    g.setFont(24.0f);
    g.drawFittedText("Guitar Tone Emulator", getLocalBounds().removeFromTop(40),
                     juce::Justification::centred, 1);
}

void GuitarToneEmulatorEditor::resized()
{
    const int margin = 10;
    const int buttonHeight = 30;
    const int sliderSize = 100;
    
    // Layout mode buttons
    auto buttonArea = getLocalBounds().removeFromTop(buttonHeight).reduced(margin);
    ampModeButton.setBounds(buttonArea.removeFromLeft(buttonArea.getWidth() / 2).reduced(margin));
    toneMatchButton.setBounds(buttonArea.reduced(margin));
    
    // Layout visualizer
    auto visualizerArea = getLocalBounds().removeFromTop(getHeight() / 2).reduced(margin);
    visualizer.setBounds(visualizerArea);
    
    // Layout controls
    auto controlArea = getLocalBounds().reduced(margin);
    const int sliderSpacing = controlArea.getWidth() / 3;
    
    // Gain control
    auto gainArea = controlArea.removeFromLeft(sliderSpacing);
    gainSlider.setBounds(gainArea.removeFromTop(sliderSize));
    gainLabel.setBounds(gainArea.removeFromTop(30));
    
    // Tone control
    auto toneArea = controlArea.removeFromLeft(sliderSpacing);
    toneSlider.setBounds(toneArea.removeFromTop(sliderSize));
    toneLabel.setBounds(toneArea.removeFromTop(30));
    
    // Presence control
    auto presenceArea = controlArea;
    presenceSlider.setBounds(presenceArea.removeFromTop(sliderSize));
    presenceLabel.setBounds(presenceArea.removeFromTop(30));
}

void GuitarToneEmulatorEditor::timerCallback()
{
    // Update visualizer with current audio data
    if (processor.getMode() == Mode::ToneMatching)
    {
        visualizer.repaint();
    }
    else
    {
        visualizer.repaint();
    }
}

void GuitarToneEmulatorEditor::setMode(Mode mode)
{
    processor.setMode(mode);
    
    // Update button states
    ampModeButton.setToggleState(mode == Mode::AmpSimulation, juce::dontSendNotification);
    toneMatchButton.setToggleState(mode == Mode::ToneMatching, juce::dontSendNotification);
}

} // namespace GuitarToneEmulator 