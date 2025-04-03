#include <JuceHeader.h>
#include "Audio/AudioProcessor.h"
#include "Features/FeatureExtractor.h"
#include "Neural/ToneMatcher.h"

using namespace GuitarToneEmulator;

int main() {
    // Initialize JUCE audio format manager
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();
    
    // Load input file
    juce::File inputFile("/Users/peterarango/cursor experiments/GuitarToneEmulator/Test/Test/input_1.wav");
    std::unique_ptr<juce::AudioFormatReader> reader(
        formatManager.createReaderFor(inputFile)
    );
    
    if (!reader) {
        std::cerr << "Error: Could not load input file" << std::endl;
        return 1;
    }
    
    // Create audio buffer for the entire file
    juce::AudioBuffer<float> fileBuffer(
        reader->numChannels,
        static_cast<int>(reader->lengthInSamples)
    );
    
    // Read the entire file into the buffer
    reader->read(&fileBuffer, 
                 0, 
                 static_cast<int>(reader->lengthInSamples),
                 0, 
                 true, 
                 true);
    
    // Initialize our processing chain
    GuitarToneEmulator::AudioProcessor processor;
    processor.prepare(reader->sampleRate, 1024); // Process in blocks of 1024 samples
    
    // Process the audio in blocks
    const int blockSize = 1024;
    const int numBlocks = fileBuffer.getNumSamples() / blockSize;
    
    std::cout << "Processing audio..." << std::endl;
    
    for (int block = 0; block < numBlocks; ++block) {
        // Create a temporary buffer for this block
        juce::AudioBuffer<float> blockBuffer(fileBuffer.getNumChannels(), blockSize);
        
        // Copy data from the file buffer to the block buffer
        for (int channel = 0; channel < blockBuffer.getNumChannels(); ++channel) {
            blockBuffer.copyFrom(channel, 0,
                               fileBuffer, channel,
                               block * blockSize,
                               blockSize);
        }
        
        // Process the block
        processor.processBlock(blockBuffer);
        
        // Copy processed data back to the file buffer
        for (int channel = 0; channel < blockBuffer.getNumChannels(); ++channel) {
            fileBuffer.copyFrom(channel,
                              block * blockSize,
                              blockBuffer, channel,
                              0, blockSize);
        }
    }
    
    // Save the processed audio
    juce::File outputFile("/Users/peterarango/cursor experiments/GuitarToneEmulator/Test/Test/output_processed.wav");
    std::unique_ptr<juce::AudioFormatWriter> writer(
        formatManager.findFormatForFileExtension("wav")->createWriterFor(
            new juce::FileOutputStream(outputFile),
            reader->sampleRate,
            reader->numChannels,
            24, // bit depth
            {}, 0
        )
    );
    
    if (writer) {
        writer->writeFromAudioSampleBuffer(fileBuffer, 0, fileBuffer.getNumSamples());
    } else {
        std::cerr << "Error: Could not create output file" << std::endl;
        return 1;
    }
    
    std::cout << "Processing complete. Output saved to: " << outputFile.getFullPathName() << std::endl;
    return 0;
} 