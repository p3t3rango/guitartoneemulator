#include "FeatureExtractor.h"

namespace GuitarToneEmulator {

FeatureExtractor::FeatureExtractor()
    : fftSize(2048)
    , sampleRate(44100.0)
    , samplesPerBlock(512)
    , rmsLevel(0.0f)
    , zeroCrossingRate(0.0f)
    , spectralCentroid(0.0f)
    , spectralSpread(0.0f)
    , spectralFlatness(0.0f)
    , modulationIndex(0.0f)
    , modulationFrequency(0.0f)
    , modulationDepth(0.0f)
    , attackTime(0.0f)
    , decayTime(0.0f)
    , sustainLevel(0.0f)
    , releaseTime(0.0f)
    , envelopeFollower(0.0f)
    , resonance(0.0f)
    , harmonicContent(0.0f)
    , fundamental(0.0f)
    , resonanceBands{{
        FrequencyBand(80.0f, 200.0f),    // Low
        FrequencyBand(200.0f, 800.0f),   // Low-mid
        FrequencyBand(800.0f, 2500.0f),  // Mid
        FrequencyBand(2500.0f, 8000.0f)  // High
    }}
    , frequencyBands{{
        {20.0f, 150.0f},    // Bass
        {150.0f, 400.0f},   // Low-mid
        {400.0f, 1000.0f},  // Mid
        {1000.0f, 2500.0f}, // High-mid
        {2500.0f, 20000.0f} // Treble
    }}
{
    prepareFFT();
}

void FeatureExtractor::prepare(double newSampleRate, int newBlockSize)
{
    sampleRate = newSampleRate;
    samplesPerBlock = newBlockSize;
    prepareFFT();
}

void FeatureExtractor::prepareFFT()
{
    fft = std::make_unique<juce::dsp::FFT>(std::log2(fftSize));
    window = std::make_unique<juce::dsp::WindowingFunction<float>>(fftSize, juce::dsp::WindowingFunction<float>::hann);
    
    fftBuffer.resize(fftSize * 2); // Complex numbers need twice the space
    magnitudeSpectrum.resize(fftSize / 2 + 1);
    spectrum.resize(fftSize);
    previousMagnitudeSpectrum.resize(fftSize / 2 + 1);
    previousFFT.resize(fftSize);
    envelopeBuffer.resize(fftSize);
}

void FeatureExtractor::processBlock(const juce::AudioBuffer<float>& buffer)
{
    const float* channelData = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();

    // Copy data to FFT buffer
    for (int i = 0; i < numSamples && i < fftSize; ++i)
        spectrum[i] = channelData[i];

    // Zero-pad if needed
    if (numSamples < fftSize)
        std::fill(spectrum.begin() + numSamples, spectrum.end(), 0.0f);

    // Apply window function
    window->multiplyWithWindowingTable(spectrum.data(), fftSize);

    // Perform FFT
    performFFT(spectrum);
}

void FeatureExtractor::performFFT(const std::vector<float>& input)
{
    // Copy input to complex FFT buffer
    for (size_t i = 0; i < fftSize; ++i)
    {
        fftBuffer[i * 2] = input[i];
        fftBuffer[i * 2 + 1] = 0.0f;
    }

    // Perform FFT
    fft->perform(reinterpret_cast<const juce::dsp::Complex<float>*>(fftBuffer.data()),
                 reinterpret_cast<juce::dsp::Complex<float>*>(fftBuffer.data()),
                 false);

    // Calculate magnitude spectrum
    for (size_t i = 0; i <= fftSize / 2; ++i)
    {
        float real = fftBuffer[i * 2];
        float imag = fftBuffer[i * 2 + 1];
        magnitudeSpectrum[i] = std::sqrt(real * real + imag * imag);
    }
}

void FeatureExtractor::calculateRMS(const juce::AudioBuffer<float>& buffer)
{
    float sum = 0.0f;
    const float* data = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();
    
    for (int i = 0; i < numSamples; ++i)
        sum += data[i] * data[i];
    
    rmsLevel = std::sqrt(sum / numSamples);
}

void FeatureExtractor::calculateZeroCrossingRate(const juce::AudioBuffer<float>& buffer)
{
    int crossings = 0;
    const float* data = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();
    
    for (int i = 1; i < numSamples; ++i)
        if ((data[i] >= 0.0f && data[i - 1] < 0.0f) ||
            (data[i] < 0.0f && data[i - 1] >= 0.0f))
            ++crossings;
    
    zeroCrossingRate = static_cast<float>(crossings) / numSamples;
}

void FeatureExtractor::calculateSpectralFeatures(const juce::AudioBuffer<float>& buffer)
{
    float weightedSum = 0.0f;
    float totalEnergy = 0.0f;
    
    for (size_t i = 0; i <= fftSize / 2; ++i)
    {
        float frequency = i * sampleRate / fftSize;
        float magnitude = magnitudeSpectrum[i];
        float energy = magnitude * magnitude;
        
        weightedSum += frequency * energy;
        totalEnergy += energy;
    }
    
    // Calculate spectral centroid
    spectralCentroid = weightedSum / (totalEnergy + 1e-6f);
    
    // Calculate spectral spread
    float spreadSum = 0.0f;
    for (size_t i = 0; i <= fftSize / 2; ++i)
    {
        float frequency = i * sampleRate / fftSize;
        float magnitude = magnitudeSpectrum[i];
        float energy = magnitude * magnitude;
        float diff = frequency - spectralCentroid;
        spreadSum += diff * diff * energy;
    }
    spectralSpread = std::sqrt(spreadSum / (totalEnergy + 1e-6f));
    
    // Calculate spectral flatness
    float geometricMean = 0.0f;
    float arithmeticMean = 0.0f;
    int nonZeroCount = 0;
    
    for (size_t i = 0; i <= fftSize / 2; ++i)
    {
        float magnitude = magnitudeSpectrum[i];
        if (magnitude > 1e-6f)
        {
            geometricMean += std::log(magnitude);
            arithmeticMean += magnitude;
            ++nonZeroCount;
        }
    }
    
    if (nonZeroCount > 0)
    {
        geometricMean = std::exp(geometricMean / nonZeroCount);
        arithmeticMean /= nonZeroCount;
        spectralFlatness = geometricMean / (arithmeticMean + 1e-6f);
    }
    else
    {
        spectralFlatness = 0.0f;
    }
}

void FeatureExtractor::calculateModulationFeatures(const juce::AudioBuffer<float>& buffer)
{
    // Simple amplitude modulation detection
    const float* data = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();
    
    // Calculate envelope
    std::vector<float> envelope(numSamples);
    for (int i = 0; i < numSamples; ++i)
        envelope[i] = std::abs(data[i]);
    
    // Find modulation rate and depth
    float maxEnv = 0.0f;
    float minEnv = std::numeric_limits<float>::max();
    int zeroCrossings = 0;
    
    for (int i = 0; i < numSamples; ++i)
    {
        maxEnv = std::max(maxEnv, envelope[i]);
        minEnv = std::min(minEnv, envelope[i]);
        
        if (i > 0 && ((envelope[i] >= 0.0f && envelope[i - 1] < 0.0f) ||
                      (envelope[i] < 0.0f && envelope[i - 1] >= 0.0f)))
            ++zeroCrossings;
    }
    
    modulationDepth = (maxEnv - minEnv) / (maxEnv + 1e-6f);
    modulationFrequency = static_cast<float>(zeroCrossings) * sampleRate / (2.0f * numSamples);
    modulationIndex = modulationDepth * modulationFrequency;
}

void FeatureExtractor::calculateHarmonicContent(const juce::AudioBuffer<float>& buffer)
{
    // Find fundamental frequency with improved pitch detection
    float fundamental = findFundamentalFrequencyYIN(buffer);
    
    // Phase-aligned harmonic analysis
    analyzeHarmonicsPhaseAligned(buffer, fundamental);
    
    // Calculate harmonic-to-noise ratio with improved method
    harmonicContent = calculateHarmonicToNoiseRatioImproved(buffer);
}

float FeatureExtractor::findFundamentalFrequencyYIN(const juce::AudioBuffer<float>& buffer)
{
    const int bufferSize = buffer.getNumSamples();
    std::vector<float> difference(bufferSize/2);
    std::vector<float> cumulativeMean(bufferSize/2);
    
    // Step 2: Difference function
    for (int tau = 0; tau < bufferSize/2; tau++) {
        float sum = 0.0f;
        for (int j = 0; j < bufferSize/2; j++) {
            float diff = buffer.getSample(0, j) - buffer.getSample(0, j + tau);
            sum += diff * diff;
        }
        difference[tau] = sum;
    }
    
    // Step 3: Cumulative mean normalized difference
    float runningSum = 0.0f;
    cumulativeMean[0] = 1.0f;
    for (int tau = 1; tau < bufferSize/2; tau++) {
        runningSum += difference[tau];
        cumulativeMean[tau] = difference[tau] * tau / runningSum;
    }
    
    // Step 4: Absolute threshold
    const float threshold = 0.1f;
    int minTau = 0;
    for (int tau = 2; tau < bufferSize/2; tau++) {
        if (cumulativeMean[tau] < threshold) {
            minTau = tau;
            break;
        }
    }
    
    // Step 5: Parabolic interpolation
    if (minTau != 0 && minTau < bufferSize/2 - 1) {
        float alpha = cumulativeMean[minTau-1];
        float beta = cumulativeMean[minTau];
        float gamma = cumulativeMean[minTau+1];
        float p = 0.5f * (alpha - gamma) / (alpha - 2*beta + gamma);
        minTau += p;
    }
    
    // Convert to frequency
    float sampleRate = 44100.0f; // TODO: Get actual sample rate
    return sampleRate / minTau;
}

void FeatureExtractor::analyzeHarmonicsPhaseAligned(const juce::AudioBuffer<float>& buffer, float fundamental)
{
    const int numHarmonics = 10;
    const float sampleRate = 44100.0f; // TODO: Get actual sample rate
    const int bufferSize = buffer.getNumSamples();
    
    harmonicAmplitudes.resize(numHarmonics);
    harmonicPhases.resize(numHarmonics);
    
    // Convert buffer to complex spectrum
    std::vector<juce::dsp::Complex<float>> spectrum(bufferSize);
    for (int i = 0; i < bufferSize; ++i) {
        spectrum[i].real(buffer.getSample(0, i));
        spectrum[i].imag(0.0f);
    }
    
    // Apply Hann window
    for (int i = 0; i < bufferSize; ++i) {
        float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (bufferSize - 1)));
        spectrum[i].real(spectrum[i].real() * window);
        spectrum[i].imag(spectrum[i].imag() * window);
    }
    
    // Perform FFT
    fft->perform(spectrum.data(), spectrum.data(), false);
    
    // Analyze harmonics
    float freqResolution = sampleRate / bufferSize;
    for (int h = 1; h <= numHarmonics; ++h) {
        float harmonicFreq = h * fundamental;
        int centerBin = static_cast<int>(harmonicFreq / freqResolution);
        
        if (centerBin >= bufferSize/2) continue;
        
        // Find peak using quadratic interpolation
        auto& prev = (centerBin > 0) ? spectrum[centerBin-1] : spectrum[centerBin];
        auto& curr = spectrum[centerBin];
        auto& next = (centerBin < bufferSize-1) ? spectrum[centerBin+1] : spectrum[centerBin];
        
        float alpha = std::sqrt(prev.real() * prev.real() + prev.imag() * prev.imag());
        float beta = std::sqrt(curr.real() * curr.real() + curr.imag() * curr.imag());
        float gamma = std::sqrt(next.real() * next.real() + next.imag() * next.imag());
        
        float p = 0.5f * (alpha - gamma) / (alpha - 2*beta + gamma);
        
        harmonicAmplitudes[h-1] = beta * (1.0f - p * p);
        harmonicPhases[h-1] = std::atan2(curr.imag(), curr.real());
    }
}

float FeatureExtractor::calculateHarmonicToNoiseRatioImproved(const juce::AudioBuffer<float>& buffer)
{
    const float sampleRate = 44100.0f; // TODO: Get actual sample rate
    const int bufferSize = buffer.getNumSamples();
    float harmonicEnergy = 0.0f;
    float noiseEnergy = 0.0f;
    
    // Sum energy of detected harmonics
    for (size_t h = 0; h < harmonicAmplitudes.size(); ++h) {
        float harmonicFreq = fundamental * (h + 1);
        harmonicEnergy += harmonicAmplitudes[h] * harmonicAmplitudes[h];
    }
    
    // Calculate noise energy (total energy minus harmonic energy)
    float totalEnergy = 0.0f;
    for (int i = 0; i < bufferSize; ++i) {
        float sample = buffer.getSample(0, i);
        totalEnergy += sample * sample;
    }
    
    noiseEnergy = totalEnergy - harmonicEnergy;
    if (noiseEnergy < 1e-6f) noiseEnergy = 1e-6f;
    
    return 10.0f * std::log10(harmonicEnergy / noiseEnergy);
}

void FeatureExtractor::calculateResonance(const juce::AudioBuffer<float>& buffer)
{
    const float freqResolution = static_cast<float>(sampleRate) / fftSize;
    
    // Calculate resonance for each frequency band
    for (auto& band : resonanceBands)
    {
        int startBin = static_cast<int>(band.lowFreq / freqResolution);
        int endBin = static_cast<int>(band.highFreq / freqResolution);
        
        float maxPeak = 0.0f;
        float avgEnergy = 0.0f;
        int numBins = 0;
        
        for (int bin = startBin; bin <= endBin && bin < magnitudeSpectrum.size(); ++bin)
        {
            float mag = magnitudeSpectrum[bin];
            maxPeak = std::max(maxPeak, mag);
            avgEnergy += mag;
            numBins++;
        }
        
        if (numBins > 0)
        {
            avgEnergy /= numBins;
            band.resonance = maxPeak / (avgEnergy + 1e-6f);
        }
    }
    
    // Calculate overall resonance as weighted sum of band resonances
    resonance = 0.0f;
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < resonanceBands.size(); ++i)
    {
        float weight = 1.0f / (i + 1);  // Give more weight to lower bands
        resonance += resonanceBands[i].resonance * weight;
        totalWeight += weight;
    }
    
    resonance /= totalWeight;
}

void FeatureExtractor::calculateEnvelopeFeatures(const juce::AudioBuffer<float>& buffer)
{
    const float* data = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();
    
    // Multi-stage envelope following
    const float fastAttackTime = 0.001f;  // 1ms
    const float slowAttackTime = 0.010f;  // 10ms
    const float fastReleaseTime = 0.050f; // 50ms
    const float slowReleaseTime = 0.100f; // 100ms
    
    float fastEnvelope = 0.0f;
    float slowEnvelope = 0.0f;
    
    const float fastAttackCoeff = std::exp(-1.0f / (fastAttackTime * sampleRate));
    const float slowAttackCoeff = std::exp(-1.0f / (slowAttackTime * sampleRate));
    const float fastReleaseCoeff = std::exp(-1.0f / (fastReleaseTime * sampleRate));
    const float slowReleaseCoeff = std::exp(-1.0f / (slowReleaseTime * sampleRate));
    
    std::vector<float> envelope(numSamples);
    std::vector<float> transientDetector(numSamples);
    
    // Process envelopes
    for (int i = 0; i < numSamples; ++i)
    {
        float input = std::abs(data[i]);
        
        // Fast envelope
        if (input > fastEnvelope)
            fastEnvelope = fastAttackCoeff * (fastEnvelope - input) + input;
        else
            fastEnvelope = fastReleaseCoeff * (fastEnvelope - input) + input;
            
        // Slow envelope
        if (input > slowEnvelope)
            slowEnvelope = slowAttackCoeff * (slowEnvelope - input) + input;
        else
            slowEnvelope = slowReleaseCoeff * (slowEnvelope - input) + input;
            
        // Transient detection
        transientDetector[i] = fastEnvelope / (slowEnvelope + 1e-6f);
        
        // Store transients when ratio exceeds threshold
        if (transientDetector[i] > 1.5f && (transients.empty() || i - transients.back().position > 1000)) {
            TemporalFeatures::Transient transient;
            transient.position = static_cast<float>(i) / sampleRate;
            transient.strength = transientDetector[i];
            transient.frequency = 0.0f; // TODO: Calculate frequency
            transients.push_back(transient);
        }
        
        // Final envelope
        envelope[i] = fastEnvelope;
    }
    
    // Analyze envelope segments
    float maxEnv = 0.0f;
    int attackEnd = 0;
    bool inAttack = true;
    
    for (int i = 0; i < numSamples; ++i)
    {
        if (envelope[i] > maxEnv)
        {
            maxEnv = envelope[i];
            if (!inAttack)
            {
                inAttack = true;
                attackEnd = i;
            }
        }
        else if (inAttack && envelope[i] < maxEnv * 0.9f)
        {
            inAttack = false;
        }
    }
    
    // Calculate ADSR parameters
    attackTime = static_cast<float>(attackEnd) / sampleRate;
    
    int decayEnd = attackEnd;
    while (decayEnd < numSamples && envelope[decayEnd] > maxEnv * 0.3f)
        ++decayEnd;
    
    decayTime = static_cast<float>(decayEnd - attackEnd) / sampleRate;
    
    // Calculate sustain level as average of middle portion
    float sustainSum = 0.0f;
    int sustainCount = 0;
    for (int i = decayEnd; i < numSamples * 3/4; ++i)
    {
        sustainSum += envelope[i];
        ++sustainCount;
    }
    sustainLevel = sustainCount > 0 ? sustainSum / (sustainCount * maxEnv) : 0.0f;
    
    // Calculate release time
    int releaseStart = numSamples * 3/4;
    while (releaseStart > 0 && envelope[releaseStart] > maxEnv * 0.1f)
        --releaseStart;
    
    releaseTime = static_cast<float>(numSamples - releaseStart) / sampleRate;
    
    // Store envelope for visualization or further analysis
    envelopeBuffer = envelope;
}

void FeatureExtractor::analyzeToneCharacteristics(ToneFeatures& features)
{
    // Calculate brightness (ratio of high frequency to total energy)
    float totalEnergy = 0.0f;
    float highFreqEnergy = 0.0f;
    const float brightnessCutoff = 5000.0f; // Hz
    
    for (size_t i = 0; i <= fftSize / 2; ++i)
    {
        float frequency = i * sampleRate / fftSize;
        float energy = magnitudeSpectrum[i] * magnitudeSpectrum[i];
        
        totalEnergy += energy;
        if (frequency > brightnessCutoff)
            highFreqEnergy += energy;
    }
    
    features.brightness = highFreqEnergy / (totalEnergy + 1e-6f);
    features.warmth = 1.0f - features.brightness;
    features.presence = spectralFlatness;
    features.clarity = 1.0f - spectralSpread / (sampleRate / 4.0f);
    features.fullness = rmsLevel;
}

SpectralFeatures FeatureExtractor::analyzeSpectralContent(const juce::AudioBuffer<float>& buffer)
{
    SpectralFeatures features;
    
    // Process the buffer to get the spectrum
    processBlock(buffer);
    
    // Calculate spectral features
    calculateSpectralFeatures(buffer);
    
    // Copy calculated features
    features.centroid = spectralCentroid;
    features.spread = spectralSpread;
    features.flatness = spectralFlatness;
    
    // Calculate frequency band energies
    features.frequencyBandEnergies.resize(frequencyBands.size());
    for (size_t band = 0; band < frequencyBands.size(); ++band)
    {
        float bandEnergy = 0.0f;
        float minFreq = frequencyBands[band].first;
        float maxFreq = frequencyBands[band].second;
        
        for (size_t i = 0; i <= fftSize / 2; ++i)
        {
            float frequency = i * sampleRate / fftSize;
            if (frequency >= minFreq && frequency <= maxFreq)
            {
                float magnitude = magnitudeSpectrum[i];
                bandEnergy += magnitude * magnitude;
            }
        }
        
        features.frequencyBandEnergies[band] = bandEnergy;
    }
    
    // Calculate harmonic content
    const int numHarmonics = 10;
    features.harmonicAmplitudes.resize(numHarmonics);
    features.harmonicPhases.resize(numHarmonics);
    
    // Find fundamental frequency (using simple peak detection)
    float maxMagnitude = 0.0f;
    size_t fundamentalBin = 0;
    
    for (size_t i = 1; i <= fftSize / 2; ++i)
    {
        if (magnitudeSpectrum[i] > maxMagnitude)
        {
            maxMagnitude = magnitudeSpectrum[i];
            fundamentalBin = i;
        }
    }
    
    float fundamentalFreq = fundamentalBin * sampleRate / fftSize;
    
    // Extract harmonics
    float totalHarmonicEnergy = 0.0f;
    for (int h = 0; h < numHarmonics; ++h)
    {
        float harmonicFreq = fundamentalFreq * (h + 1);
        size_t harmonicBin = static_cast<size_t>(harmonicFreq * fftSize / sampleRate);
        
        if (harmonicBin <= fftSize / 2)
        {
            features.harmonicAmplitudes[h] = magnitudeSpectrum[harmonicBin];
            features.harmonicPhases[h] = std::atan2(fftBuffer[harmonicBin * 2 + 1], 
                                                   fftBuffer[harmonicBin * 2]);
            totalHarmonicEnergy += features.harmonicAmplitudes[h];
        }
    }
    
    // Calculate harmonic balance (ratio of harmonic energy to total energy)
    float totalEnergy = 0.0f;
    for (size_t i = 0; i <= fftSize / 2; ++i)
        totalEnergy += magnitudeSpectrum[i];
    
    features.harmonicBalance = totalHarmonicEnergy / (totalEnergy + 1e-6f);
    
    return features;
}

TemporalFeatures FeatureExtractor::analyzeTemporalContent(const juce::AudioBuffer<float>& buffer)
{
    TemporalFeatures features;
    
    // Calculate basic temporal features
    calculateRMS(buffer);
    calculateZeroCrossingRate(buffer);
    
    features.rms = rmsLevel;
    features.zeroCrossingRate = zeroCrossingRate;
    
    // Analyze envelope
    const float* data = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();
    
    // Calculate envelope using simple peak detection
    std::vector<float> envelope(numSamples);
    for (int i = 0; i < numSamples; ++i)
        envelope[i] = std::abs(data[i]);
    
    // Find attack time (time to reach 90% of peak)
    float peakLevel = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        peakLevel = std::max(peakLevel, envelope[i]);
    
    float attackThreshold = 0.9f * peakLevel;
    int attackSamples = 0;
    while (attackSamples < numSamples && envelope[attackSamples] < attackThreshold)
        ++attackSamples;
    
    features.attackTime = static_cast<float>(attackSamples) / sampleRate;
    
    // Find decay time (time to fall to sustain level)
    float sustainLevel = 0.0f;
    for (int i = numSamples / 2; i < numSamples; ++i)
        sustainLevel += envelope[i];
    sustainLevel /= (numSamples / 2);
    
    int decaySamples = attackSamples;
    while (decaySamples < numSamples && envelope[decaySamples] > sustainLevel * 1.1f)
        ++decaySamples;
    
    features.decayTime = static_cast<float>(decaySamples - attackSamples) / sampleRate;
    features.sustainLevel = sustainLevel / peakLevel;
    
    // Calculate dynamic range
    float minLevel = envelope[0];
    for (int i = 1; i < numSamples; ++i)
        minLevel = std::min(minLevel, envelope[i]);
    
    features.dynamicRange = 20.0f * std::log10((peakLevel + 1e-6f) / (minLevel + 1e-6f));
    
    // Detect transients
    const float transientThreshold = 0.5f * peakLevel;
    const int minTransientDistance = static_cast<int>(0.05f * sampleRate); // 50ms
    
    int lastTransientPos = -minTransientDistance;
    for (int i = 1; i < numSamples - 1; ++i)
    {
        if (i - lastTransientPos >= minTransientDistance &&
            envelope[i] > transientThreshold &&
            envelope[i] > envelope[i - 1] &&
            envelope[i] >= envelope[i + 1])
        {
            TemporalFeatures::Transient transient;
            transient.position = static_cast<float>(i) / sampleRate;
            transient.strength = envelope[i] / peakLevel;
            
            // Estimate frequency using zero crossings around the transient
            int zeroCrossings = 0;
            for (int j = std::max(0, i - 100); j < std::min(numSamples - 1, i + 100); ++j)
            {
                if ((data[j] >= 0.0f && data[j + 1] < 0.0f) ||
                    (data[j] < 0.0f && data[j + 1] >= 0.0f))
                    ++zeroCrossings;
            }
            transient.frequency = zeroCrossings * sampleRate / 200.0f;
            
            features.transients.push_back(transient);
            lastTransientPos = i;
        }
    }
    
    return features;
}

ModulationFeatures FeatureExtractor::analyzeModulationContent(const juce::AudioBuffer<float>& buffer)
{
    ModulationFeatures features;
    
    // Calculate modulation features
    calculateModulationFeatures(buffer);
    
    features.index = modulationIndex;
    features.frequency = modulationFrequency;
    features.depth = modulationDepth;
    
    // Analyze vibrato (frequency modulation)
    const float* data = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();
    
    // Calculate zero-crossing rate in small windows to track pitch variations
    const int windowSize = 1024;
    std::vector<float> pitchTrack;
    
    for (int i = 0; i < numSamples - windowSize; i += windowSize / 2)
    {
        int zeroCrossings = 0;
        for (int j = i; j < i + windowSize - 1; ++j)
        {
            if ((data[j] >= 0.0f && data[j + 1] < 0.0f) ||
                (data[j] < 0.0f && data[j + 1] >= 0.0f))
                ++zeroCrossings;
        }
        pitchTrack.push_back(zeroCrossings * sampleRate / windowSize);
    }
    
    // Calculate vibrato rate and depth
    if (pitchTrack.size() > 2)
    {
        float meanPitch = 0.0f;
        float maxPitch = pitchTrack[0];
        float minPitch = pitchTrack[0];
        
        for (float pitch : pitchTrack)
        {
            meanPitch += pitch;
            maxPitch = std::max(maxPitch, pitch);
            minPitch = std::min(minPitch, pitch);
        }
        meanPitch /= pitchTrack.size();
        
        features.vibratoDepth = (maxPitch - minPitch) / (2.0f * meanPitch);
        
        // Count pitch oscillations
        int oscillations = 0;
        for (size_t i = 1; i < pitchTrack.size() - 1; ++i)
        {
            if ((pitchTrack[i] > pitchTrack[i - 1] && pitchTrack[i] > pitchTrack[i + 1]) ||
                (pitchTrack[i] < pitchTrack[i - 1] && pitchTrack[i] < pitchTrack[i + 1]))
                ++oscillations;
        }
        
        float duration = static_cast<float>(numSamples) / sampleRate;
        features.vibratoRate = oscillations / (2.0f * duration);
    }
    
    // Analyze tremolo (amplitude modulation)
    std::vector<float> amplitudeEnvelope(numSamples);
    for (int i = 0; i < numSamples; ++i)
        amplitudeEnvelope[i] = std::abs(data[i]);
    
    float maxAmp = 0.0f;
    float minAmp = std::numeric_limits<float>::max();
    int tremoloOscillations = 0;
    
    for (int i = 1; i < numSamples - 1; ++i)
    {
        maxAmp = std::max(maxAmp, amplitudeEnvelope[i]);
        minAmp = std::min(minAmp, amplitudeEnvelope[i]);
        
        if ((amplitudeEnvelope[i] > amplitudeEnvelope[i - 1] && 
             amplitudeEnvelope[i] > amplitudeEnvelope[i + 1]) ||
            (amplitudeEnvelope[i] < amplitudeEnvelope[i - 1] && 
             amplitudeEnvelope[i] < amplitudeEnvelope[i + 1]))
            ++tremoloOscillations;
    }
    
    features.tremoloDepth = (maxAmp - minAmp) / (maxAmp + 1e-6f);
    float duration = static_cast<float>(numSamples) / sampleRate;
    features.tremoloRate = tremoloOscillations / (2.0f * duration);
    
    return features;
}

ToneFeatures FeatureExtractor::analyzeTone(const juce::AudioBuffer<float>& buffer)
{
    ToneFeatures features;
    
    // Process the audio block
    processBlock(buffer);
    
    // Calculate all features
    calculateRMS(buffer);
    calculateZeroCrossingRate(buffer);
    calculateSpectralFeatures(buffer);
    calculateModulationFeatures(buffer);
    calculateHarmonicContent(buffer);
    calculateResonance(buffer);
    calculateEnvelopeFeatures(buffer);
    
    // Fill in the features structure
    features.spectral.centroid = spectralCentroid;
    features.spectral.spread = spectralSpread;
    features.spectral.flatness = spectralFlatness;
    features.spectral.harmonicContent = harmonicContent;
    features.spectral.resonance = resonance;
    features.spectral.harmonicAmplitudes = harmonicAmplitudes;
    features.spectral.harmonicPhases = harmonicPhases;
    
    features.temporal.rms = rmsLevel;
    features.temporal.zeroCrossingRate = zeroCrossingRate;
    features.temporal.attackTime = attackTime;
    features.temporal.decayTime = decayTime;
    features.temporal.sustainLevel = sustainLevel;
    features.temporal.releaseTime = releaseTime;
    features.temporal.transients = transients;
    
    features.modulation.frequency = modulationFrequency;
    features.modulation.depth = modulationDepth;
    
    // Calculate derived features
    analyzeToneCharacteristics(features);
    
    return features;
}

FeatureExtractor::ToneType FeatureExtractor::classifyTone(const ToneFeatures& features)
{
    // Simple rule-based classification
    if (features.brightness > 0.7f && features.presence > 0.6f)
        return ToneType::Metal;
    else if (features.brightness > 0.5f && features.presence > 0.4f)
        return ToneType::Lead;
    else if (features.warmth > 0.6f && features.fullness > 0.5f)
        return ToneType::Crunch;
    else if (features.warmth > 0.7f && features.fullness > 0.7f)
        return ToneType::Bass;
    else if (features.clarity > 0.7f && features.presence > 0.5f)
        return ToneType::Acoustic;
    else
        return ToneType::Clean;
}

FeatureExtractor::Features FeatureExtractor::extractFeatures(const juce::AudioBuffer<float>& buffer) {
    // Convert AudioBuffer to vector for FFT processing
    std::vector<float> audioData;
    audioData.resize(buffer.getNumSamples());
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        audioData[i] = buffer.getSample(0, i);
    }
    
    // Calculate fundamental frequency
    fundamental = findFundamentalFrequencyYIN(buffer);
    
    // Analyze harmonics
    analyzeHarmonicsPhaseAligned(buffer, fundamental);
    
    // Calculate harmonic content
    harmonicContent = calculateHarmonicToNoiseRatioImproved(buffer);
    
    // Process frequency bands
    float sampleRate = 44100.0f; // TODO: Get actual sample rate
    float freqResolution = sampleRate / buffer.getNumSamples();
    
    std::vector<float> spectrum;
    spectrum.resize(buffer.getNumSamples());
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        spectrum[i] = buffer.getSample(0, i);
    }
    performFFT(spectrum);
    
    for (auto& band : resonanceBands) {
        int startBin = static_cast<int>(band.lowFreq / freqResolution);
        int endBin = static_cast<int>(band.highFreq / freqResolution);
        
        float maxPeak = 0.0f;
        float avgEnergy = 0.0f;
        int numBins = 0;
        
        for (int bin = startBin; bin <= endBin && bin < spectrum.size(); ++bin) {
            float magnitude = std::abs(spectrum[bin]);
            maxPeak = std::max(maxPeak, magnitude);
            avgEnergy += magnitude;
            numBins++;
        }
        
        if (numBins > 0) {
            avgEnergy /= numBins;
            band.resonance = maxPeak / (avgEnergy + 1e-6f);
        }
    }

    // Return the features
    Features features;
    const float* data = buffer.getReadPointer(0);
    const int numSamples = buffer.getNumSamples();
    
    features.spectral = spectralProcessor.process(data, numSamples);
    features.temporal = temporalProcessor.process(data, numSamples);
    features.modulation = modulationProcessor.process(data, numSamples);
    
    return features;
}

float FeatureExtractor::calculateResonance() const
{
    float resonance = 0.0f;
    float totalWeight = 0.0f;
    
    // Calculate weighted average of resonance values
    for (size_t i = 0; i < resonanceBands.size(); ++i) {
        float weight = 1.0f / (i + 1); // Higher weight for lower frequency bands
        resonance += resonanceBands[i].resonance * weight;
        totalWeight += weight;
    }
    
    return resonance / totalWeight;
}

std::vector<float> FeatureExtractor::SpectralProcessor::process(const float* data, int numSamples) {
    std::vector<float> features;
    features.reserve(6); // Storing 6 key spectral features
    
    // Prepare FFT buffer
    std::fill(fftBuffer.begin(), fftBuffer.end(), 0.0f);
    for (int i = 0; i < std::min(numSamples, currentFftSize); ++i) {
        fftBuffer[i * 2] = data[i];
    }
    
    // Apply Hann window
    for (int i = 0; i < currentFftSize; ++i) {
        float windowValue = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (currentFftSize - 1)));
        fftBuffer[i * 2] *= windowValue;
    }
    
    // Perform FFT
    fft->performRealOnlyForwardTransform(fftBuffer.data());
    
    // Calculate spectral features
    float centroid = 0.0f;
    float spread = 0.0f;
    float totalEnergy = 0.0f;
    float maxMagnitude = 0.0f;
    
    for (int i = 0; i < currentFftSize/2; ++i) {
        float real = fftBuffer[i * 2];
        float imag = fftBuffer[i * 2 + 1];
        float magnitude = std::sqrt(real * real + imag * imag);
        float frequency = i * sampleRate / currentFftSize;
        
        centroid += frequency * magnitude;
        totalEnergy += magnitude;
        maxMagnitude = std::max(maxMagnitude, magnitude);
    }
    
    centroid /= totalEnergy;
    
    features.push_back(centroid);                    // Spectral centroid
    features.push_back(spread / totalEnergy);        // Spectral spread
    features.push_back(maxMagnitude / totalEnergy);  // Peak ratio
    features.push_back(totalEnergy);                 // Total energy
    
    return features;
}

std::vector<float> FeatureExtractor::TemporalProcessor::process(const float* data, int numSamples) {
    std::vector<float> features;
    features.reserve(4); // Storing 4 key temporal features
    
    float rms = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        rms += data[i] * data[i];
    }
    rms = std::sqrt(rms / numSamples);
    
    float peak = 0.0f;
    int zeroCrossings = 0;
    float prevSample = 0.0f;
    
    for (int i = 0; i < numSamples; ++i) {
        peak = std::max(peak, std::abs(data[i]));
        if (i > 0 && (data[i] * prevSample) < 0) {
            zeroCrossings++;
        }
        prevSample = data[i];
    }
    
    features.push_back(rms);
    features.push_back(peak);
    features.push_back(static_cast<float>(zeroCrossings) / numSamples);
    features.push_back(peak > 0.01f ? rms / peak : 0.0f);
    
    return features;
}

std::vector<float> FeatureExtractor::ModulationProcessor::process(const float* data, int numSamples) {
    std::vector<float> features;
    features.reserve(3); // Storing 3 key modulation features
    
    std::vector<float> envelope(numSamples);
    float alpha = 0.1f;
    
    envelope[0] = std::abs(data[0]);
    for (int i = 1; i < numSamples; ++i) {
        envelope[i] = alpha * std::abs(data[i]) + (1.0f - alpha) * envelope[i-1];
    }
    
    float maxEnv = 0.0f;
    float minEnv = std::numeric_limits<float>::max();
    int modulations = 0;
    
    for (int i = 1; i < numSamples - 1; ++i) {
        maxEnv = std::max(maxEnv, envelope[i]);
        minEnv = std::min(minEnv, envelope[i]);
        
        if (envelope[i] > envelope[i-1] && envelope[i] > envelope[i+1]) {
            modulations++;
        }
    }
    
    float modulationDepth = (maxEnv - minEnv) / (maxEnv + 1e-6f);
    float modulationRate = static_cast<float>(modulations) * sampleRate / (2.0f * numSamples);
    
    features.push_back(modulationRate);
    features.push_back(modulationDepth);
    features.push_back(maxEnv);
    
    return features;
}

} // namespace GuitarToneEmulator 