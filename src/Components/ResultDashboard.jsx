import React from 'react';
import EmotionBarChart from './EmotionBarChart';
import WaveformPlot from './WaveformPlot';
import SpectrogramPlot from './SpectrogramPlot';

function ResultDashboard({ emotionResult, audioFeatures }) {
  return (
    <div className="space-y-8">
      {/* Emotion Bar Chart */}
      <EmotionBarChart emotionResult={emotionResult} />
      
      {/* Waveform Plot */}
      <WaveformPlot audioFeatures={audioFeatures} />
      
      {/* Spectrogram Plot */}
      <SpectrogramPlot audioFeatures={audioFeatures} />
    </div>
  );
}

export default ResultDashboard;
