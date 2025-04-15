import React from "react";
import Plot from "react-plotly.js";

export default function EmotionBarChart({ emotionResult }) {
  if (!emotionResult) return null;

  const sortedEmotions = Object.entries(emotionResult).sort((a, b) => b[1] - a[1]);
  const emotions = sortedEmotions.map(([emotion]) => emotion.charAt(0).toUpperCase() + emotion.slice(1));
  const probabilities = sortedEmotions.map(([_, value]) => value);

  const emotionColors = {
    Happy: '#FFD700',
    Sad: '#4169E1',
    Angry: '#FF4500',
    Neutral: '#808080',
    Fear: '#800080',
    Surprise: '#FFA500',
    Disgust: '#006400'
  };

  const colors = emotions.map(emotion => emotionColors[emotion] || '#1f77b4');

  return (
    <Plot
      data={[
        {
          x: emotions,
          y: probabilities,
          type: "bar",
          marker: { color: colors },
          text: probabilities.map(p => p.toFixed(2)),
          textposition: "auto"
        }
      ]}
      layout={{
        yaxis: { title: "Probability", range: [0, 1] },
        xaxis: { title: "Emotion" },
        margin: { l: 20, r: 20, t: 30, b: 20 },
        height: 400
      }}
    />
  );
}
