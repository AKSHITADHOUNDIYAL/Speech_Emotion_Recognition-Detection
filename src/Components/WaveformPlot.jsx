import React from "react";
import Plot from "react-plotly.js";

export default function WaveformPlot({ audioFeatures }) {
  if (!audioFeatures?.waveform) {
    return <p>No waveform data available</p>;
  }

  const y = audioFeatures.waveform;
  const sr = audioFeatures.sampling_rate || 22050;
  let time = Array.from({ length: y.length }, (_, i) => i / sr);

  // Reduce points if too long
  if (y.length > 10000) {
    const step = Math.floor(y.length / 10000);
    time = time.filter((_, i) => i % step === 0);
  }

  return (
    <Plot
      data={[
        {
          x: time,
          y: y,
          type: "scatter",
          mode: "lines",
          line: { color: "#1f77b4", width: 1 },
          name: "Waveform"
        }
      ]}
      layout={{
        xaxis: { title: "Time (seconds)" },
        yaxis: { title: "Amplitude" },
        margin: { l: 20, r: 20, t: 30, b: 20 },
        height: 400
      }}
    />
  );
}
