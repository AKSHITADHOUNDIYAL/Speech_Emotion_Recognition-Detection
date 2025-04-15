import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import * as tf from "@tensorflow/tfjs";

export default function SpectrogramPlot({ audioFeatures }) {
  const [spectrogram, setSpectrogram] = useState(null);

  useEffect(() => {
    if (!audioFeatures?.waveform) return;

    const computeSpectrogram = async () => {
      const y = audioFeatures.waveform;
      const sr = audioFeatures.sampling_rate || 22050;

      const stftSize = 1024;
      const hopLength = 512;
      const spec = [];

      for (let i = 0; i + stftSize < y.length; i += hopLength) {
        const window = y.slice(i, i + stftSize);
        const windowed = tf.tensor1d(window).mul(tf.signal.hammingWindow(stftSize));
        const fft = tf.spectral.fft(windowed).abs().arraySync();
        spec.push(fft.slice(0, stftSize / 2)); // Take half (real spectrum)
      }

      setSpectrogram(spec);
    };

    computeSpectrogram();
  }, [audioFeatures]);

  if (!spectrogram) return <p>Loading spectrogram...</p>;

  return (
    <Plot
      data={[
        {
          z: spectrogram,
          type: "heatmap",
          colorscale: "Viridis"
        }
      ]}
      layout={{
        xaxis: { title: "Time" },
        yaxis: { title: "Frequency Bin" },
        margin: { l: 20, r: 20, t: 30, b: 20 },
        height: 400
      }}
    />
  );
}
