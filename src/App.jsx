import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Tabs, Tab, TabList, TabPanel } from 'react-tabs';
import { motion } from 'framer-motion';
import 'react-tabs/style/react-tabs.css';
import ResultDashboard from './Components/ResultDashboard';
import './index.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileInfo, setFileInfo] = useState("");
  const [emotionResult, setEmotionResult] = useState(null);
  const [audioFeatures, setAudioFeatures] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState("default");
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const audioRef = useRef(null);

  const supportedFormats = ['wav', 'mp3', 'flac'];

  // Handle File Upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && !supportedFormats.includes(file.name.split('.').pop().toLowerCase())) {
      alert("Unsupported file format. Please upload a valid audio file.");
      return;
    }
    setUploadedFile(file);
    setFileInfo(`${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
  };

  // Analyze Audio (Upload)
  const analyzeAudio = async (customFile = null) => {
    const fileToAnalyze = customFile || uploadedFile;
    if (!fileToAnalyze) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append("audio_file", fileToAnalyze);

    try {
      const response = await axios.post("http://localhost:8000/analyze-audio", formData);
      setEmotionResult(response.data.emotions);
      setAudioFeatures(response.data.audio_features);
    } catch (error) {
      console.error("Error analyzing audio:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Start Recording
  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      setRecordedBlob(blob);
      audioRef.current.src = URL.createObjectURL(blob);
    };

    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  // Stop Recording
  const stopRecording = () => {
    mediaRecorder.stop();
    setRecording(false);
  };

  // Handle Recorded Audio Upload
  const analyzeRecordedAudio = () => {
    if (recordedBlob) {
      const file = new File([recordedBlob], "recorded_audio.webm");
      analyzeAudio(file);
    }
  };

  const generateBubbles = () => {
    const colors = [
      "rgba(255, 99, 132, 0.3)",
      "rgba(54, 162, 235, 0.3)",
      "rgba(255, 206, 86, 0.3)",
      "rgba(75, 192, 192, 0.3)",
      "rgba(153, 102, 255, 0.3)",
      "rgba(255, 159, 64, 0.3)",
    ];
  
    return Array.from({ length: 200 }).map((_, i) => {
      const size = Math.random() * 40 + 20;
      const left = Math.random() * 100;
      const duration = Math.random() * 10 + 10;
      const delay = Math.random() * 5;
      const color = colors[Math.floor(Math.random() * colors.length)];
      const blur = Math.random() > 0.7 ? "blur-md" : "";
  
      return (
        <li
          key={i}
          className={`absolute rounded-full animate-bubble ${blur}`}
          style={{
            width: `${size}px`,
            height: `${size}px`,
            left: `${left}%`,
            backgroundColor: color,
            animationDuration: `${duration}s`,
            animationDelay: `${delay}s`,
            bottom: '-100px',
            boxShadow: `0 0 10px ${color}`,
          }}
        ></li>
      );
    });
  };  

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-pink-900 via-purple-900 to-black text-white overflow-hidden">
    {/* Animated Bubbles in the background */}
    <ul className="bubbles absolute top-0 left-0 w-full h-full pointer-events-none z-0">
      {generateBubbles()}
    </ul>
  
    {/* Main Content */}
    <div className="main-content relative z-10 p-6 flex items-center justify-center flex-col min-h-screen">
      <motion.h1 
        className="text-4xl font-extrabold mb-3 text-center text-blue-700"
        initial={{ y: -30, opacity: 0 }} 
        animate={{ y: 0, opacity: 1 }}
      >
        🎭 Speech Emotion Recognition
      </motion.h1>
  
      <motion.p 
        className="mb-6 text-center text-white-600 text-lg"
        initial={{ y: 20, opacity: 0 }} 
        animate={{ y: 0, opacity: 1 }} 
        transition={{ delay: 0.2 }}
      >
        Upload or record audio to analyze emotions using our smart model.
      </motion.p>
  
        <Tabs>
          <TabList className="flex gap-2 mb-6 justify-center">
            <Tab>Upload Audio</Tab>
            <Tab>Record Audio</Tab>
            <Tab>Model Integration</Tab>
            <Tab>API Access</Tab>
          </TabList>
  
          {/* Upload Tab */}
          <TabPanel>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h2 className="text-xl font-semibold">Upload Audio File</h2>
              <p>Supported formats: {supportedFormats.join(", ")}</p>
              <input 
                type="file" 
                accept={supportedFormats.map(f => `.${f}`).join(",")} 
                onChange={handleFileUpload} 
                className="my-2"
              />
              {fileInfo && <p className="text-sm text-gray-600">{fileInfo}</p>}
              <button
                onClick={() => analyzeAudio()}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-all"
                disabled={isLoading}
              >
                {isLoading ? 'Analyzing...' : 'Analyze Emotions'}
              </button>
  
              {isLoading && <div className="mt-2 text-center text-blue-500">Loading...</div>}
  
              {emotionResult && (
                <motion.div className="mt-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <h3 className="font-semibold">Detected Emotions</h3>
                  <ul>
                    {Object.entries(emotionResult).map(([emotion, confidence]) => (
                      <li key={emotion}>{emotion}: {confidence.toFixed(2)}</li>
                    ))}
                  </ul>
                  <p className="mt-2 text-green-700 font-semibold">
                    Predominant emotion: {Object.entries(emotionResult).reduce((a, b) => a[1] > b[1] ? a : b)[0]}
                  </p>
                </motion.div>
              )}
  
              {emotionResult && audioFeatures && (
                <ResultDashboard 
                  emotionResult={emotionResult} 
                  audioFeatures={audioFeatures} 
                />
              )}
            </motion.div>
          </TabPanel>
  
          {/* Record Tab */}
          <TabPanel>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h2 className="text-xl font-semibold mb-4">Record Your Voice</h2>
              <div className="flex flex-col items-center gap-4">
                <audio ref={audioRef} controls className="mb-2" />
                {!recording ? (
                  <button
                    className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition"
                    onClick={startRecording}
                  >
                    🎙️ Start Recording
                  </button>
                ) : (
                  <button
                    className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 transition"
                    onClick={stopRecording}
                  >
                    ⏹️ Stop Recording
                  </button>
                )}
                {recordedBlob && (
                  <button
                    className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
                    onClick={analyzeRecordedAudio}
                  >
                    Analyze Recorded Audio
                  </button>
                )}
              </div>
            </motion.div>
          </TabPanel>
  
          {/* Model Integration */}
          <TabPanel>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h2 className="text-xl font-semibold">Model Integration</h2>
              <p>Upload a custom model to use in this app.</p>
              <input type="file" className="my-2" />
              <button className="bg-green-500 text-white px-4 py-2 rounded">Upload Model</button>
              {modelStatus === 'uploaded' && (
                <div className="text-green-700 mt-2">Model successfully integrated! Please refresh.</div>
              )}
            </motion.div>
          </TabPanel>
  
          {/* API Access */}
          <TabPanel>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h2 className="text-xl font-semibold">API Access</h2>
              <p>Use these endpoints in your app:</p>
              <ul className="list-disc ml-6 text-sm text-black-700">
                <li><code>GET /</code> - API info</li>
                <li><code>GET /formats</code> - Supported formats</li>
                <li><code>GET /model-info</code> - Model metadata</li>
                <li><code>POST /analyze-audio</code> - Upload audio</li>
              </ul>
              <h3 className="mt-4 font-semibold">Example:</h3>
              <pre className="bg-black-100 p-4 rounded text-xs overflow-x-auto">
  {`import requests
  
  files = {"audio_file": ("audio.wav", open("path/to/audio.wav", "rb"))}
  response = requests.post("http://localhost:8000/analyze-audio", files=files)
  result = response.json()
  
  print("Predominant emotion:", result['predominant_emotion'])
  for emotion, prob in result['emotions'].items():
      print(emotion, ":", prob)`}
              </pre>
            </motion.div>
          </TabPanel>
        </Tabs>
      </div> 
    </div> 
  );
}

export default App;  