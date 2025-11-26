import React, { useCallback, useState, useEffect } from 'react';
import { Client } from "@gradio/client";

const SLMGenerator = () => {
  const [prompt, setPrompt] = useState(
    'INT. DARK ALLEY - NIGHT. MARTHA stands alone, checking her watch. She whispers, "It\'s late."'
  );
  const [script, setScript] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Model parameters
  const [maxTokens, setMaxTokens] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(20);

  // State for the Gradio client
  const [gradioClient, setGradioClient] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('Connecting to backend...');

  // 1. Connect to Gradio on Load
  useEffect(() => {
    const connect = async () => {
      try {
        // Connects to your local Python server
        const client = await Client.connect("https://spenrtgs-moviescripter01.hf.space");
        setGradioClient(client);
        setConnectionStatus('Connected to Model');
        console.log("Connected to Gradio backend successfully!");
      } catch (error) {
        console.error("Connection failed:", error);
        setConnectionStatus('Error: Ensure app.py is running');
      }
    };
    connect();
  }, []);

const handleGenerate = useCallback(async () => {
    if (isGenerating || !prompt.trim()) return;
    if (!gradioClient) {
      setScript("Error: Backend not connected. Is app.py running?");
      return;
    }

    setIsGenerating(true);
    setScript('Generating script...');

    try {
      const result = await gradioClient.predict("/predict", [
        prompt,       
        maxTokens,    
        temperature,  
        topK === null ? 0 : topK,          
      ]);

      // Gradio Client returns the result directly inside the data array
      const generatedText = result.data[0];
      setScript(generatedText);

    } catch (error) {
      console.error("Generation failed:", error);
      setScript('Error: Generation failed. Check console (F12) for details.');
    } finally {
      setIsGenerating(false);
    }
  }, [isGenerating, prompt, maxTokens, temperature, topK, gradioClient]);

  return (
    <div className="glass-container">
      <header className="heading">
        <h1 className="text-gradient" style={{ fontSize: '2.6rem', fontWeight: 800, margin: '0 0 8px' }}>
          SLM Dialogue Writer
        </h1>
        <p className="subtitle">Fine-tuned on cinematic subtitles. Optimized with LoRA adapters.</p>
        <div style={{ fontSize: '0.8rem', color: gradioClient ? '#4ade80' : '#f87171', marginTop: '10px' }}>
             ● {connectionStatus}
        </div>
      </header>

      <section className="section">
        <label className="label" htmlFor="prompt">
          Scene Prompt / Opening Dialogue
        </label>
        <textarea
          id="prompt"
          className="glass-field input-field"
          placeholder="Start the scene..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={5}
          disabled={isGenerating}
        />

        <div className="control-bar">
          <div className="control-group">
            <label className="subtitle">Max Tokens</label>
            <input
              type="number"
              min="50"
              max="500"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 50)}
              className="glass-field number-input"
            />
          </div>

          <div className="control-group">
            <label className="subtitle">Temperature</label>
            <input
              type="number"
              min="0.1"
              max="1.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.1)}
              className="glass-field number-input"
            />
          </div>

          <div className="control-group">
            <label className="subtitle">Top-K</label>
            <input
              type="number"
              min="1"
              max="50"
              value={topK === null ? '' : topK}
              onChange={(e) => setTopK(parseInt(e.target.value, 10) || null)}
              className="glass-field number-input"
            />
          </div>

          <button
            className="glass-button"
            onClick={handleGenerate}
            disabled={isGenerating || !gradioClient}
          >
            {isGenerating ? 'Writing...' : 'Generate Script'}
          </button>
        </div>
      </section>

      <section className="section">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h2 className="section-title">Generated Dialogue</h2>
        </div>
        <pre className="glass-field output-box">
          {script || 'Hit "Generate Script" to send the prompt to the backend model.'}
        </pre>
      </section>
    </div>
  );
};

export default SLMGenerator;