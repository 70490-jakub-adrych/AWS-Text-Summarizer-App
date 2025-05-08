import React, { useState } from 'react';
import { API_ENDPOINT } from '../config';
import { jsPDF } from 'jspdf';
import './Summarize.css';

const Summarize = ({ idToken }) => {
  const [inputText, setInputText] = useState('');
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [debug, setDebug] = useState(null);
  const [progress, setProgress] = useState(0); // Add loading progress indicator
  const [endpointStatus, setEndpointStatus] = useState('unknown');

  const handleSummarize = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to summarize');
      return;
    }

    setIsLoading(true);
    setSummary('');
    setError(null);
    setDebug(null);
    setProgress(10); // Start progress indication

    // Check if text is too long and show warning
    if (inputText.length > 10000) {
      setError('Warning: Long texts may take more time to process or time out. Consider using a shorter text for better results.');
    }

    // Start a progress timer to give user feedback during long operations
    const progressTimer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressTimer);
          return 90;
        }
        return prev + 5;
      });
    }, 1000);

    try {
      console.log('Sending request to:', `${API_ENDPOINT}/summarize`);
      console.log('Input text length:', inputText.length);
      
      // Simple, clean payload
      const payload = { text: inputText.trim() };
      
      const response = await fetch(`${API_ENDPOINT}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken || ''}`,
        },
        body: JSON.stringify(payload)
      });

      // Get the response text
      const responseText = await response.text();
      console.log('Response text:', responseText);
      
      // Clear the progress timer
      clearInterval(progressTimer);
      setProgress(100);
      
      // Parse the response
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        throw new Error(`Failed to parse response as JSON: ${responseText}`);
      }
      
      // Save the debug info
      setDebug(JSON.stringify(data, null, 2));

      // Check for endpoint-related errors
      if (data.error && 
         (data.error.includes("Could not find endpoint") || 
          data.error.includes("Endpoint not available") || 
          data.error.includes("is not available"))) {
        setEndpointStatus('stopped');
        throw new Error("The summarization service is currently offline. Please try again later when the service is reactivated.");
      } else {
        setEndpointStatus('running');
      }
      
      // Handle success or error based on status code
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${data.error || response.statusText}`);
      }
      
      // Extract the summary - handle both direct and nested formats
      let summaryText = '';
      
      // Check if we have a nested body that's a string
      if (data.body && typeof data.body === 'string') {
        try {
          // Try to parse the nested body
          const bodyData = JSON.parse(data.body);
          if (bodyData.summary) {
            summaryText = bodyData.summary;
          }
        } catch (e) {
          console.error('Failed to parse body as JSON:', e);
        }
      } 
      // If we didn't get a summary from the nested body, check if it's directly in the data
      else if (data.summary) {
        summaryText = data.summary;
      }
      
      if (summaryText) {
        setSummary(summaryText);
        console.log('Setting summary:', summaryText);
      } else {
        throw new Error('Could not find summary in response');
      }
    } catch (err) {
      clearInterval(progressTimer);
      setProgress(0);
      setError(`Failed to summarize: ${err.message}`);
      console.error('Summarization error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const generatePDF = () => {
    const doc = new jsPDF();
    
    // Add title
    doc.setFontSize(16);
    doc.text('Text Summarization', 20, 20);
    
    // Add original text
    doc.setFontSize(12);
    doc.text('Original Text:', 20, 30);
    const wrappedOriginalText = doc.splitTextToSize(inputText, 170);
    doc.text(wrappedOriginalText, 20, 40);
    
    // Add summary
    const summaryYPosition = 40 + (wrappedOriginalText.length * 7);
    doc.text('Summary:', 20, summaryYPosition);
    const wrappedSummary = doc.splitTextToSize(summary, 170);
    doc.text(wrappedSummary, 20, summaryYPosition + 10);
    
    // Save PDF
    doc.save('text-summary.pdf');
  };

  return (
    <div className="summarize-container">
      <h2 className="summarize-title">Text Summarizer</h2>
      
      {endpointStatus === 'stopped' && (
        <div className="endpoint-status warning">
          <p>⚠️ The summarization service is currently offline to save costs.</p>
          <p>Contact the administrator to restart the service when needed.</p>
        </div>
      )}
      
      <div className="input-section">
        <textarea
          className="text-input"
          placeholder="Enter text to summarize..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          rows={10}
        />
        
        <button 
          className="summarize-btn"
          onClick={handleSummarize}
          disabled={isLoading}
        >
          {isLoading ? 'Summarizing...' : 'Summarize'}
        </button>
        
        {/* Add character count indicator */}
        <div className="char-counter">
          Character count: {inputText.length} 
          {inputText.length > 5000 ? 
            <span className="warning"> (Long texts may take longer to process)</span> : 
            ''}
        </div>
      </div>

      {isLoading && (
        <div className="progress-container">
          <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          <div className="progress-text">Processing... This may take up to 30 seconds for long texts</div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}
      
      {debug && (
        <div className="debug-info">
          <details>
            <summary>Debug Information</summary>
            <pre>{debug}</pre>
          </details>
        </div>
      )}

      {summary && (
        <div className="result-section">
          <h3>Summary</h3>
          <div className="summary-box">{summary}</div>
          <button 
            className="download-btn"
            onClick={generatePDF}
          >
            Download PDF
          </button>
        </div>
      )}
    </div>
  );
};

export default Summarize;
