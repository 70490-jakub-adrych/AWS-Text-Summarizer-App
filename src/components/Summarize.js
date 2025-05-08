import React, { useState } from 'react';
import { API_ENDPOINT } from '../config';
import { jsPDF } from 'jspdf';
import './Summarize.css';

const Summarize = ({ idToken }) => {
  const [inputText, setInputText] = useState('');
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSummarize = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to summarize');
      return;
    }

    setIsLoading(true);
    setSummary('');
    setError(null);

    try {
      const response = await fetch(`${API_ENDPOINT}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken}`
        },
        body: JSON.stringify({ text: inputText })
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setSummary(data.summary);
    } catch (err) {
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
      </div>

      {error && <div className="error-message">{error}</div>}

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
