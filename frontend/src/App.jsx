import React, { useState } from 'react';
import ScoreViewer from './components/ScoreViewer';
import ChordFeedbackUI from './components/ChordFeedbackUI';
import { submitChordFeedback } from './state/feedbackStore';

const sampleAbc = `X:1
T:Cooley's Reel
M:4/4
L:1/8
K:Edor
|:D2|"Em"EBBA B2 EB|"Em"B2 AB dBAG|"D"FDAD BDAD|"D"FDAD dAFD|
"Em"EBBA B2 EB|"Em"B2 AB defg|"D"afef dBAF|"Em"DEFD E2:|
|:gf|"Em"eB B2 efge|"Em"eB B2 gedB|"D"A2 FA DAFA|"D"A2 FA defg|
"Em"eB B2 efge|"Em"eB B2 defg|"D"afef dBAF|"Em"DEFD E2:|`;

const App = () => {
  const [feedbackState, setFeedbackState] = useState({
    visible: false,
    position: { x: 0, y: 0 },
    chord: '',
    measureNumber: -1
  });

  const handleChordClick = (event) => {
    // Determine position. Maybe offset slightly so it doesn't cover the chord immediately
    // or position relative to the click.
    const position = { x: event.x + 10, y: event.y + 10 };

    setFeedbackState({
      visible: true,
      position,
      chord: event.chord,
      measureNumber: event.measureNumber
    });
  };

  const handleClose = () => {
    setFeedbackState(prev => ({ ...prev, visible: false }));
  };

  const handleSubmit = (payload) => {
    submitChordFeedback(payload);
    console.log("Feedback submitted:", payload);
  };

  return (
    <div className="App" style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>ABC Tunebook Editor - Feedback Demo</h1>
      <p>Click on any chord symbol (e.g., "Em", "D") to provide feedback.</p>

      <div style={{ border: '1px solid #ccc', padding: '10px', marginTop: '20px' }}>
        <ScoreViewer
          abcString={sampleAbc}
          onChordClick={handleChordClick}
        />
      </div>

      <ChordFeedbackUI
        visible={feedbackState.visible}
        position={feedbackState.position}
        chord={feedbackState.chord}
        measureNumber={feedbackState.measureNumber}
        tuneTitle="Cooley's Reel"
        onClose={handleClose}
        onSubmit={handleSubmit}
      />
    </div>
  );
};

export default App;
