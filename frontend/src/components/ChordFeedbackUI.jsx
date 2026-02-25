import React, { useState, useEffect } from 'react';

const ChordFeedbackUI = ({
  visible,
  position,
  chord,
  measureNumber,
  tuneTitle,
  onClose,
  onSubmit
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedChord, setEditedChord] = useState('');

  // Reset state when chord or visibility changes
  useEffect(() => {
    if (visible) {
      setIsEditing(false);
      setEditedChord(chord || '');
    }
  }, [visible, chord]);

  if (!visible) return null;

  const handleAction = (type) => {
    const payload = {
      timestamp: new Date().toISOString(),
      tune_title: tuneTitle,
      measure_number: measureNumber,
      predicted_chord: chord,
      user_label: type === 'accept' ? 1 : 0,
      corrected_chord: type === 'edit' ? editedChord : null
    };

    if (onSubmit) {
      onSubmit(payload);
    }
    onClose();
  };

  const handleEditSave = () => {
    handleAction('edit');
  };

  const containerStyle = {
    position: 'absolute',
    left: position.x,
    top: position.y,
    backgroundColor: 'white',
    border: '1px solid #ccc',
    borderRadius: '8px',
    padding: '12px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
    zIndex: 1000,
    minWidth: '200px'
  };

  const titleStyle = {
    margin: '0 0 8px 0',
    fontSize: '14px',
    fontWeight: 'bold'
  };

  const buttonGroupStyle = {
    display: 'flex',
    gap: '8px',
    marginTop: '8px'
  };

  const buttonStyle = {
    padding: '4px 8px',
    cursor: 'pointer',
    borderRadius: '4px',
    border: '1px solid #ddd',
    backgroundColor: '#f9f9f9'
  };

  return (
    <div style={containerStyle} data-testid="chord-feedback-ui">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={titleStyle}>Chord Feedback</h4>
        <button
          onClick={onClose}
          style={{ border: 'none', background: 'none', cursor: 'pointer', fontSize: '16px' }}
          aria-label="Close"
        >
          &times;
        </button>
      </div>

      <div style={{ marginBottom: '8px', fontSize: '14px' }}>
        Model predicted: <strong>{chord}</strong>
      </div>

      {isEditing ? (
        <div style={{ display: 'flex', gap: '4px' }}>
          <input
            type="text"
            value={editedChord}
            onChange={(e) => setEditedChord(e.target.value)}
            style={{ padding: '4px', width: '80px' }}
            autoFocus
            aria-label="Corrected chord"
          />
          <button onClick={handleEditSave} style={buttonStyle}>Save</button>
          <button onClick={() => setIsEditing(false)} style={buttonStyle}>Cancel</button>
        </div>
      ) : (
        <div style={buttonGroupStyle}>
          <button onClick={() => handleAction('accept')} style={buttonStyle} title="Accept">
            👍
          </button>
          <button onClick={() => handleAction('reject')} style={buttonStyle} title="Reject">
            👎
          </button>
          <button onClick={() => setIsEditing(true)} style={buttonStyle} title="Edit">
            ✏️
          </button>
        </div>
      )}
    </div>
  );
};

export default ChordFeedbackUI;
