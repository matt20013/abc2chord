import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChordFeedbackUI from './ChordFeedbackUI';

describe('ChordFeedbackUI', () => {
  const mockOnClose = jest.fn();
  const mockOnSubmit = jest.fn();

  const defaultProps = {
    visible: true,
    position: { x: 100, y: 100 },
    chord: 'Am',
    measureNumber: 5,
    tuneTitle: 'Test Tune',
    onClose: mockOnClose,
    onSubmit: mockOnSubmit
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders correctly when visible', () => {
    render(<ChordFeedbackUI {...defaultProps} />);

    expect(screen.getByText('Model predicted:')).toBeInTheDocument();
    expect(screen.getByText('Am')).toBeInTheDocument();
    expect(screen.getByTitle('Accept')).toBeInTheDocument();
    expect(screen.getByTitle('Reject')).toBeInTheDocument();
    expect(screen.getByTitle('Edit')).toBeInTheDocument();
  });

  test('does not render when hidden', () => {
    render(<ChordFeedbackUI {...defaultProps} visible={false} />);
    expect(screen.queryByText('Model predicted:')).not.toBeInTheDocument();
  });

  test('calls onSubmit with Accept payload', () => {
    render(<ChordFeedbackUI {...defaultProps} />);

    fireEvent.click(screen.getByTitle('Accept'));

    expect(mockOnSubmit).toHaveBeenCalledWith(expect.objectContaining({
      tune_title: 'Test Tune',
      measure_number: 5,
      predicted_chord: 'Am',
      user_label: 1,
      corrected_chord: null
    }));
    expect(mockOnClose).toHaveBeenCalled();
  });

  test('calls onSubmit with Reject payload', () => {
    render(<ChordFeedbackUI {...defaultProps} />);

    fireEvent.click(screen.getByTitle('Reject'));

    expect(mockOnSubmit).toHaveBeenCalledWith(expect.objectContaining({
      user_label: 0,
      corrected_chord: null
    }));
  });

  test('handles Edit flow correctly', async () => {
    render(<ChordFeedbackUI {...defaultProps} />);

    // Click Edit button
    fireEvent.click(screen.getByTitle('Edit'));

    // Verify input appears
    const input = screen.getByRole('textbox', { name: /corrected chord/i });
    expect(input).toBeInTheDocument();
    expect(input).toHaveValue('Am');

    // Change value
    fireEvent.change(input, { target: { value: 'G7' } });

    // Click Save
    fireEvent.click(screen.getByText('Save'));

    expect(mockOnSubmit).toHaveBeenCalledWith(expect.objectContaining({
      user_label: 0,
      corrected_chord: 'G7'
    }));
  });
});
