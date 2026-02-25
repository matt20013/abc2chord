// Mock store for feedback payloads

const feedbackHistory = [];

export const submitChordFeedback = (payload) => {
  console.log("Submitting Chord Feedback:", payload);
  feedbackHistory.push(payload);
  // In a real app, this would send a POST request to an API
  return Promise.resolve({ success: true, id: feedbackHistory.length });
};

export const getFeedbackHistory = () => {
  return feedbackHistory;
};
