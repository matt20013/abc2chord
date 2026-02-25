import React, { useEffect, useRef } from 'react';
import abcjs from 'abcjs';

const ScoreViewer = ({ abcString, onChordClick }) => {
  const scoreRef = useRef(null);
  const onChordClickRef = useRef(onChordClick);

  // Keep the callback ref up to date
  useEffect(() => {
    onChordClickRef.current = onChordClick;
  }, [onChordClick]);

  useEffect(() => {
    if (scoreRef.current && abcString) {
      const renderOptions = {
        add_classes: true,
        responsive: "resize",
        clickListener: (abcElem, tuneNumber, classes, analysis, drag, mouseEvent) => {
          console.log("Clicked element:", abcElem, classes); // Debugging

          // Check if the clicked element is a chord.
          // abcjs adds "abcjs-chord" class to chord symbols.
          // abcElem.el_type might be "chord" if strictly parsed as such.
          // Also check if it's a note with a chord attached (abcjs often groups them).
          const isChordElement = (classes && classes.includes("abcjs-chord")) || (abcElem && abcElem.el_type === "chord");
          const hasChordAttached = abcElem && abcElem.chord && abcElem.chord.length > 0;

          if (isChordElement || hasChordAttached) {
            // Extract chord name from abcElem.name or abcElem.abcelem.name or abcElem.chord[0].name
            let chordName = abcElem.name;
            if (!chordName && abcElem.abcelem && abcElem.abcelem.name) {
                chordName = abcElem.abcelem.name;
            }
            if (!chordName && hasChordAttached) {
                chordName = abcElem.chord[0].name;
            }

            // Extract measure number from classes (e.g., "abcjs-m4")
            let measureNumber = -1;
            if (classes) {
              const match = classes.match(/abcjs-m(\d+)/);
              if (match) {
                measureNumber = parseInt(match[1], 10);
              }
            }

            if (chordName) {
              // Trigger the callback using the ref
              if (onChordClickRef.current) {
                onChordClickRef.current({
                  x: mouseEvent.pageX,
                  y: mouseEvent.pageY,
                  chord: chordName,
                  measureNumber: measureNumber
                });
              }
            }
          }
        }
      };

      abcjs.renderAbc(scoreRef.current, abcString, renderOptions);
    }
  }, [abcString]); // Only re-render if abcString changes

  return (
    <div className="score-viewer">
      <div ref={scoreRef} style={{ width: '100%', minHeight: '200px', background: '#fff', border: '1px solid #eee' }}></div>
    </div>
  );
};

export default ScoreViewer;
