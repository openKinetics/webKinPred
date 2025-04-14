import React, { useState } from 'react';
import { Alert } from 'react-bootstrap';

const ExpandableErrorMessage = ({ errorMessage }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const truncatedMessage = errorMessage.length > 100 
    ? errorMessage.substring(0, 100) + '...' 
    : errorMessage;

  return (
    <div>
      <Alert 
        variant="warning" 
        className="mt-3 d-flex justify-content-between align-items-center"
        onClick={() => setIsExpanded(!isExpanded)}
        style={{ cursor: 'pointer' }}
      >
        <pre style={{
          margin: 0,
          whiteSpace: 'pre-wrap',
          flex: 1,
          fontSize: '1.2rem' // ← change this value as needed
        }}>
          {isExpanded ? errorMessage : truncatedMessage}
        </pre>
        {errorMessage.length > 100 && (
          <small className="text-muted ms-3">
            {isExpanded ? '▼ Collapse' : '► Expand'}
          </small>
        )}
      </Alert>
    </div>
  );
};

export default ExpandableErrorMessage;