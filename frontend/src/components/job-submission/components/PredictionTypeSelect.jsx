// PredictionTypeSelect.jsx
import React from 'react';
import PropTypes from 'prop-types';
import { Card, Form } from 'react-bootstrap';
import '../../../styles/components/PredictionTypeSelect.css'; // 1. Import the new CSS

export default function PredictionTypeSelect({ value, onChange }) {
  return (
    <Card className="section-container section-how-to-use mb-4">
      <Card.Header as="h3" className="text-center">
        What would you like to predict?
      </Card.Header>
      <Card.Body className="d-flex justify-content-center">
        {/* 2. Add the styled wrapper with a conditional class */}
        <div className={`kave-select-wrapper ${value ? 'is-selected' : ''}`}>
          <Form.Group controlId="predictionType" className="h-100">
            {/* 3. Use the new class on the select element */}
            <Form.Control
              as="select"
              value={value}
              onChange={(e) => onChange(e.target.value)}
              className="kave-select h-100"
              required
            >
              <option value="">Select a prediction type...</option>
              <option value="kcat">Turnover Number (kcat)</option>
              <option value="Km">Michaelis Constant (KM)</option>
              <option value="both">Both kcat and KM</option>
            </Form.Control>
          </Form.Group>
        </div>
      </Card.Body>
    </Card>
  );
}

PredictionTypeSelect.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};
