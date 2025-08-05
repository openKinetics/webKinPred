import React from 'react';
import PropTypes from 'prop-types';
import { Card, Form } from 'react-bootstrap';

export default function PredictionTypeSelect({ value, onChange }) {
  return (
    <Card className="section-container section-how-to-use mb-4">
      <Card.Body>
        <h3>What would you like to predict?</h3>
        <Form.Group controlId="predictionType">
          <Form.Control
            as="select"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="custom-select"
            required
          >
            <option value="">Select</option>
            <option value="kcat">Turnover Number (kcat)</option>
            <option value="Km">Michaelis Constant (KM)</option>
            <option value="both">Both</option>
          </Form.Control>
        </Form.Group>
      </Card.Body>
    </Card>
  );
}

PredictionTypeSelect.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};
