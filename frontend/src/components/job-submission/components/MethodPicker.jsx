import React from 'react';
import PropTypes from 'prop-types';
import { Card, Row, Col, Form, Button } from 'react-bootstrap';
import MethodDetails from './MethodDetails';
import ExperimentalSwitch from './ExperimentalSwitch';

export default function MethodPicker({
  predictionType,
  allowedKcatMethods,
  kcatMethod,
  setKcatMethod,
  kmMethod,
  setKmMethod,
  csvFormatInfo,
  useExperimental,
  setUseExperimental,
  onSubmit,
  isSubmitting,
}) {
  const showKcat = predictionType === 'kcat' || predictionType === 'both';
  const showKm = predictionType === 'Km' || predictionType === 'both';

  return (
    <Card className="section-container section-method-selection mb-4">
      <Card.Body>
        <h3>Select Prediction Method</h3>
        <Row>
          {showKcat && (
            <Col md={6}>
              <Form.Group controlId="kcatMethod">
                <Form.Control
                  as="select"
                  disabled={!csvFormatInfo?.csv_type}
                  value={kcatMethod}
                  onChange={(e) => setKcatMethod(e.target.value)}
                  className="custom-select"
                  required
                >
                  <option value="">Select kcat method</option>
                  {allowedKcatMethods.map((method) => (
                    <option key={method} value={method}>
                      {method === 'EITLEM' ? 'EITLEM-Kinetics' : method}
                    </option>
                  ))}
                </Form.Control>
                {kcatMethod && <MethodDetails methodKey={kcatMethod} citationOnly />}
              </Form.Group>
            </Col>
          )}

          {showKm && (
            <Col md={6}>
              <Form.Group controlId="kmMethod">
                <Form.Control
                  as="select"
                  value={kmMethod}
                  disabled={!csvFormatInfo?.csv_type}
                  onChange={(e) => setKmMethod(e.target.value)}
                  className="custom-select"
                  required
                >
                  <option value="">Select KM method</option>
                  <option value="EITLEM">EITLEM-Kinetics</option>
                  <option value="UniKP">UniKP</option>
                </Form.Control>
                {kmMethod && <MethodDetails methodKey={kmMethod} citationOnly />}
              </Form.Group>
            </Col>
          )}
        </Row>
      </Card.Body>

      {(kcatMethod || kmMethod) && (
        <div className="mt-4 d-flex justify-content-end align-items-center px-3 pb-3">
          <ExperimentalSwitch checked={useExperimental} onChange={setUseExperimental} />
          <Button
            className="kave-btn"
            onClick={onSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Submitting…' : 'Submit Job'}
          </Button>
        </div>
      )}
    </Card>
  );
}

MethodPicker.propTypes = {
  predictionType: PropTypes.string.isRequired,
  allowedKcatMethods: PropTypes.arrayOf(PropTypes.string).isRequired,
  kcatMethod: PropTypes.string.isRequired,
  setKcatMethod: PropTypes.func.isRequired,
  kmMethod: PropTypes.string.isRequired,
  setKmMethod: PropTypes.func.isRequired,
  csvFormatInfo: PropTypes.object,
  useExperimental: PropTypes.bool.isRequired,
  setUseExperimental: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  isSubmitting: PropTypes.bool.isRequired,
};
