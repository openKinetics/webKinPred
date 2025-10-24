import React from 'react';
import PropTypes from 'prop-types';
import { Card, Row, Col, Form, Button } from 'react-bootstrap';
import MethodDetails from './MethodDetails';
import ExperimentalSwitch from './ExperimentalSwitch';
import '../../../styles/components/PredictionTypeSelect.css'; // Import the same CSS

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
      <Card.Header as="h3" className="text-center">
        Select Prediction Method(s)
      </Card.Header>
      <Card.Body>
        <Row>
          {showKcat && (
            <Col md={showKm ? 6 : 12} className="mb-3 mb-md-0">
              <div className={`kave-select-wrapper ${kcatMethod ? 'is-selected' : ''}`}>
                <Form.Group controlId="kcatMethod" className="h-100">
                  <Form.Control
                    as="select"
                    disabled={!csvFormatInfo?.csv_type}
                    value={kcatMethod}
                    onChange={(e) => setKcatMethod(e.target.value)}
                    className="kave-select h-100"
                    required
                  >
                    <option value="">Select kcat method...</option>
                    {allowedKcatMethods.map((method) => (
                      <option key={method} value={method}>
                        {method === 'EITLEM' ? 'EITLEM-Kinetics' : method}
                      </option>
                    ))}
                  </Form.Control>
                </Form.Group>
              </div>
              {kcatMethod && <MethodDetails methodKey={kcatMethod} citationOnly />}
            </Col>
          )}

          {showKm && (
            <Col md={showKcat ? 6 : 12}>
              <div className={`kave-select-wrapper ${kmMethod ? 'is-selected' : ''}`}>
                <Form.Group controlId="kmMethod" className="h-100">
                  <Form.Control
                    as="select"
                    value={kmMethod}
                    disabled={!csvFormatInfo?.csv_type}
                    onChange={(e) => setKmMethod(e.target.value)}
                    className="kave-select h-100"
                    required
                  >
                    <option value="">Select KM method...</option>
                    <option value="EITLEM">EITLEM-Kinetics</option>
                    <option value="UniKP">UniKP</option>
                    <option value="KinForm-H">KinForm-H</option>
                  </Form.Control>
                </Form.Group>
              </div>
              {kmMethod && <MethodDetails methodKey={kmMethod} citationOnly />}
            </Col>
          )}
        </Row>
      </Card.Body>

      {(kcatMethod || kmMethod) && (
        <Card.Footer className="d-flex justify-content-end align-items-center">
          <ExperimentalSwitch checked={useExperimental} onChange={setUseExperimental} />
          <Button
            className="kave-btn ms-3"
            onClick={onSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Submitting…' : 'Submit Job'}
          </Button>
        </Card.Footer>
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