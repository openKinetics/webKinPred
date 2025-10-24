import React from 'react';
import PropTypes from 'prop-types';
import { Card, Button, Alert, Table, Form, Tabs, Tab } from 'react-bootstrap';
import {
  ChevronUp,
  ChevronDown,
  CheckCircleFill,
  XCircleFill,
  Rulers,
  BarChartFill,
} from 'react-bootstrap-icons';
import SequenceSimilarityHistogram from '../../SequenceSimilarityHistogram';
import InvalidItems from './InvalidItems';
import '../../../styles/components/ValidationResults.css'; // The new CSS file for tabs

export default function ValidationResults({
  submissionResult,
  showValidationResults,
  setShowValidationResults,
  handleLongSeqs,
  setHandleLongSeqs,
  similarityData,
}) {
  const lengthViol = submissionResult?.length_violations || {};
  const hasAnyLengthIssues = Object.values(lengthViol).some((v) => v > 0);
  const hasInvalidItems =
    submissionResult?.invalid_substrates?.length > 0 || submissionResult?.invalid_proteins?.length > 0;

  return (
    <Card className="section-container section-validation-results mt-4">
      <Card.Header
        as="h3"
        className="text-center d-flex justify-content-center align-items-center position-relative"
        onClick={() => setShowValidationResults(!showValidationResults)}
        style={{ cursor: 'pointer' }}
      >
        Validation Summary
        <Button variant="outline-light" className="position-absolute end-0 me-3 border-0">
          {showValidationResults ? <ChevronUp /> : <ChevronDown />}
        </Button>
      </Card.Header>

      {showValidationResults && (
        <Card.Body>
          <Tabs defaultActiveKey={similarityData ? "similarity" : "validation"} id="validation-tabs" className="validation-tabs mb-4" justify>
            {/* Tab 1: Input Validation */}
            <Tab
              eventKey="validation"
              title={
                // MODIFIED: Added justify-content-center to center the icon and text
                <span className="d-flex align-items-center justify-content-center">
                  {hasInvalidItems ? (
                    <XCircleFill className="me-2 text-warning" />
                  ) : (
                    <CheckCircleFill className="me-2 text-success" />
                  )}
                  Input Validation
                </span>
              }
            >
              <div className="tab-content-wrapper">
                {hasInvalidItems ? (
                  <>
                    <Alert variant="warning">
                      ⛔ Some entries are invalid and will be skipped. You do not need to remove them from your file.
                    </Alert>
                    {submissionResult?.invalid_substrates?.length > 0 && (
                      <InvalidItems title="Invalid Substrates" items={submissionResult.invalid_substrates} />
                    )}
                    {submissionResult?.invalid_proteins?.length > 0 && (
                      <InvalidItems title="Invalid Proteins" items={submissionResult.invalid_proteins} />
                    )}
                  </>
                ) : (
                  <Alert variant="success">
                    ✅ All entries are valid. No issues were found with your input data.
                  </Alert>
                )}
              </div>
            </Tab>

            {/* Tab 2: Length Warnings (Conditional) */}
            {hasAnyLengthIssues && (
              <Tab
                eventKey="length"
                title={
                  // MODIFIED: Added justify-content-center
                  <span className="d-flex align-items-center justify-content-center">
                    <Rulers className="me-2 text-warning" />
                    Length Warnings
                  </span>
                }
              >
                <div className="tab-content-wrapper">
                  <h5 className="tab-section-header text-center mb-3">Protein Sequence Length Limits</h5>
                  <Table striped bordered hover size="sm" className="bg-dark">
                    <thead>
                      <tr>
                        <th className="text-white" colSpan="3" style={{ backgroundColor: '#4e4e4e' }}>
                          <strong>Model Limits</strong>
                        </th>
                      </tr>
                      <tr>
                        <th className="text-white">Category</th>
                        <th className="text-white">Limit</th>
                        <th className="text-white">Violations</th>
                      </tr>
                    </thead>
                    <tbody className="text-secondary">
                      {[
                        { key: 'KinForm-H', label: 'KinForm-H', limit: 1500 },
                        { key: 'KinForm-L', label: 'KinForm-L', limit: 1500 },
                        { key: 'EITLEM', label: 'EITLEM', limit: 1024 },
                        { key: 'TurNup', label: 'TurNup', limit: 1024 },
                        { key: 'UniKP', label: 'UniKP', limit: 1000 },
                        { key: 'DLKcat', label: 'DLKcat', limit: '∞' },
                      ].map(({ key, label, limit }) =>
                        lengthViol[key] > 0 ? (
                          <tr key={key}>
                            <td className="text-white">{label}</td>
                            <td className="text-white">{limit}</td>
                            <td className="text-danger">{lengthViol[key]}</td>
                          </tr>
                        ) : null
                      )}
                    </tbody>
                    <tfoot>
                      {lengthViol.Server > 0 && (
                        <tr style={{ borderTop: '2px solid rgb(78, 78, 78)' }}>
                          <td className="text-white">
                            <strong>Overall Server Limit</strong>
                          </td>
                          <td className="text-white">10,000</td>
                          <td className="text-danger">{lengthViol.Server}</td>
                        </tr>
                      )}
                    </tfoot>
                  </Table>

                  <div className="mt-4 p-3 bg-light bg-opacity-10 rounded">
                    <p className="text-warning mb-3" style={{ fontSize: '1.05rem' }}>
                      <strong>How to handle long sequences?</strong>
                    </p>
                    <Form.Group>
                      <Form.Check
                        type="radio"
                        id="truncate-option"
                        name="longSeqHandling"
                        value="truncate"
                        label="Truncate sequences (default)"
                        checked={handleLongSeqs === 'truncate'}
                        onChange={() => setHandleLongSeqs('truncate')}
                      />
                      <Form.Text className="text-white-50 ms-4">
                        Truncation preserves the first and last portions of a sequence (e.g., for a 1000-limit, a 1200-length sequence becomes first 500 + last 500).
                      </Form.Text>
                      <div className="mt-3">
                        <Form.Check
                          type="radio"
                          id="skip-option"
                          name="longSeqHandling"
                          value="skip"
                          label="Skip sequences"
                          checked={handleLongSeqs === 'skip'}
                          onChange={() => setHandleLongSeqs('skip')}
                        />
                        <Form.Text className="text-white-50 ms-4">
                          Excludes any datapoint that contains a sequence exceeding length limits.
                        </Form.Text>
                      </div>
                    </Form.Group>
                  </div>
                </div>
              </Tab>
            )}

            {/* Tab 3: Similarity Analysis (Conditional) */}
            {similarityData && (
              <Tab
                eventKey="similarity"
                title={
                  <span className="d-flex align-items-center justify-content-center">
                    <BarChartFill className="me-2" />
                    Similarity Analysis
                  </span>
                }
              >
                <div className="tab-content-wrapper">
                  <SequenceSimilarityHistogram similarityData={similarityData} />
                </div>
              </Tab>
            )}
          </Tabs>
        </Card.Body>
      )}
    </Card>
  );
}

ValidationResults.propTypes = {
  submissionResult: PropTypes.object,
  showValidationResults: PropTypes.bool.isRequired,
  setShowValidationResults: PropTypes.func.isRequired,
  handleLongSeqs: PropTypes.string.isRequired,
  setHandleLongSeqs: PropTypes.func.isRequired,
  similarityData: PropTypes.object,
};