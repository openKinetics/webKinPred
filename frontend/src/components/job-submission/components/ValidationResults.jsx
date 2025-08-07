import React from 'react';
import PropTypes from 'prop-types';
import { Card, Button, Alert, Table, Form } from 'react-bootstrap';
import SequenceSimilaritySummary from '../../SequenceSimilaritySummary';
import InvalidItems from './InvalidItems';

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

  return (
    <Card className="section-container section-validation-results mt-4">
      <Card.Body>
        <div className="d-flex justify-content-between align-items-center">
          <h3 className="mb-0">Validation Results</h3>
          <Button
            variant="outline-secondary"
            onClick={() => setShowValidationResults(!showValidationResults)}
          >
            {showValidationResults ? 'Hide' : 'Show'}
          </Button>
        </div>

        {showValidationResults && (
          <div className="mt-4">
            {submissionResult?.invalid_substrates?.length === 0 &&
            submissionResult?.invalid_proteins?.length === 0 ? (
              <Alert variant="success">
                ✅ All entries are valid. No issues were found with your input data.
              </Alert>
            ) : (
              <>
                <Alert variant="warning">
                    ⛔ Some entries are invalid. Predictions will not be generated for them (outputs remain empty). You do not need to remove them from the CSV.
                </Alert>
                {submissionResult?.invalid_substrates?.length > 0 && (
                <InvalidItems
                    title="Invalid Substrates"
                    items={submissionResult.invalid_substrates}
                />
                )}
                {submissionResult?.invalid_proteins?.length > 0 && (
                <InvalidItems
                    title="Invalid Proteins"
                    items={submissionResult.invalid_proteins}
                />
                )}
              </>
            )}
            {hasAnyLengthIssues && (
                <div className="mt-3">
                    <hr style={{ borderTop: '5px solid rgb(229, 228, 243)' }} />
                    <h5 className="text-center mt-3 mb-3">⚠️ Protein Sequence Length Warnings</h5>
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
                                <tr>
                                    <td className="text-white">
                                        <strong>Overall Server Limit</strong>
                                    </td>
                                    <td className="text-white">1500</td>
                                    <td className="text-danger">{lengthViol.Server}</td>
                                </tr>
                            )}
                        </tfoot>
                    </Table>

                    <div className="mt-4 p-3 bg-light bg-opacity-10 rounded">
                        <p className="text-warning mb-3" style={{ fontSize: '1.05rem' }}>
                            <strong>How to handle violating sequences?</strong>
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
                                Truncation preserves the first and last portions of a sequence (e.g., for a 1000-limit, a 1200-length sequence becomes first 500+ last 500). This method is generally effective in maintaining model performance.
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
                                    Excludes any datapoint that contains a violating sequence.
                                </Form.Text>
                            </div>
                        </Form.Group>
                    </div>
                </div>
            )}
            {similarityData && (
              <>
                <hr style={{ borderTop: '5px solid rgb(229, 228, 243)' }} />
                <div className="mt-4">
                  <SequenceSimilaritySummary similarityData={similarityData} />
                </div>
              </>
            )}
          </div>
        )}
      </Card.Body>
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
