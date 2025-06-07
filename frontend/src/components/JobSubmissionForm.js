// src/components/JobSubmissionForm.js
import React, { useState, useEffect } from 'react';
import { Form, Container, Row, Col, Card, Alert, Modal, Button } from 'react-bootstrap';
import axios from 'axios'; // Import axios for sending the file
import SequenceSimilaritySummary from './SequenceSimilaritySummary';
import { Table } from 'react-bootstrap';

function JobSubmissionForm() {
  const [isValidating, setIsValidating] = useState(false);
  const [predictionType, setPredictionType] = useState('');
  const [kcatMethod, setKcatMethod] = useState('');
  const [kmMethod, setKmMethod] = useState('');
  const [csvFile, setCsvFile] = useState(null);
  const [fileName, setFileName] = useState('No file chosen'); 
  const [showKcatMethodDetails, setShowKcatMethodDetails] = useState(false);
  const [showKmMethodDetails, setShowKmMethodDetails] = useState(false);
  const [submissionResult, setSubmissionResult] = useState(null);
  const [showModal, setShowModal] = useState(false); 
  const [incompatibilityMessage, setIncompatibilityMessage] = useState('');
  const [showPreprocessPrompt, setShowPreprocessPrompt] = useState(false);
  const [similarityData, setSimilarityData] = useState(null);
  const [csvFormatInfo, setCsvFormatInfo] = useState(null);
  const [csvFormatError, setCsvFormatError] = useState('');
  const [csvFormatValid, setCsvFormatValid] = useState(false);
  const [showValidationResults, setShowValidationResults] = useState(false);
  const [handleLongSeqs, setHandleLongSeqs] = useState('truncate');  
  const kcatMethodsByCsvType = {
    single: ['DLKcat', 'EITLEM', 'UniKP'],
    multi: ['TurNup']
  };
  

  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL;

  const methodDetails = {
    UniKP: {
      description: 'Predicts kcat or KM for a reaction given protein sequence + substrate.',
      authors: 'Han Yu, Huaxiang Deng, Jiahui He, Jay D. Keasling & Xiaozhou Luo',
      publicationTitle: 'UniKP: a unified framework for the prediction of enzyme kinetic parameters',
      citationUrl: 'https://www.nature.com/articles/s41467-023-44113-1',
      repoUrl: 'https://github.com/Luo-SynBioLab/UniKP'
    },
    DLKcat: {
      description: 'Predicts kcat for a reaction given protein sequence + substrate.',
      authors: 'Feiran Li, Le Yuan, Hongzhong Lu, Gang Li, Yu Chen, Martin K. M. Engqvist, Eduard J. Kerkhoven & Jens Nielsen',
      publicationTitle: 'Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction',
      citationUrl: 'https://www.nature.com/articles/s41929-022-00798-z',
      repoUrl: 'https://github.com/SysBioChalmers/DLKcat',
      moreInfo: ''
    },
    TurNup: {
      description: 'Predicts kcat for each reaction given protein sequence + list of substrates + list of products.',
      authors: 'Alexander Kroll, Yvan Rousset, Xiao-Pan Hu, Nina A. Liebrand & Martin J. Lercher',
      publicationTitle: 'Turnover number predictions for kinetically uncharacterized enzymes using machine and deep learning',
      citationUrl: 'https://www.nature.com/articles/s41467-023-39840-4',
      repoUrl: 'https://github.com/AlexanderKroll/Kcat_prediction',
      moreInfo: 'Recommended to use for natural reactions of wild-type enzymes.'
    },
    EITLEM: {
      description: 'Predicts kcat or KM for a reaction given protein sequence + substrate.',
      authors: 'Xiaowei Shen, Ziheng Cui, Jianyu Long, Shiding Zhang, Biqiang Chen, Tianwei Tan',
      publicationTitle: 'EITLEM-Kinetics: A deep-learning framework for kinetic parameter prediction of mutant enzymes',
      citationUrl: 'https://www.sciencedirect.com/science/article/pii/S2667109324002665',
      repoUrl: 'https://github.com/XvesS/EITLEM-Kinetics',
      moreInfo: 'Recommended to use for mutants'
    },

  };
  
  function MethodDetails({ methodKey, citationOnly = false }) {
    const method = methodDetails[methodKey];
    if (!method) return null;
  
    return (
      <Alert variant="info" className="mt-3">
        {!citationOnly && <p>{method.description}</p>}
        {method.publicationTitle && method.citationUrl && (
          <p>
            <strong>Publication: </strong>
            <a href={method.citationUrl} target="_blank" rel="noopener noreferrer">
              {method.publicationTitle}
            </a>
          </p>
        )}
        {!citationOnly && method.moreInfo && (
          <p><strong>Note: </strong>{method.moreInfo}</p>
        )}
      </Alert>
    );
  }
  useEffect(() => {
    if (predictionType) {
      // Reset method selections and detail views if predictionType is pre-set on mount
      setKcatMethod('');
      setKmMethod('');
      setShowKcatMethodDetails(false);
      setShowKmMethodDetails(false);
      setIncompatibilityMessage('');
    }
  }, []); // ← run once on mount
  

  const validateCsv = async () => {
    const formData = new FormData();
    formData.append('file', csvFile);
    formData.append('predictionType', predictionType);
    formData.append('kcatMethod', kcatMethod);
    formData.append('kmMethod', kmMethod);
  
    try {
      const response = await axios.post(`${apiBaseUrl}/api/validate-input/`, formData);
      return response.data;
    } catch (err) {
      console.error('Validation error:', err);
      throw err;
    }
  };
  const detectCsvFormat = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
  
    try {
      const response = await axios.post(`${apiBaseUrl}/api/detect-csv-format/`, formData);
      const data = response.data;
  
      if (data.status === 'valid') {
        setCsvFormatInfo(data);          // includes csv_type
        setCsvFormatValid(true);
        setCsvFormatError('');
      } else {
        setCsvFormatInfo(null);
        setCsvFormatValid(false);
        if (data.errors && Array.isArray(data.errors)) {
          setCsvFormatError(data.errors.join('; ')); // Join errors into a readable string
        } else {
          setCsvFormatError('Invalid CSV format.');
        }
      }
    } catch (error) {
      console.error('CSV format detection error:', error);
      setCsvFormatInfo(null);
      setCsvFormatValid(false);
      setCsvFormatError(
        error.response?.data?.error || 'Error detecting CSV format.'
      );
    }
  };

  const submitJob = () => {

    const formData = new FormData();
    formData.append('predictionType', predictionType);
    formData.append('kcatMethod', kcatMethod);
    formData.append('kmMethod', kmMethod);
    formData.append('file', csvFile);
    formData.append('handleLongSequences', handleLongSeqs);

    axios
      .post(`${apiBaseUrl}/api/submit-job/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      .then((response) => {
        setSubmissionResult((prev) => ({
          ...prev,
          message: response.data.message,
          job_id: response.data.job_id
        }));
        setShowModal(true);
      })      
      .catch((error) => {
        console.error('There was an error submitting the job:', error);
        if (error.response && error.response.data && error.response.data.error) {
          alert('Failed to submit job \n' + error.response.data.error);
        } else {
          alert('Failed to submit job');
        }
      });
  };
  const allowedKcatMethods = csvFormatInfo?.csv_type
  ? kcatMethodsByCsvType[csvFormatInfo.csv_type] || []
  : [];

  return (
    <Container className="mt-5 pb-5">
      <Row className="justify-content-center">
        <Col md={10}>
        <Card className="section-container section-how-to-use mb-4">
            <Card.Body>
              <h3>How to Use This Tool</h3>
              <p>
                This tool predicts kinetic parameters (kcat and/or KM) for enzyme-catalyzed reactions using various ML models.
              </p>
              <p><strong>Steps:</strong></p>
              <ol>
                <li>Select what you want to predict (kcat, KM, or both).</li>
                <li>Upload your reaction data as a CSV file.</li>
                <li>Choose prediction method(s) — optionally after preprocessing validation.</li>
              </ol>

              <h5>CSV Input Format Requirements</h5>
              <ul>
                <li>
                  <strong>'Protein Sequence'</strong> column is required for all methods.
                </li>
                <li>
                  <strong>Single-substrate models (DLKcat, EITLEM, UniKP):</strong> one <code>SMILES</code> per row, in 'Substrate' column.
                </li>
                <li>
                  <strong>Multi-substrate model (TurNup):</strong> use a <code>Substrates</code> column (semicolon-separated SMILES) and a <code>Products</code> column.
                </li>
              </ul>

              <p className="ps-3" style={{ marginLeft: '1rem' }}>
                You can also use multi-substrate data for KM predictions. Each substrate in the <code>Substrates</code> column will receive its own KM prediction, returned as a semicolon-separated list.  
                <br />
                If using single-substrate data, one KM value will be predicted per row.
              </p>
              <h6>📥 Example Templates:</h6>
              <ul>
                <li><a href="/templates/single_substrate_template.csv" download>Download single-substrate template</a></li>
                <li><a href="/templates/multi_substrate_template.csv" download>Download multi-substrate template</a></li>
              </ul>
              <h4 className="mt-5 mb-3">📔 Available Prediction Methods</h4>
              <Row>
                {Object.entries(methodDetails).map(([key, method]) => (
                  <Col md={6} key={key} className="mb-4">
                    <Card className="h-100 shadow-sm">
                      <Card.Body>
                        <h5 className="mb-2">{key}</h5>
                        <p className="mb-2">{method.description}</p>

                        {method.moreInfo && (
                          <p className="mb-2" style={{ color: '#808080' }}>
                            <em>{method.moreInfo}</em>
                          </p>
                        )}

                        {method.authors && (
                          <p className="mb-2" style={{ fontSize: '0.9rem' }}>
                            <strong>Authors:</strong> {method.authors}
                          </p>
                        )}

                        {method.publicationTitle && method.citationUrl && (
                          <div>
                            <p className="mb-2" style={{ wordBreak: 'break-word' }}>
                              <strong>Publication:</strong>{' '}
                              <a
                                href={method.citationUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{
                                  textDecoration: 'underline',
                                  color: '#a0d2eb',
                                  fontSize: '0.95rem',
                                }}
                              >
                                {method.publicationTitle}
                              </a>
                            </p>

                            <hr style={{ borderTop: '1px solid #444', marginTop: '0.5rem', marginBottom: '0.5rem' }} />

                            {/* GitHub repo link with icon */}
                            {method.repoUrl && (
                              <p className="mb-0">
                                <a
                                  href={method.repoUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    textDecoration: 'none',
                                    color: '#8dcaff',
                                    fontSize: '0.9rem',
                                    fontWeight: 500
                                  }}
                                >
                                  {/* GitHub SVG icon */}
                                  <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="16"
                                    height="16"
                                    fill="currentColor"
                                    className="me-1"
                                    viewBox="0 0 16 16"
                                    style={{ marginRight: '6px' }}
                                  >
                                    <path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.52-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                                  </svg>
                                  GitHub Repository
                                </a>
                              </p>
                            )}
                          </div>
                        )}
                      </Card.Body>
                    </Card>
                  </Col>
                ))}
              </Row>

            </Card.Body>
          </Card>
          {/* Prediction Type Section */}
          <Card className="section-container section-prediction-type">
            <Card.Body>
              <h3>What would you like to predict?</h3>
              <Form.Group controlId="predictionType">
                <Form.Control
                  as="select"
                  value={predictionType}
                  onChange={(e) => {
                    setPredictionType(e.target.value);
                    setKcatMethod('');
                    setKmMethod('');
                    setShowKcatMethodDetails(false);
                    setShowKmMethodDetails(false);
                    setIncompatibilityMessage('');
                  }}
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
          {/* CSV Upload Section */}
          {predictionType && (
            <Card className="section-container section-reaction-info">
              <Card.Body>
                <h3>Upload Reaction Information</h3>
                <p>Please upload a CSV file with the columns mentioned above</p>
                <Form>
                  <Form.Group controlId="csvFile" className="mt-3">
                    <div className="file-upload">
                      <Form.Control
                        type="file"
                        accept=".csv"
                        onChange={(e) => {
                          const file = e.target.files[0];
                          setCsvFile(file);
                          setFileName(file?.name || 'No file chosen');
                          if (file) {
                            detectCsvFormat(file);
                            setKcatMethod('');
                            setKmMethod('');
                            setShowKcatMethodDetails(false);
                            setShowKmMethodDetails(false);

                            setSubmissionResult(null);
                            setSimilarityData(null);
                            setShowValidationResults(false);
                          }
                        }}
                        style={{ display: 'none' }} // Hide the default input
                        required
                      />
                      <label htmlFor="csvFile" className="custom-file-upload">
                        Choose File
                      </label>
                      <span id="file-selected">{fileName}</span>
                    </div>
                  </Form.Group>
                </Form>
                {csvFormatValid && csvFormatInfo?.csv_type && (
                <Alert variant="success" className="mt-3">
                  ✅ Detected a <strong>{csvFormatInfo.csv_type === 'multi' ? 'multi-substrate' : 'single-substrate'}</strong> CSV. You may now choose compatible methods.
                </Alert>
                )}
                {!csvFormatValid && csvFormatError && (
                  <Alert variant="danger" className="mt-3">
                    ❌ Invalid CSV: {csvFormatError}
                  </Alert>
                )}
                {csvFormatValid && (
                  <div className="mt-4 d-flex justify-content-end">
                    <button
                      type="button"
                      className="kave-btn kave-btn-secondary"
                      onClick={() => setShowPreprocessPrompt(true)}
                    >
                      <span className="kave-line"></span>
                      Validate Inputs (Optional)
                    </button>
                  </div>
                )}
              </Card.Body>
            </Card>
          )}
          {submissionResult && (
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
                        {submissionResult?.invalid_substrates?.length > 0 && (
                          <>
                            <h5>Invalid Substrates</h5>
                            <ul>
                              {submissionResult.invalid_substrates.map((entry, i) => (
                                <li key={i}>Row {entry.row}: {entry.reason}</li>
                              ))}
                            </ul>
                          </>
                        )}

                        {submissionResult?.invalid_proteins?.length > 0 && (
                          <>
                            <h5>Invalid Proteins</h5>
                            <ul>
                              {submissionResult.invalid_proteins.map((entry, i) => (
                                <li key={i}>Row {entry.row}: {entry.reason}</li>
                              ))}
                            </ul>
                          </>
                        )}
                      </>
                    )}
                    {submissionResult?.length_violations && (
                      <div className="mt-3">
                        <h5>⚠️ Protein Sequence Length Warnings</h5>
                        <Table striped bordered hover size="sm" className="bg-dark">
                          <thead>
                            <tr>
                              <th className="text-white">Model</th>
                              <th className="text-white">Limit</th>
                              <th className="text-white">Datapoints Containing Sequence Over Limit</th>
                            </tr>
                          </thead>
                          <tbody className="text-secondary">
                            {[
                              { key: 'EITLEM', label: 'EITLEM', limit: 1024 },
                              { key: 'TurNup', label: 'TurNup', limit: 1024 },
                              { key: 'UniKP', label: 'UniKP', limit: 1000 },
                              { key: 'DLKcat', label: 'DLKcat', limit: '∞' },
                              { key: 'Server', label: 'Server', limit: 1500 },
                            ].map(({ key, label, limit }) =>
                              submissionResult.length_violations[key] > 0 ? (
                                <tr key={key}>
                                  <td className="text-white"><strong>{label}</strong></td>
                                  <td className="text-white">{limit}</td>
                                  <td className="text-white">{submissionResult.length_violations[key]}</td>
                                </tr>
                              ) : null
                            )}
                          </tbody>
                        </Table>
                        <Form.Group className="mt-3">
                          <Form.Label className="text-white" style={{ fontSize: '1rem' }}>
                            For sequences longer than the model limit:
                          </Form.Label>
                          <div>
                            <Form.Check
                              inline
                              type="radio"
                              id="truncate-option"
                              label="Truncate (default)"
                              name="longSeqHandling"
                              value="truncate"
                              checked={handleLongSeqs === 'truncate'}
                              onChange={() => setHandleLongSeqs('truncate')}
                            />
                            <Form.Check
                              inline
                              type="radio"
                              id="skip-option"
                              label="Skip"
                              name="longSeqHandling"
                              value="skip"
                              checked={handleLongSeqs === 'skip'}
                              onChange={() => setHandleLongSeqs('skip')}
                            />
                          </div>
                          <Form.Text className="text-white" style={{ fontSize: '1rem' }}>
                            Truncation keeps the first and last halves (e.g. 1200 → 500+500 for a 1000-limit model).
                            Skipping means these entries will be excluded from predictions.
                          </Form.Text>
                        </Form.Group>
                      </div>
                    )}
                    {similarityData && (
                      <div className="mt-4">
                        <SequenceSimilaritySummary similarityData={similarityData} />
                      </div>
                    )}
                  </div>

                )}
              </Card.Body>
            </Card>
          )}
          {/* Method Selection Section */}
          {predictionType && csvFile && csvFormatValid && (
            <Card className="section-container section-method-selection">
              <Card.Body>
                <h3>Select Prediction Method</h3>
                <Row>
                  {(predictionType === 'kcat' || predictionType === 'both') && (
                    <Col md={6}>
                      <Form.Group controlId="kcatMethod">
                      <Form.Control
                        as="select"
                        disabled={!csvFormatInfo?.csv_type}
                        value={kcatMethod}
                        onChange={(e) => {
                          setKcatMethod(e.target.value);
                          setShowKcatMethodDetails(true);
                        }}
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

                        {showKcatMethodDetails && kcatMethod && (
                          <MethodDetails methodKey={kcatMethod} citationOnly />
                        )}
                      </Form.Group>
                    </Col>
                  )}
                  {(predictionType === 'Km' || predictionType === 'both') && (
                    <Col md={6}>
                      <Form.Group controlId="kmMethod">
                        <Form.Control
                          as="select"
                          value={kmMethod}
                          disabled={!csvFormatInfo?.csv_type}
                          onChange={(e) => {
                            setKmMethod(e.target.value);
                            setShowKmMethodDetails(true);
                          }}
                          className="custom-select"
                          required
                        >
                          <option value="">Select KM method</option>
                          <option value="EITLEM">EITLEM-Kinetics</option>
                          <option value="UniKP">UniKP</option>
                        </Form.Control>
                        {showKmMethodDetails && kmMethod && (
                          <MethodDetails methodKey={kmMethod} citationOnly />
                        )}
                      </Form.Group>
                    </Col>
                  )}
                </Row>
              </Card.Body>
              {csvFormatValid && (kcatMethod || kmMethod) && (
                <div className="mt-4 d-flex justify-content-end">
                  <button type="button" className="kave-btn" onClick={submitJob}>
                    <span className="kave-line"></span>
                    Submit Job
                  </button>
                </div>
              )}
            </Card>
          )}
          {incompatibilityMessage && (
            <Alert variant="danger" className="mt-3">
              {incompatibilityMessage}
            </Alert>
          )}
        </Col>
      </Row>
      {/* Modal: Ask user whether to preprocess before submitting */}
      <Modal
        show={showPreprocessPrompt}
        onHide={() => setShowPreprocessPrompt(false)}
        size="xl"  // make it larger
      >
        <Modal.Header closeButton>
          <Modal.Title>Preprocess Before Prediction?</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>
            Would you like to validate your input data before running predictions?
          </p>
          <p>
            This will identify invalid SMILES/InChIs and protein sequences perform sequence similarity checks (using mmseqs2) against the training datasets of the methods.
          </p>
          <p>
            <strong>Note:</strong> Even if you skip this step, invalid rows will be automatically excluded during prediction and will not produce results.
          </p>
          <p className="fw-bold">Recommended if you're unsure about input quality.</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowPreprocessPrompt(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={async () => {
            setShowPreprocessPrompt(false);
            setIsValidating(true);
            try {
              const validation = await validateCsv();
              const { invalid_substrates, invalid_proteins, length_violations } = validation;
              const formData = new FormData();
              formData.append('file', csvFile);
              const similarityResponse = await axios.post(`${apiBaseUrl}/api/sequence-similarity-summary/`, formData);
              setSimilarityData(similarityResponse.data);
              setSubmissionResult({ invalid_substrates, invalid_proteins, length_violations});
              setShowValidationResults(true);               
            } catch (err) {
              alert('Validation failed. Please try again.' + err);
              console.error('Validation error:', err);
            } finally {
              setIsValidating(false);
            }
          }}>
            Run Validation
          </Button>
        </Modal.Footer>

      </Modal>

      {/* Modal for Submission Result */}
      <Modal show={showModal} onHide={() => setShowModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Job Successfully Submitted</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <h5>{submissionResult?.message}</h5>
          <p>Job ID: {submissionResult?.job_id}</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModal(false)}>
            Close
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              setShowModal(false);
              window.location.href = `/track-job/${submissionResult.job_id}`;
            }}
          >
            Track Job
          </Button>
        </Modal.Footer>
      </Modal>
      {isValidating && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          zIndex: 9999,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          color: 'white',
          fontSize: '1.5rem',
          flexDirection: 'column',
        }}>
          <div className="spinner-border text-light mb-3" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <div>Validating Inputs and Running MMseqs2...</div>
        </div>
      )}
    </Container>
  );
}

export default JobSubmissionForm;
