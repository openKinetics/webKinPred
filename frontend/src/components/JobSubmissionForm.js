// src/components/JobSubmissionForm.js
import React, { useState, useEffect } from 'react';
import { Form, Container, Row, Col, Card, Alert, Modal, Button } from 'react-bootstrap';
import axios from 'axios'; // Import axios for sending the file
import SequenceSimilaritySummary from './SequenceSimilaritySummary';


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
  const kcatMethodsByCsvType = {
    single: ['DLKcat', 'EITLEM', 'UniKP'],
    multi: ['TurNup']
  };
  

  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL;

  const methodDetails = {
    TurNup: {
      description: 'Predicts kcat for each reaction given protein sequence + list of substrates + list of products.',
      authors: 'Alexander Kroll, Yvan Rousset, Xiao-Pan Hu, Nina A. Liebrand & Martin J. Lercher',
      publicationTitle: 'Turnover number predictions for kinetically uncharacterized enzymes using machine and deep learning',
      citationUrl: 'https://www.nature.com/articles/s41467-023-39840-4',
      moreInfo: 'Recommended to use for natural reactions of wild-type enzymes.'
    },
    DLKcat: {
      description: 'Predicts kcat for a reaction given protein sequence + substrate.',
      authors: 'Feiran Li, Le Yuan, Hongzhong Lu, Gang Li, Yu Chen, Martin K. M. Engqvist, Eduard J. Kerkhoven & Jens Nielsen',
      publicationTitle: 'Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction',
      citationUrl: 'https://www.nature.com/articles/s41929-022-00798-z',
      moreInfo: ''
    },
    EITLEM: {
      description: 'Predicts kcat or KM for a reaction given protein sequence + substrate.',
      authors: 'Xiaowei Shen, Ziheng Cui, Jianyu Long, Shiding Zhang, Biqiang Chen, Tianwei Tan',
      publicationTitle: 'EITLEM-Kinetics: A deep-learning framework for kinetic parameter prediction of mutant enzymes',
      citationUrl: 'https://www.sciencedirect.com/science/article/pii/S2667109324002665',
      moreInfo: 'Recommended to use for reactions that include mutants'
    },
    UniKP: {
      description: 'Predicts kcat or KM for a reaction given protein sequence + substrate.',
      authors: 'Han Yu, Huaxiang Deng, Jiahui He, Jay D. Keasling & Xiaozhou Luo',
      publicationTitle: 'UniKP: a unified framework for the prediction of enzyme kinetic parameters',
      citationUrl: 'https://www.nature.com/articles/s41467-023-44113-1'
    }
  };
  
  
  const incompatibleMethods = [
    {
      kcatMethod: 'TurNup',
      kmMethod: 'EITLEM',
      message: 'TurNup is not compatible with EITLEM-Kinetics when predicting both kcat and KM. Please select compatible methods. TurNup expects a list of substrates and products, while EITLEM-Kinetics expects a single substrate. In FAQ, we explain how to use single-substrate data.' 
    },
    {
      kcatMethod: 'TurNup',
      kmMethod: 'UniKP',
      message: 'TurNup is not compatible with EITLEM-Kinetics when predicting both kcat and KM. Please select compatible methods. TurNup expects a list of substrates and products, while UniKP expects a single substrate. In FAQ, we explain how to use single-substrate data.'
    },
    
  ];
  function MethodDetails({ methodKey, citationOnly = false }) {
    const method = methodDetails[methodKey];
    if (!method) return null;
  
    return (
      <Alert variant="info" className="mt-3">
        {!citationOnly && <p>{method.description}</p>}
        {method.citation && method.citationUrl && (
          <p>
            <strong>Publication: </strong>
            <a href={method.citationUrl} target="_blank" rel="noopener noreferrer">
              {method.citation}
            </a>
          </p>
        )}
        {!citationOnly && method.moreInfo && (
          <p><strong>Note: </strong>{method.moreInfo}</p>
        )}
      </Alert>
    );
  }
  
  const checkCompatibility = (kcatMethod, kmMethod) => {
    if (predictionType !== 'both') {
      // Methods are compatible by default when not predicting both
      setIncompatibilityMessage('');
      return true;
    }
    if (kcatMethod && kmMethod) {
      const incompatibility = incompatibleMethods.find(
        combo =>
          combo.kcatMethod === kcatMethod &&
          combo.kmMethod === kmMethod
      );
      if (incompatibility) {
        setIncompatibilityMessage(incompatibility.message);
        return false;
      }
    }
    // Clear the message if compatible or methods are not both selected yet
    setIncompatibilityMessage('');
    return true;
  };
  useEffect(() => {
    if (predictionType === 'both' && kcatMethod && kmMethod) {
      checkCompatibility(kcatMethod, kmMethod);
    } else {
      setIncompatibilityMessage('');
    }
  }, [predictionType, kcatMethod, kmMethod]);
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

    axios
      .post(`${apiBaseUrl}/api/submit-job/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      .then((response) => {
        console.log('Job Submitted:', response.data);
        setSubmissionResult(response.data); // Set the submission result
        setShowModal(true); // Show the modal on successful submission
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

              <h5>🧬 CSV Input Format Requirements</h5>
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

              <h6>📥 Example Templates:</h6>
              <ul>
                <li><a href="/templates/single_substrate_template.csv" download>Download single-substrate template</a></li>
                <li><a href="/templates/multi_substrate_template.csv" download>Download multi-substrate template</a></li>
              </ul>
              <h4 className="mt-5 mb-3">🧠 Available Prediction Methods</h4>
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
                          <p
                            className="mb-0"
                            style={{
                              wordBreak: 'break-word',
                              overflowWrap: 'anywhere',
                            }}
                          >
                            <strong>Publication:</strong>{' '}
                            <a
                              href={method.citationUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{
                                textDecoration: 'underline',
                                color: '#a0d2eb',
                                fontSize: '0.95rem',
                                display: 'inline',
                                whiteSpace: 'normal',
                                wordBreak: 'break-word',
                              }}
                            >
                              {method.publicationTitle}
                            </a>
                          </p>
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
                    <Button
                      variant="outline-secondary"
                      onClick={() => setShowPreprocessPrompt(true)}
                    >
                      Validate Inputs (Optional)
                    </Button>
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
              const { invalid_substrates, invalid_proteins } = validation;
              const formData = new FormData();
              formData.append('file', csvFile);
              const similarityResponse = await axios.post(`${apiBaseUrl}/api/sequence-similarity-summary/`, formData);
              setSimilarityData(similarityResponse.data);
              setSubmissionResult({ invalid_substrates, invalid_proteins });
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
          <div>Validating Inputs and Running mmseqs2...</div>
        </div>
      )}
    </Container>
  );
}

export default JobSubmissionForm;
