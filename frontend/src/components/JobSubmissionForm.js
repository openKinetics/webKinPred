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
  const [showValidationModal, setShowValidationModal] = useState(false);
  const [showPreprocessPrompt, setShowPreprocessPrompt] = useState(false);
  const [showValidationPassedModal, setShowValidationPassedModal] = useState(false);
  const [similarityData, setSimilarityData] = useState(null);

  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL;
  console.log('API Base URL:', apiBaseUrl);

  const methodDetails = {
    TurNup: {
      description: 'TurNup predicts kcat for each reaction given protein sequence + list of substrates + list of products.',
      citation: 'Turnover number predictions for kinetically uncharacterized enzymes using machine and deep learning',
      citationUrl: 'https://www.nature.com/articles/s41467-023-39840-4',
      moreInfo: 'Recommended to use for natural reactions of wild-type enzymes.'
    },
    DLKcat: {
      description: 'DLKcat predicts kcat for a reaction given protein sequence + substrate.',
      citation: 'Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction',
      citationUrl: 'https://www.nature.com/articles/s41929-022-00798-z',
      moreInfo: ''
    },
    EITLEM: {
      description: 'EITLEM-Kinetics predicts kcat or KM for a reaction given protein sequence + substrate.',
      citation: 'EITLEM-Kinetics: A deep-learning framework for kinetic parameter prediction of mutant enzymes',
      citationUrl: 'https://www.sciencedirect.com/science/article/pii/S2667109324002665',
      moreInfo: ''
    },
    // Add other methods as needed
  };
  
  const incompatibleMethods = [
    {
      kcatMethod: 'TurNup',
      kmMethod: 'EITLEM',
      message: 'TurNup is not compatible with EITLEM-Kinetics when predicting both kcat and KM. Please select compatible methods.'
    },
  ];
  function MethodDetails({ methodKey }) {
    const method = methodDetails[methodKey];
    if (!method) return null;

    return (
      <Alert variant="info" className="mt-3">
        <p>{method.description}</p>
        {method.citation && method.citationUrl && (
          <p>
            <strong>Citation: </strong>
            <a href={method.citationUrl} target="_blank" rel="noopener noreferrer">
              {method.citation}
            </a>
          </p>
        )}
        {method.moreInfo && (
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
  
  const handleSubmit = (e) => {
    e.preventDefault();
    setShowPreprocessPrompt(true);
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

  return (
    <Container className="mt-5 pb-5">
      <Row className="justify-content-center">
        <Col md={8}>
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

          {/* Method Selection Section */}
          {(predictionType === 'both' || predictionType === 'kcat' || predictionType === 'Km') && (
            <Card className="section-container section-method-selection">
              <Card.Body>
                <h3>Select Prediction Method</h3>
                <Row>
                  {(predictionType === 'kcat' || predictionType === 'both') && (
                    <Col md={6}>
                      <Form.Group controlId="kcatMethod">
                        <Form.Control
                          as="select"
                          value={kcatMethod}
                          onChange={(e) => {
                            setKcatMethod(e.target.value);
                            setShowKcatMethodDetails(true);
                          }}
                          className="custom-select"
                          required
                        >
                          <option value="">Select kcat method</option>
                          <option value="TurNup">TurNup</option>
                          <option value="DLKcat">DLKcat</option>
                          <option value="EITLEM">EITLEM-Kinetics</option>
                        </Form.Control>
                        {showKcatMethodDetails && kcatMethod && (
                        <MethodDetails methodKey={kcatMethod} />
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
                          onChange={(e) => {
                            setKmMethod(e.target.value);
                            setShowKmMethodDetails(true);
                          }}
                          className="custom-select"
                          required
                        >
                          <option value="">Select KM method</option>
                          <option value="EITLEM">EITLEM-Kinetics</option>
                        </Form.Control>
                        {showKmMethodDetails && kmMethod && (
                            <MethodDetails methodKey={kmMethod} />
                        )}
                      </Form.Group>
                    </Col>
                  )}
                </Row>
              </Card.Body>
            </Card>
          )}
          {/* CSV Upload Section */}
          {(kcatMethod || kmMethod) && !incompatibilityMessage && (
            <Card className="section-container section-reaction-info">
              <Card.Body>
                <h3>Upload Reaction Information</h3>
                <p>Please upload a CSV file with the following columns:</p>
                <ul>
                  <li>
                    <strong>Protein Sequence:</strong> The sequence of the protein that catalyzes the reaction (one per row).
                  </li>
                  <li>
                    <strong>Protein Accession Number:</strong> For methods that require protein structure if protein ID is available, structure retrieval is faster and
                    more effective. Leave an empty string if not available (one per row).
                  </li>
                  {kcatMethod === 'TurNup' && (
                    <>
                      <li>
                        <strong>Substrates:</strong> A list of chemical identifiers (InChIs or
                        SMILES) separated by a semicolon (`;`). For example:
                        `InChI1;InChI2;SMILES3;...` (n per row, where n is the number of substrates in the reaction)
                      </li>
                      <li>
                        <strong>Products:</strong> A list of chemical identifiers (InChIs or SMILES)
                        separated by a semicolon (`;`). For example: `InChI1;SMILES2;...` (n per row, where n is the number of products in the reaction)
                      </li>
                    </>
                  )}
                  {(kcatMethod === 'DLKcat' || kcatMethod === 'EITLEM') && (
                    <li>
                      <strong>Substrate:</strong> Either SMILES or InChI (one per row)
                    </li>
                  )}
                </ul>
                <Form onSubmit={handleSubmit}>
                  <Form.Group controlId="csvFile" className="mt-3">
                    <div className="file-upload">
                      <Form.Control
                        type="file"
                        accept=".csv"
                        onChange={(e) => {
                          setCsvFile(e.target.files[0]);
                          setFileName(e.target.files[0]?.name || 'No file chosen');
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
                  <div className="mt-4 d-flex justify-content-end">
                    <button type="submit" className="kave-btn">
                      <span className="kave-line"></span>
                      Submit Job
                    </button>
                  </div>
                </Form>
              </Card.Body>
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
          <Button variant="secondary" onClick={() => {
            setShowPreprocessPrompt(false);
            submitJob();  // skip validation
          }}>
            Skip Validation – Predict Now
          </Button>
          <Button variant="primary" onClick={async () => {
          setShowPreprocessPrompt(false);
          setIsValidating(true); // <-- SHOW OVERLAY
          try {
            const validation = await validateCsv();
            const { invalid_substrates, invalid_proteins } = validation;

            const formData = new FormData();
            formData.append('file', csvFile);
            const similarityResponse = await axios.post(`${apiBaseUrl}/api/sequence-similarity-summary/`, formData);
            setSimilarityData(similarityResponse.data);

            if (invalid_substrates.length > 0 || invalid_proteins.length > 0) {
              setSubmissionResult({ invalid_substrates, invalid_proteins });
              setShowValidationModal(true);
            } else {
              setShowValidationPassedModal(true);
            }
          } catch (err) {
            alert('Validation failed. Please try again.' + err);
            console.error('Validation error:', err);
          } finally {
            setIsValidating(false); 
          }
        }}>
          Preprocess and Validate
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
      {/* 🔍 Modal for Validation Issues */}
      <Modal show={showValidationModal} onHide={() => setShowValidationModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Validation Results</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {submissionResult?.invalid_substrates?.length > 0 && (
            <>
              <p><strong>Invalid Substrates:</strong></p>
              <ul>
                {submissionResult.invalid_substrates.map((entry, i) => (
                  <li key={i}>Row {entry.row}: {entry.reason}</li>
                ))}
              </ul>
            </>
          )}
          <p className="mt-3">These rows will be excluded from prediction. Do you want to continue?</p>
          {submissionResult?.invalid_proteins?.length > 0 && (
            <>
              <p><strong>Invalid Proteins:</strong></p>
              <ul>
                {submissionResult.invalid_proteins.map((entry, i) => (
                  <li key={i}>Row {entry.row}: {entry.reason}</li>
                ))}
              </ul>
            </>
          )}
          {similarityData && (
            <SequenceSimilaritySummary similarityData={similarityData} />
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowValidationModal(false)}>Cancel</Button>
          <Button variant="primary" onClick={() => {
            setShowValidationModal(false);
            submitJob();
          }}>
            Proceed Anyway
          </Button>
        </Modal.Footer>
      </Modal>
      {/* ✅ Modal: All Good */}
      <Modal show={showValidationPassedModal} onHide={() => setShowValidationPassedModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Validation Successful</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>✅ All entries are valid. No issues were found with your input data.</p>
          {similarityData && (
            <SequenceSimilaritySummary similarityData={similarityData} />
          )}
          <p>You may now proceed to run predictions.</p>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="primary" onClick={() => {
            setShowValidationPassedModal(false);
            submitJob();  // proceed to run prediction
          }}>
            Proceed to Prediction
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
