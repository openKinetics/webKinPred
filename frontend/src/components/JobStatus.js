// JobStatus.js

import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Form, Container, Row, Col, Card, Alert, ProgressBar, Spinner } from 'react-bootstrap';
import axios from 'axios';
import moment from 'moment';
import ExpandableErrorMessage from './ExpandableErrorMessage';

function JobStatus() {
  const { jobId: routeJobId } = useParams();
  const [inputJobId, setInputJobId] = useState(routeJobId || '');
  const [jobId, setJobId] = useState(routeJobId || '');
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [timeElapsed, setTimeElapsed] = useState('');
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL ;

  useEffect(() => {
    let intervalId;

    if (jobId) {
      const fetchJobStatus = () => {
        axios.get(`${apiBaseUrl}/api/job-status/${jobId}/`)
          .then(response => {
            setJobStatus(response.data);
            setError(null);
          })
          .catch(error => {
            console.error('Error fetching job status:', error);
            setError('Failed to fetch job status');
            setJobStatus(null);
          });
      };
      fetchJobStatus();
      intervalId = setInterval(fetchJobStatus, 500); 
    }

    return () => {
      clearInterval(intervalId);
    };
  }, [jobId]);

  useEffect(() => {
    let timerId;
  
    if (jobStatus && jobStatus.submission_time) {
      if (jobStatus.status === 'Processing' || jobStatus.status === 'Completed' || jobStatus.status === 'Failed') {
        timerId = setInterval(() => {
          const submissionTime = moment(jobStatus.submission_time);
          const currentTime = moment();
          
          let endTime = currentTime;
          if (jobStatus.status === 'Completed' || jobStatus.status === 'Failed') {
            // Use completion time if the job has finished
            endTime = jobStatus.completion_time ? moment(jobStatus.completion_time) : currentTime;
          }
  
          const duration = moment.duration(endTime.diff(submissionTime));
          setTimeElapsed(formatDuration(duration));
        }, 5000);
      }
    }
  
    return () => {
      clearInterval(timerId);
    };
  }, [jobStatus]);  

  const handleCheckStatus = (e) => {
    e.preventDefault();
    setJobId(inputJobId);
    setJobStatus(null);
    setError(null);
  };

  // Format time elapsed
  const formatDuration = (duration) => {
    if (!duration) return '';
    const hours = Math.floor(duration.asHours());
    const minutes = duration.minutes();
    const seconds = duration.seconds();
    return `${hours}h ${minutes}m ${seconds}s`;
  };

  return (
    <Container className="mt-5 pb-5">
      <Row className="justify-content-center">
        <Col md={8}>
          <Card className="section-container">
            <Card.Body>
              <h3>Track Job Status</h3>
              {!routeJobId && (
                <Form onSubmit={handleCheckStatus}>
                  <Form.Group controlId="jobIdInput">
                    <Form.Label>Enter Job ID</Form.Label>
                    <Form.Control
                      type="text"
                      value={inputJobId}
                      onChange={(e) => setInputJobId(e.target.value)}
                      required
                    />
                  </Form.Group>
                  <div className="mt-4 d-flex justify-content-end">
                    <button type="submit" className="kave-btn">
                      <span className="kave-line"></span>
                      Check Status
                    </button>
                  </div>
                </Form>
              )}
              {error && (
                <Alert variant="danger" className="mt-3">
                  {error}
                </Alert>
              )}
              {jobStatus && (
                <div className="mt-4">
                  <h4>Job Status</h4>
                  <p><strong>Job ID:</strong> {jobStatus.public_id}</p>
                  <p><strong>Status:</strong> {jobStatus.status}</p>
                  {jobStatus.submission_time && (
                    <p><strong>Time Elapsed:</strong> {timeElapsed}</p>
                  )}

                  {jobStatus.status === 'Processing' && (
                    <>
                      {/* Molecule Processing Progress */}
                      {jobStatus.total_molecules > 0 && (
                        <div className="mb-3">
                          <p>Reactions Processed: {jobStatus.molecules_processed} / {jobStatus.total_molecules}</p>
                          <ProgressBar now={(jobStatus.molecules_processed / jobStatus.total_molecules) * 100} />
                        </div>
                      )}
                      {/* Invalid Molecules */}
                      {jobStatus.invalid_molecules > 0 && (
                        <div className="mb-3">
                          <p>Invalid Substrates: {jobStatus.invalid_molecules}</p>
                        </div>
                      )}
                      {/* Predictions Progress */}
                      {jobStatus.total_predictions > 0 && (
                        <div className="mb-3">
                          <p>Predictions Made: {jobStatus.predictions_made} / {jobStatus.total_predictions}</p>
                          <ProgressBar now={(jobStatus.predictions_made / jobStatus.total_predictions) * 100} />
                        </div>
                      )}
                      {/* Spinner during processing */}
                      <div className="mt-4 d-flex justify-content-center">
                        <Spinner animation="border" role="status" />
                      </div>
                    </>
                  )}

                  {jobStatus.status === 'Completed' && jobStatus.output_file_url && (
                    <div>
                      <p>Your job is completed. You can download the results below:</p>
                      <a href={`${apiBaseUrl}${jobStatus.output_file_url}`} download>
                        Download Results
                      </a>
                    </div>
                  )}
                  {jobStatus.error_message && (
                    <ExpandableErrorMessage errorMessage={jobStatus.error_message} />
                  )}

                  {jobStatus.status === 'Failed' && jobStatus.error_message && (
                    <div>
                      <p>Your job failed with the following error:</p>
                      <pre style={{ whiteSpace: 'pre-wrap' }}>{jobStatus.error_message}</pre>
                    </div>
                  )}
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default JobStatus;
