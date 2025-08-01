import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Form, Container, Row, Col, Card, Alert, ProgressBar, Spinner } from 'react-bootstrap';
import moment from 'moment';
import ExpandableErrorMessage from './ExpandableErrorMessage';
import apiClient from './appClient';

function JobStatus() {
  const { public_id: routePublicId } = useParams();
  const [inputPublicId, setInputPublicId] = useState(routePublicId || '');
  const [public_id, setPublicId] = useState(routePublicId || '');
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [timeElapsed, setTimeElapsed] = useState('');
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL;
  const intervalDuration = 1000; 
  const [pollingInterval, setPollingInterval] = useState(intervalDuration);
  let intervalId = null; 

  useEffect(() => {
    if (public_id) {
      const fetchJobStatus = () => {
        apiClient.get(`/api/job-status/${public_id}/`)
          .then(response => {
            setJobStatus(response.data);
            setError(null);
            console.log('Job status fetched:', response.data);
            
            if (response.data.status === 'Completed' || response.data.status === 'Failed') {
              clearInterval(intervalId); 
            } else {
              setPollingInterval(intervalDuration); 
            }
          })
          .catch(error => {
            console.error('Error fetching job status:', error);
            setError('Failed to fetch job status');
            setJobStatus(null);
          });
      };

      fetchJobStatus();
      intervalId = setInterval(fetchJobStatus, pollingInterval);
    }
    return () => {
      clearInterval(intervalId); // Cleanup interval when component unmounts or public_id changes
    };
  }, [public_id, pollingInterval]); // Re-run effect when public_id or pollingInterval changes

  useEffect(() => {
    if (jobStatus && jobStatus.submission_time) {
      const submissionTime = moment(jobStatus.submission_time);
      let endTime = moment();

      if (jobStatus.status === 'Completed' && jobStatus.completion_time) {
        endTime = moment(jobStatus.completion_time); // Use completion time for completed jobs
      }

      const duration = moment.duration(endTime.diff(submissionTime));
      setTimeElapsed(formatDuration(duration));
    }
  }, [jobStatus]);

  const handleCheckStatus = (e) => {
    e.preventDefault();
    setPublicId(inputPublicId);
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
              {!routePublicId && (
                <Form onSubmit={handleCheckStatus}>
                  <Form.Group controlId="jobIdInput">
                    <Form.Label>Enter Job ID</Form.Label>
                    <Form.Control
                      type="text"
                      value={inputPublicId}
                      onChange={(e) => setInputPublicId(e.target.value)}
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
                          <p className="mb-2" style={{ color: '#808080' }}>
                            (Note: Real-time progress updates are only accurate for TurNup, DLKcat, and EITLEM-kinetics.)
                          </p>
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
