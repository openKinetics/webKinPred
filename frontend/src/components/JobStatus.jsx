import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import {
  Form,
  Container,
  Row,
  Col,
  Card,
  Alert,
  ProgressBar,
  Spinner,
  Button,
  Badge
} from 'react-bootstrap';
import {
  HourglassSplit,
  CheckCircle,
  XCircle,
  ExclamationTriangle,
  Clipboard,
  ClipboardCheck,
  ArrowClockwise,
  FileEarmarkArrowDown,
  Stopwatch
} from 'react-bootstrap-icons';
import moment from 'moment';
import ExpandableErrorMessage from './ExpandableErrorMessage';
import apiClient from './appClient';
import '../styles/components/JobStatus.css';

function JobStatus() {
  const { public_id: routePublicId } = useParams();
  const [inputPublicId, setInputPublicId] = useState(routePublicId || '');
  const [publicId, setPublicId] = useState(routePublicId || '');

  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [timeElapsed, setTimeElapsed] = useState('');
  const [isCopying, setIsCopying] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Persist progress so x/y never drops to 0/0 when the job completes
  const [metrics, setMetrics] = useState({
    moleculesProcessed: 0,
    totalMolecules: 0,
    predictionsMade: 0,
    totalPredictions: 0,
    invalidMolecules: 0,
  });

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api';

  // Polling control
  const timerRef = useRef(null);
  const isMounted = useRef(false);

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const scheduleNextPoll = useCallback(
    (delayMs) => {
      clearTimer();
      if (delayMs != null) {
        timerRef.current = setTimeout(() => {
          if (isMounted.current) fetchJobStatus(publicId);
        }, delayMs);
      }
    },
    [publicId]
  );

  const fetchJobStatus = useCallback(
    async (id, { manual = false } = {}) => {
      if (!id) return;
      if (manual) setIsRefreshing(true); // only for manual
      try {
        const response = await apiClient.get(`/job-status/${id}/`);
        const data = response.data;

        if (!isMounted.current) return;

        setJobStatus(data);
        setError(null);

        setMetrics(() => ({
          moleculesProcessed: data.molecules_processed,
          totalMolecules: data.total_molecules,
          predictionsMade: data.predictions_made,
          totalPredictions: data.total_predictions,
          invalidMolecules: data.invalid_molecules,
        }));

        const nextDelay =
          data.status === 'Processing' ? 1000 :
          data.status === 'Pending'    ? 3000 :
          null;

        scheduleNextPoll(nextDelay);
      } catch (err) {
        console.error('Error fetching job status:', err);
        if (isMounted.current) {
          setError('Unable to fetch job status. Retrying…');
          scheduleNextPoll(5000);
        }
      } finally {
        if (manual && isMounted.current) setIsRefreshing(false);
      }
    },
    [scheduleNextPoll]
  );


  // Mount / unmount
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      clearTimer();
    };
  }, []);

  // Kick off or restart polling only when the job ID changes
  useEffect(() => {
    clearTimer();
    setJobStatus(null);
    setError(null);
    // Reset sticky metrics for a new job
    setMetrics({
      moleculesProcessed: 0,
      totalMolecules: 0,
      predictionsMade: 0,
      totalPredictions: 0,
      invalidMolecules: 0,
    });
    if (publicId) fetchJobStatus(publicId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [publicId]);

  // Elapsed time updater
  useEffect(() => {
    if (!(jobStatus && jobStatus.submission_time)) return;

    const tick = () => {
      const submissionTime = moment(jobStatus.submission_time);
      const end =
        jobStatus.status === 'Completed' && jobStatus.completion_time
          ? moment(jobStatus.completion_time)
          : moment();
      const duration = moment.duration(end.diff(submissionTime));
      setTimeElapsed(formatDuration(duration));
    };

    // initial render
    tick();
    // Update every second whilst active
    const active = jobStatus.status === 'Processing' || jobStatus.status === 'Pending';
    const id = active ? setInterval(tick, 1000) : null;

    return () => {
      if (id) clearInterval(id);
    };
  }, [jobStatus]);

  const handleCheckStatus = (e) => {
    e.preventDefault();
    if (!inputPublicId.trim()) return;
    setPublicId(inputPublicId.trim());
  };

  const handleManualRefresh = () => {
    if (publicId) {
      clearTimer();
      fetchJobStatus(publicId, { manual: true });
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setIsCopying(true);
      setTimeout(() => setIsCopying(false), 1200);
    } catch {
      // ignore
    }
  };

  const statusMeta = useMemo(() => {
    const s = jobStatus?.status;
    if (s === 'Completed') return { variant: 'success', icon: <CheckCircle className="me-1" />, label: 'Completed' };
    if (s === 'Failed') return { variant: 'danger', icon: <XCircle className="me-1" />, label: 'Failed' };
    if (s === 'Processing') return { variant: 'info', icon: <HourglassSplit className="me-1" />, label: 'Processing' };
    if (s === 'Pending') return { variant: 'secondary', icon: <HourglassSplit className="me-1" />, label: 'Pending' };
    return { variant: 'secondary', icon: <HourglassSplit className="me-1" />, label: '—' };
  }, [jobStatus]);

  // Percentages based on sticky metrics
  const moleculesPct = useMemo(() => {
    const done = metrics.moleculesProcessed || 0;
    const total = metrics.totalMolecules || 0;
    return total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  }, [metrics]);

  const predsPct = useMemo(() => {
    const made = metrics.predictionsMade || 0;
    const total = metrics.totalPredictions || 0;
    return total > 0 ? Math.min(100, Math.round((made / total) * 100)) : 0;
  }, [metrics]);

  // Build a nice expandable block for rows we couldn’t predict (if the API returns any flavour of this)
  const skippedRowsMessage = useMemo(() => {
    if (!jobStatus) return null;
    const parts = [];

    const collect = (value) => {
      if (!value) return;
      if (Array.isArray(value)) {
        if (value.length === 0) return;
        const lines = value.map((item) => {
          if (item == null) return '';
          if (typeof item === 'object') {
            const row = item.row ?? item.index ?? item.line ?? item.id ?? '';
            const reason = item.reason ?? item.error ?? item.message ?? '';
            if (row !== '' && reason) return `Row ${row}: ${reason}`;
            if (row !== '') return `Row ${row}`;
            return String(reason || JSON.stringify(item));
          }
          // primitive
          return isFinite(item) ? `Row ${item}` : String(item);
        }).filter(Boolean);
        if (lines.length) {
          parts.push(`${lines.join('\n')}`);
        }
      } else if (typeof value === 'string') {
        parts.push(`${value}`);
      }
    };

    // Try a few common keys
    collect(jobStatus.error_message);

    if (parts.length === 0) return null;
    return parts.join('\n\n');
  }, [jobStatus]);

  return (
    <Container className="mt-5 pb-5">
      <Row className="justify-content-center">
        <Col md={10} lg={9}>
          <Card className="section-container job-status-card mb-4">
            <Card.Header as="h3" className="text-center">Track Job Status</Card.Header>

            <Card.Body>
              {!routePublicId && (
                <Form onSubmit={handleCheckStatus} className="mb-4">
                  <Form.Group controlId="jobIdInput">
                    <Form.Label className="mb-2">Enter Job ID</Form.Label>
                    <div className="d-flex gap-2">
                      <Form.Control
                        type="text"
                        value={inputPublicId}
                        placeholder="e.g., 8f4e7a9b-1234-4acb-9d01-abcdef123456"
                        onChange={(e) => setInputPublicId(e.target.value)}
                        required
                        className="kave-input"
                      />
                      <Button type="submit" className="kave-btn">
                        <span className="kave-line"></span>
                        Check Status
                      </Button>
                    </div>
                  </Form.Group>
                </Form>
              )}

              {error && (
                <Alert variant="warning" className="d-flex align-items-center">
                  <ExclamationTriangle className="me-2" />
                  <div>{error}</div>
                </Alert>
              )}

              {publicId && (
                <div className="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3">
                  <div className="d-flex align-items-center gap-2">
                    <span className="label-muted">Job ID:</span>
                    <code className="jobid-chip">{publicId}</code>
                    <Button
                      variant="outline-light"
                      size="sm"
                      className="chip-action"
                      onClick={() => copyToClipboard(publicId)}
                    >
                      {isCopying ? <ClipboardCheck size={16} /> : <Clipboard size={16} />}
                    </Button>
                  </div>

                  <div className="d-flex align-items-center gap-2">
                    <Badge bg={statusMeta.variant} className="status-pill">
                      {statusMeta.icon}{statusMeta.label}
                    </Badge>
                    <Button
                      size="sm"
                      className="btn btn-custom-subtle"
                      onClick={handleManualRefresh}
                      disabled={isRefreshing}
                    >
                      <ArrowClockwise className={`me-1 ${isRefreshing ? 'spin' : ''}`} />
                      Refresh
                    </Button>
                  </div>
                </div>
              )}

              {jobStatus && (
                <>
                  {/* Stats always show sticky x/y, even after completion */}
                  <Row className="mb-3 g-3 stats-grid">
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label"><Stopwatch className="me-2" />Time Elapsed</div>
                        <div className="stat-value">{timeElapsed || '—'}</div>
                      </div>
                    </Col>
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label">Preprocessed</div>
                        <div className="stat-value">
                          {metrics.moleculesProcessed}
                          <span className="stat-sub"> / {metrics.totalMolecules}</span>
                        </div>
                      </div>
                    </Col>
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label">Predictions</div>
                        <div className="stat-value">
                          {metrics.predictionsMade}
                          <span className="stat-sub"> / {metrics.totalPredictions}</span>
                        </div>
                      </div>
                    </Col>
                    <Col sm={6} lg={3}>
                      <div className="stat-card">
                        <div className="stat-label">Invalid Rows</div>
                        <div className="stat-value">{metrics.invalidMolecules}</div>
                      </div>
                    </Col>
                  </Row>

                  {(jobStatus.status === 'Processing' || jobStatus.status === 'Pending') && (
                    <>
                      {metrics.totalMolecules && (
                        <div className="mb-3">
                          <div className="progress-row">
                            <div className="progress-title">Reactions Processed</div>
                            <div className="progress-count">{moleculesPct}%</div>
                          </div>
                          <ProgressBar now={moleculesPct} className="kave-progress" />
                        </div>
                      )}

                      {metrics.totalPredictions && (
                        <div className="mb-2">
                          <div className="progress-row">
                            <div className="progress-title">Predictions Made</div>
                            <div className="progress-count">{predsPct}%</div>
                          </div>
                          <ProgressBar now={predsPct} className="kave-progress" />
                        </div>
                      )}

                      <p className="note-muted mb-4">
                        Real-time progress is only accurate for <strong>TurNup</strong>, <strong>DLKcat</strong>, and <strong>EITLEM-Kinetics</strong>.
                      </p>

                      <div className="mt-3 d-flex justify-content-center">
                        <Spinner animation="border" role="status" />
                      </div>
                    </>
                  )}

                  {jobStatus.status === 'Completed' && (
                    <div className="d-flex align-items-center justify-content-between flex-wrap gap-2 mt-3">
                      <div>Job completed. Download your results below.</div>
                        <a
                          className="btn btn-custom-subtle"
                          href={`${apiBaseUrl}/jobs/${publicId}/download/`}
                        >
                          <FileEarmarkArrowDown className="me-2" />
                          Download Results
                        </a>
                    </div>
                  )}

                  {/* Beautiful, expandable details for skipped/unpredicted rows (if any) */}
                  {skippedRowsMessage && (
                    <div className="mt-3">
                      <ExpandableErrorMessage errorMessage={skippedRowsMessage} />
                    </div>
                  )}
                  {/* Keep main backend error (if any) in a separate block */}
                  {jobStatus.status === 'Failed' && (
                    <Alert variant="danger" className="mt-3">
                      <div className="fw-semibold mb-2">Job failed</div>
                      {jobStatus.error_message ? (
                        <ExpandableErrorMessage errorMessage={jobStatus.error_message} />
                      ) : (
                        <div>No error message provided.</div>
                      )}
                    </Alert>
                  )}
                </>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default JobStatus;

// Helpers
function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}
function formatDuration(duration) {
  if (!duration) return '';
  const hours = Math.floor(duration.asHours());
  const minutes = duration.minutes();
  const seconds = duration.seconds();
  return `${hours}h ${pad(minutes)}m ${pad(seconds)}s`;
}
function pad(n) {
  return String(n).padStart(2, '0');
}
