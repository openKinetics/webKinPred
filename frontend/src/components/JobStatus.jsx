import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { Container, Row, Col } from 'react-bootstrap';
import {
  CheckCircleFill,
  XCircleFill,
  ExclamationTriangleFill,
  Clipboard,
  ClipboardCheck,
  ArrowClockwise,
  CloudArrowDown,
} from 'react-bootstrap-icons';
import moment from 'moment';
import ExpandableErrorMessage from './ExpandableErrorMessage';
import apiClient from './appClient';
import '../styles/components/JobStatus.css';

// ─ AnimatedNumber ─────────────────────────────────────────────────────────────
// Plays a snap-down animation whenever `value` changes.
function AnimatedNumber({ value }) {
  const [tick, setTick] = useState(0);
  const prev = useRef(value);
  useEffect(() => {
    if (prev.current !== value) {
      setTick(t => t + 1);
      prev.current = value;
    }
  }, [value]);
  return (
    <span key={tick} className={tick > 0 ? 'trk-num trk-num--snap' : 'trk-num'}>
      {value}
    </span>
  );
}

// ─ StageRow ───────────────────────────────────────────────────────────────────
// One card per prediction target. Includes inline protein embedding progress
// at the bottom so users never have to cross-reference a separate section.
function StageRow({ stage, idx }) {
  // ── Prediction fields ──
  const pred        = stage.prediction || {};
  const made        = num(pred.predictions_made    ?? pred.predictionsMade    ?? 0);
  const predTotal   = num(pred.predictions_total   ?? pred.predictionsTotal   ?? 0);
  const processed   = num(pred.molecules_processed ?? pred.moleculesProcessed ?? 0);
  const molTotal    = num(pred.molecules_total     ?? pred.moleculesTotal     ?? 0);
  const invalid     = num(pred.invalid_rows        ?? pred.invalidRows        ?? 0);

  const ss          = String(stage.status || 'pending').toLowerCase();
  const predPct     = predTotal > 0
    ? Math.min(100, Math.round((made / predTotal) * 100))
    : (ss === 'completed' ? 100 : 0);
  const predMod     = ss === 'completed' ? 'ok' : ss === 'failed' ? 'err' : ss === 'running' ? 'run' : 'dim';
  const isActive    = ss !== 'pending';
  const hasPredData = molTotal > 0 || ss === 'completed' || ss === 'failed';

  const n           = Number.isFinite(Number(stage.index)) ? Number(stage.index) + 1 : idx + 1;
  const method      = stage.method_name || stage.methodName || '';
  const duration    = getStageDuration(stage);

  // ── Embedding fields ──
  const emb          = stage.embedding || {};
  const embEnabled   = Boolean(emb.enabled);
  const embState     = String(emb.state || (embEnabled ? 'pending' : 'not_required')).toLowerCase();
  const embRequired  = embEnabled && embState !== 'not_required';
  const embPending   = embRequired && embState === 'pending';
  const embCached    = num(emb.cached_already   ?? emb.cachedAlready   ?? 0);
  const embNeed      = num(emb.need_computation ?? emb.needComputation ?? 0);
  const embComputed  = num(emb.computed         ?? 0);
  const embRemaining = num(emb.remaining        ?? Math.max(embNeed - embComputed, 0));
  const embTotal     = embCached + embNeed;
  const embPct       = embNeed > 0
    ? Math.min(100, Math.round((embComputed / embNeed) * 100))
    : (embState === 'done' ? 100 : 0);
  const embEs        = embState === 'done' ? 'completed' : embState === 'running' ? 'running' : embState === 'error' ? 'failed' : 'pending';
  const embMod       = embEs === 'completed' ? 'ok' : embEs === 'failed' ? 'err' : embEs === 'running' ? 'run' : 'dim';

  return (
    <div className={`trk__stage trk__stage--${ss}`}>
      <div className="trk__stage-stripe" />
      <div className="trk__stage-body">

        {/* ── Header: index · target name · method badge · duration · state ── */}
        <div className="trk__stage-head">
          <div className="trk__stage-ident">
            <span className="trk__stage-idx">{String(n).padStart(2, '0')}</span>
            <span className="trk__stage-name">{stage.target || `Target ${n}`}</span>
            {method && <span className="trk__stage-method">{method}</span>}
          </div>
          <div className="trk__stage-head-right">
            {duration && <span className="trk__stage-dur">{duration}</span>}
            <span className={`trk__state trk__state--${ss}`}>
              {formatPredictionStageStatus(ss)}
            </span>
          </div>
        </div>
        {/* ── Prediction progress (only once we have real data) ── */}
        {isActive && hasPredData && (
          <div className="trk__pred-block">
            {/* Row-level meta: input count, validated count, invalid count */}
            <div className="trk__pred-meta">
              <span className="trk__pred-meta-item">
                <span className="trk__pred-meta-k">Input</span>
                <span className="trk__pred-meta-v"><AnimatedNumber value={molTotal} /></span>
              </span>
              <span className="trk__pred-meta-sep" />
              <span className="trk__pred-meta-item">
                <span className="trk__pred-meta-k">Validated</span>
                <span className="trk__pred-meta-v"><AnimatedNumber value={processed} /></span>
              </span>
              {invalid > 0 && (
                <>
                  <span className="trk__pred-meta-sep" />
                  <span className="trk__pred-meta-item trk__pred-meta-item--warn">
                    <span className="trk__pred-meta-k">Invalid</span>
                    <span className="trk__pred-meta-v"><AnimatedNumber value={invalid} /></span>
                  </span>
                </>
              )}
            </div>
            {/* Progress bar: predictions made / total */}
            <div className="trk__prog">
              <span className="trk__prog-label">Predictions</span>
              <div className={`trk__prog-track trk__prog-track--${predMod}`}>
                <div className="trk__prog-fill" style={{ width: `${predPct}%` }} />
              </div>
              <span className="trk__prog-frac">
                <AnimatedNumber value={made} />&thinsp;/&thinsp;<AnimatedNumber value={predTotal} />
              </span>
              <span className="trk__prog-pct"><AnimatedNumber value={predPct} />%</span>
            </div>
          </div>
        )}

        {/* ── Protein embedding progress (inline, only when this method needs it) ── */}
        {embRequired && (
          <div className={`trk__emb-block trk__emb-block--${embEs}`}>
            <div className="trk__emb-block-hdr">
              <span className="trk__emb-block-title">Protein Embeddings</span>
              <span className={`trk__state trk__state--${embEs}`}>
                {formatEmbeddingState(embState)}
              </span>
            </div>

            {embPending ? (
              // Avoid showing misleading zeros for a stage that hasn't started yet.
              <div className="trk__emb-block-queued">
                Metrics will appear once this stage begins
              </div>
            ) : (
              <>
                {/*
                  Static equation: shows how the total is composed.
                  These three numbers are fixed from the moment the stage starts
                  and will not change while the job runs.
                */}
                <div className="trk__emb-eq">
                  <span className="trk__emb-eq-part">
                    <span className="trk__emb-eq-val"><AnimatedNumber value={embCached} /></span>
                    <span className="trk__emb-eq-lbl">cached</span>
                  </span>
                  <span className="trk__emb-eq-op">+</span>
                  <span className="trk__emb-eq-part">
                    <span className="trk__emb-eq-val"><AnimatedNumber value={embNeed} /></span>
                    <span className="trk__emb-eq-lbl">to compute</span>
                  </span>
                  <span className="trk__emb-eq-op">=</span>
                  <span className="trk__emb-eq-part trk__emb-eq-part--total">
                    <span className="trk__emb-eq-val"><AnimatedNumber value={embTotal} /></span>
                    <span className="trk__emb-eq-lbl">total</span>
                  </span>
                </div>

                {/* Live progress: computed and remaining tick every poll cycle */}
                {embNeed > 0 && (
                  <div className="trk__emb-live">
                    <div className="trk__prog trk__prog--sm">
                      <span className="trk__prog-label">Computing</span>
                      <div className={`trk__prog-track trk__prog-track--${embMod}`}>
                        <div className="trk__prog-fill" style={{ width: `${embPct}%` }} />
                      </div>
                      <span className="trk__prog-frac">
                        <AnimatedNumber value={embComputed} />&thinsp;/&thinsp;<AnimatedNumber value={embNeed} />
                      </span>
                      <span className="trk__prog-pct"><AnimatedNumber value={embPct} />%</span>
                    </div>
                    {embEs !== 'completed' && embRemaining > 0 && (
                      <div className="trk__emb-remaining">
                        <AnimatedNumber value={embRemaining} /> remaining
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

// ─ JobStatus ──────────────────────────────────────────────────────────────────
function JobStatus() {
  const { public_id: routePublicId } = useParams();
  const initialPublicId = routePublicId || readStoredTrackJobId();

  const [inputPublicId,  setInputPublicId]  = useState(initialPublicId);
  const [publicId,       setPublicId]       = useState(initialPublicId);
  const [jobStatus,      setJobStatus]      = useState(null);
  const [error,          setError]          = useState(null);
  const [queueTime,      setQueueTime]      = useState('');
  const [computeTime,    setComputeTime]    = useState('');
  const [queueSeconds,   setQueueSeconds]   = useState(null);
  const [computeSeconds, setComputeSeconds] = useState(null);
  const [queuePosition,  setQueuePosition]  = useState(null);
  const [isCopying,      setIsCopying]      = useState(false);
  const [isRefreshing,   setIsRefreshing]   = useState(false);
  const [progressStages, setProgressStages] = useState([]);

  // Legacy top-level metrics kept for backwards compat with old API responses
  const [metrics, setMetrics] = useState({
    moleculesProcessed: 0, totalMolecules: 0,
    predictionsMade: 0, totalPredictions: 0, invalidRows: 0,
    embeddingEnabled: false, embeddingState: '',
    embeddingMethodKey: '', embeddingTarget: '',
    embeddingTotal: 0, embeddingCachedAlready: 0,
    embeddingNeedComputation: 0, embeddingComputed: 0, embeddingRemaining: 0,
  });

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api';
  const timerRef   = useRef(null);
  const isMounted  = useRef(false);

  const clearTimer = () => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
  };

  const scheduleNextPoll = useCallback((delayMs) => {
    clearTimer();
    if (delayMs != null) {
      timerRef.current = setTimeout(() => {
        if (isMounted.current) fetchJobStatus(publicId);
      }, delayMs);
    }
  }, [publicId]);

  const fetchJobStatus = useCallback(async (id, { manual = false } = {}) => {
    if (!id) return;
    if (manual) setIsRefreshing(true);
    try {
      const { data } = await apiClient.get(`/job-status/${id}/`);
      if (data.queue_seconds   != null) setQueueSeconds(Number(data.queue_seconds));
      if (data.compute_seconds != null) setComputeSeconds(Number(data.compute_seconds));
      setQueuePosition(data.queue_position ?? null);
      if (!isMounted.current) return;
      setJobStatus(data);
      setError(null);
      setProgressStages(Array.isArray(data.progress_stages) ? data.progress_stages : []);
      setMetrics(() => {
        const emb = data.embedding_progress;
        return {
          moleculesProcessed:      data.molecules_processed,
          totalMolecules:          data.total_molecules,
          predictionsMade:         data.predictions_made,
          totalPredictions:        data.total_predictions,
          invalidRows:             data.invalid_rows,
          embeddingEnabled:        emb ? Boolean(emb.enabled) : false,
          embeddingState:          emb ? (emb.state || '') : '',
          embeddingMethodKey:      emb ? (emb.methodKey || emb.method_key || '') : '',
          embeddingTarget:         emb ? (emb.target || '') : '',
          embeddingTotal:          emb ? Number(emb.total || 0) : 0,
          embeddingCachedAlready:  emb ? Number(emb.cachedAlready  ?? emb.cached_already   ?? 0) : 0,
          embeddingNeedComputation: emb ? Number(emb.needComputation ?? emb.need_computation ?? 0) : 0,
          embeddingComputed:       emb ? Number(emb.computed  || 0) : 0,
          embeddingRemaining:      emb ? Number(emb.remaining || 0) : 0,
        };
      });
      const nextDelay =
        data.status === 'Processing' ? 1000 :
        data.status === 'Pending'    ? 3000 : null;
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
  }, [scheduleNextPoll]);

  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; clearTimer(); };
  }, []);

  useEffect(() => {
    if (routePublicId) { setInputPublicId(routePublicId); setPublicId(routePublicId); return; }
    const stored = readStoredTrackJobId();
    if (stored) { setInputPublicId(stored); setPublicId(stored); }
  }, [routePublicId]);

  useEffect(() => { if (publicId) writeStoredTrackJobId(publicId); }, [publicId]);

  useEffect(() => {
    clearTimer();
    setJobStatus(null); setError(null);
    setQueueSeconds(null); setComputeSeconds(null);
    setQueueTime(''); setComputeTime('');
    setQueuePosition(null); setProgressStages([]);
    setMetrics({
      moleculesProcessed: 0, totalMolecules: 0,
      predictionsMade: 0, totalPredictions: 0, invalidRows: 0,
      embeddingEnabled: false, embeddingState: '', embeddingMethodKey: '',
      embeddingTarget: '', embeddingTotal: 0, embeddingCachedAlready: 0,
      embeddingNeedComputation: 0, embeddingComputed: 0, embeddingRemaining: 0,
    });
    if (publicId) fetchJobStatus(publicId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [publicId]);

  // Queue time: ticks up every second while status is Pending
  useEffect(() => {
    if (queueSeconds == null) return;
    setQueueTime(formatDuration(moment.duration(queueSeconds * 1000)));
    if (jobStatus?.status !== 'Pending') return;
    const id = setInterval(() => {
      setQueueSeconds(s => {
        const n = (s || 0) + 1;
        setQueueTime(formatDuration(moment.duration(n * 1000)));
        return n;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [queueSeconds, jobStatus?.status]);

  // Compute time: ticks up every second while status is Processing
  useEffect(() => {
    if (computeSeconds == null) return;
    setComputeTime(formatDuration(moment.duration(computeSeconds * 1000)));
    if (jobStatus?.status !== 'Processing') return;
    const id = setInterval(() => {
      setComputeSeconds(s => {
        const n = (s || 0) + 1;
        setComputeTime(formatDuration(moment.duration(n * 1000)));
        return n;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [computeSeconds, jobStatus?.status]);

  const handleCheckStatus  = (e) => { e.preventDefault(); if (inputPublicId.trim()) setPublicId(inputPublicId.trim()); };
  const handleManualRefresh = () => { if (publicId) { clearTimer(); fetchJobStatus(publicId, { manual: true }); } };
  const copyToClipboard    = async (text) => {
    try { await navigator.clipboard.writeText(text); setIsCopying(true); setTimeout(() => setIsCopying(false), 1200); } catch {}
  };

  const statusKey = (jobStatus?.status || 'none').toLowerCase();

  const statusLabel = useMemo(() => {
    const s = jobStatus?.status;
    if (s === 'Completed')  return 'Completed';
    if (s === 'Failed')     return 'Failed';
    if (s === 'Processing') return 'Processing';
    if (s === 'Pending')    return 'Pending';
    return '—';
  }, [jobStatus]);

  // Prefer per-stage progress_stages from API; fall back to legacy top-level fields
  const normalizedStages = useMemo(() => {
    if (Array.isArray(progressStages) && progressStages.length > 0) return progressStages;
    if (!jobStatus || jobStatus.status === 'Pending') return [];
    return buildLegacyStages(jobStatus, metrics);
  }, [progressStages, jobStatus, metrics]);

  // Parse skipped-row warnings (JSON array of { rows, reason } objects)
  const skippedRowsMessage = useMemo(() => {
    if (!jobStatus) return null;
    const raw = jobStatus.error_message;
    if (!raw) return null;
    if (typeof raw === 'string' && raw.trimStart().startsWith('[')) {
      try {
        const groups = JSON.parse(raw);
        if (Array.isArray(groups) && groups.length > 0) {
          return groups.map(({ rows, reason }) => {
            const label = Array.isArray(rows) && rows.length > 0
              ? (rows.length === 1 ? `Row ${rows[0] + 1}` : `Rows ${rows.map(r => r + 1).join(', ')}`)
              : 'Some rows';
            return `${reason || 'Unknown reason'} — ${label}`;
          }).join('\n');
        }
      } catch {}
    }
    return typeof raw === 'string' ? raw : null;
  }, [jobStatus]);

  const failedMessage = useMemo(() => {
    if (jobStatus?.status !== 'Failed') return null;
    for (const v of [
      jobStatus?.error_message, jobStatus?.error,
      jobStatus?.message,       jobStatus?.detail,
      jobStatus?.failure_reason,
    ]) {
      if (typeof v === 'string' && v.trim()) return sanitiseErrorForUser(v.trim());
      if (Array.isArray(v) && v.length > 0)  return sanitiseErrorForUser(v.map(String).join('\n'));
    }
    return sanitiseErrorForUser('');
  }, [jobStatus]);

  const stageSummary = useMemo(() => {
    const s = { total: normalizedStages.length, completed: 0, running: 0, pending: 0, failed: 0 };
    normalizedStages.forEach(st => {
      const state = String(st.status || 'pending').toLowerCase();
      if      (state === 'completed') s.completed++;
      else if (state === 'running')   s.running++;
      else if (state === 'failed')    s.failed++;
      else                            s.pending++;
    });
    return s;
  }, [normalizedStages]);

  return (
    <Container className="trk-page pb-4">
      <Row className="justify-content-center">
        <Col md={10} lg={9}>

          {/* ── Search form (only at /track-job without an ID in the URL) ── */}
          {!routePublicId && (
            <form onSubmit={handleCheckStatus} className="trk-search section-container mb-3">
              <label className="trk-search__lbl">Enter Job ID</label>
              <div className="trk-search__row">
                <input
                  type="text"
                  className="trk-search__input"
                  value={inputPublicId}
                  placeholder="e.g. pl1a2V1"
                  onChange={e => setInputPublicId(e.target.value)}
                  required
                />
                <button type="submit" className="trk-search__btn">Track</button>
              </div>
              <p className="trk-search__hint">Your last job ID is saved locally in this browser.</p>
            </form>
          )}

          {/* ── Network error banner ── */}
          {error && (
            <div className="trk-alert mb-2">
              <ExclamationTriangleFill size={13} />
              <span>{error}</span>
            </div>
          )}

          {/* ── Main tracker panel ── */}
          {publicId && (
            <div className="trk section-container">

              {/* ── Identity bar: Job ID + status badge + refresh button ── */}
              <div className="trk__bar">
                <div className="trk__id">
                  <span className="trk__id-eyebrow">Job ID</span>
                  <div className="trk__id-row">
                    <code className="trk__id-code">{publicId}</code>
                    <button
                      className="trk__icon-btn"
                      onClick={() => copyToClipboard(publicId)}
                      title="Copy Job ID"
                    >
                      {isCopying ? <ClipboardCheck size={13} /> : <Clipboard size={13} />}
                    </button>
                  </div>
                </div>
                <div className="trk__bar-right">
                  {jobStatus && (
                    <div className={`trk__status trk__status--${statusKey}`}>
                      <span className="trk__status-dot" />
                      <span className="trk__status-lbl">{statusLabel}</span>
                    </div>
                  )}
                  <button
                    className="trk__icon-btn"
                    onClick={handleManualRefresh}
                    disabled={isRefreshing}
                    title="Refresh"
                  >
                    <ArrowClockwise size={15} className={isRefreshing ? 'spin' : ''} />
                  </button>
                </div>
              </div>

              {/* ── Timing readouts in Orbitron ── */}
              {jobStatus && (queueTime || computeTime) && (
                <div className="trk__timing">
                  {queueTime && (
                    <div className="trk__clock">
                      <div className="trk__clock-lbl">Queue Time</div>
                      <div className="trk__clock-val">{queueTime}</div>
                      <div className="trk__clock-sub">Time waited before processing began</div>
                    </div>
                  )}
                  {queueTime && computeTime && <div className="trk__timing-div" />}
                  {computeTime && (
                    <div className="trk__clock">
                      <div className="trk__clock-lbl">{jobStatus.execution_state ? 'Processing Elapsed' : 'Compute Time'}</div>
                      <div className="trk__clock-val">{computeTime}</div>
                      <div className="trk__clock-sub">{jobStatus.execution_state ? 'Time since processing began' : 'Active processing time on server'}</div>
                    </div>
                  )}
                </div>
              )}

              {typeof jobStatus?.active_execution_seconds === 'number' && (
                <div className="trk__timing">
                  <div className="trk__clock">
                    <div className="trk__clock-lbl">Active Execution</div>
                    <div className="trk__clock-val">{formatDuration(moment.duration(jobStatus.active_execution_seconds * 1000))}</div>
                    <div className="trk__clock-sub">Time spent executing work</div>
                  </div>
                  <div className="trk__timing-div" />
                  <div className="trk__clock">
                    <div className="trk__clock-lbl">Resource Wait</div>
                    <div className="trk__clock-val">{formatDuration(moment.duration((jobStatus.resource_wait_seconds ?? 0) * 1000))}</div>
                    <div className="trk__clock-sub">Time waiting for available capacity</div>
                  </div>
                </div>
              )}

              {jobStatus?.execution_state && !['completed', 'failed'].includes(jobStatus.execution_state) && (
                <div className="trk__pending" role="status" aria-live="polite">
                  <span className="trk__pending-text">{{
                    waiting_gpu: 'Waiting for GPU capacity',
                    waiting_cpu: 'Waiting for processing capacity',
                    paused: 'Processing is paused for maintenance',
                    retrying: 'Recovering interrupted work; completed results are preserved',
                    needs_attention: 'Processing is held for operator attention; your job is preserved',
                    running: 'Processing your job',
                  }[jobStatus.execution_state] || 'Waiting for available resources'}</span>
                </div>
              )}
              {/* ── Pending: bouncing dots + queue position ── */}
              {jobStatus?.status === 'Pending' && (
                <div className="trk__pending">
                  <div className="trk__pending-dots">
                    <span /><span /><span />
                  </div>
                  <span className="trk__pending-txt">
                    {queuePosition != null
                      ? <><strong className="trk__pending-pos">#{queuePosition}</strong>&ensp;in queue</>
                      : 'Queued — waiting to start'
                    }
                  </span>
                </div>
              )}

              {/* ── Per-target progress (each stage includes its embedding inline) ── */}
              {jobStatus && (
                <div className="trk__section">
                  <div className="trk__section-hdr">
                    <span className="trk__section-title">Predictions</span>
                    {normalizedStages.length > 0 && (
                      <div className="trk__chips">
                        <span className="trk__chip">{stageSummary.total} targets</span>
                        {stageSummary.completed > 0 && <span className="trk__chip trk__chip--ok">{stageSummary.completed} done</span>}
                        {stageSummary.running   > 0 && <span className="trk__chip trk__chip--run">{stageSummary.running} running</span>}
                        {stageSummary.pending   > 0 && <span className="trk__chip trk__chip--dim">{stageSummary.pending} pending</span>}
                        {stageSummary.failed    > 0 && <span className="trk__chip trk__chip--err">{stageSummary.failed} failed</span>}
                      </div>
                    )}
                  </div>

                  {normalizedStages.length === 0 ? (
                    <div className="trk__empty">Target progress will appear once processing begins.</div>
                  ) : (
                    <div className="trk__stage-list">
                      {normalizedStages.map((stage, idx) => (
                        <StageRow
                          key={`${stage.index ?? idx}-${stage.target}-${stage.method_key || ''}`}
                          stage={stage}
                          idx={idx}
                        />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* ── Completed: success banner + download link ── */}
              {jobStatus?.status === 'Completed' && (
                <div className="trk__done">
                  <div className="trk__done-left">
                    <CheckCircleFill size={16} className="trk__done-icon" />
                    <span>All predictions complete — results ready for download.</span>
                  </div>
                  <a className="trk__download-btn" href={`${apiBaseUrl}/jobs/${publicId}/download/`}>
                    <CloudArrowDown size={15} />
                    Download Results
                  </a>
                </div>
              )}

              {/* ── Skipped rows (partial input failures) ── */}
              {skippedRowsMessage && (
                <div className="trk__expandable">
                  <ExpandableErrorMessage errorMessage={skippedRowsMessage} />
                </div>
              )}

              {/* ── Failed: error banner ── */}
              {jobStatus?.status === 'Failed' && (
                <div className="trk__fail">
                  <div className="trk__fail-head">
                    <XCircleFill size={15} />
                    <span>Job Failed</span>
                  </div>
                  <div className="trk__fail-body">{failedMessage}</div>
                </div>
              )}

            </div>
          )}

        </Col>
      </Row>
    </Container>
  );
}

export default JobStatus;

// ─ Helpers ────────────────────────────────────────────────────────────────────

function getStageDuration(stage) {
  if (!stage.started_at || !stage.completed_at) return null;
  const start = new Date(stage.started_at);
  const end   = new Date(stage.completed_at);
  if (isNaN(start) || isNaN(end)) return null;
  const diffMs = end - start;
  return diffMs >= 0 ? formatDuration(moment.duration(diffMs)) : null;
}

function sanitiseErrorForUser(raw) {
  if (!raw || typeof raw !== 'string')
    return 'An unexpected error occurred while processing your job. Please try again or contact support.';
  const lower = raw.toLowerCase();
  if (lower.includes('out of memory') || lower.includes('oom') || lower.includes('sigkill') ||
      lower.includes('killed') || lower.includes('returncode == 137') || lower.includes('exit status 137') ||
      lower.includes('returncode == -9') || lower.includes('exit status -9'))
    return 'The prediction model ran out of memory while processing your data. Try reducing the number of rows or the length of your sequences and resubmit.';
  if (lower.includes('returned non-zero exit status') || lower.includes('calledprocesserror') || lower.includes('non-zero exit'))
    return 'The prediction model encountered an internal error and could not complete. Please verify your input data and try again.';
  if (lower.includes('timeout') || lower.includes('timed out'))
    return 'The prediction timed out. Try reducing the number of rows and resubmitting.';
  if (lower.includes('missing column'))
    return 'Your input file is missing one or more required columns. Please check the expected CSV format and resubmit.';
  if (lower.includes('failed to read input csv'))
    return 'The uploaded CSV file could not be read. Please ensure it is a valid CSV file and try again.';
  const hasInternalDetails =
    /\/[a-z_/]+\.[a-z]+/i.test(raw) || /exit status/i.test(raw) ||
    /traceback/i.test(raw) || /\bFile "/.test(raw);
  if (hasInternalDetails)
    return 'The prediction model encountered an unexpected error. Please verify your input data and try again. If the problem persists, contact support.';
  return raw;
}

function formatPredictionStageStatus(state) {
  const s = String(state || '').toLowerCase();
  if (s === 'running')   return 'Running';
  if (s === 'completed') return 'Completed';
  if (s === 'failed')    return 'Failed';
  if (s === 'pending')   return 'Pending';
  return humanizeState(s || 'pending');
}

function formatEmbeddingState(state) {
  const s = String(state || '').toLowerCase();
  if (s === 'done')         return 'Completed';
  if (s === 'running')      return 'Computing';
  if (s === 'error')        return 'Error';
  if (s === 'not_required') return 'Not Required';
  if (s === 'pending')      return 'Queued';
  return humanizeState(s || 'pending');
}

function humanizeState(value) {
  const t = String(value || '').replace(/_/g, ' ').trim();
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : 'Pending';
}

function buildLegacyStages(jobStatus, metrics) {
  const globalStatus =
    jobStatus?.status === 'Completed' ? 'completed' :
    jobStatus?.status === 'Failed'    ? 'failed'    :
    jobStatus?.status === 'Processing'? 'running'   : 'pending';
  const targets     = parsePredictionTargets(jobStatus?.prediction_type);
  const safeTargets = targets.length > 0 ? targets : ['Prediction'];
  return safeTargets.map((target, index) => {
    const isPrimary = index === 0;
    return {
      index, target,
      method_name: legacyMethodForTarget(target, jobStatus),
      method_key:  '',
      status:      legacyStageStatusForIndex(globalStatus, index),
      prediction: {
        molecules_total:    isPrimary ? (metrics.totalMolecules   || 0) : 0,
        molecules_processed:isPrimary ? (metrics.moleculesProcessed || 0) : 0,
        invalid_rows:       isPrimary ? (metrics.invalidRows      || 0) : 0,
        predictions_total:  isPrimary ? (metrics.totalPredictions || 0) : 0,
        predictions_made:   isPrimary ? (metrics.predictionsMade  || 0) : 0,
      },
      embedding: isPrimary && metrics.embeddingEnabled
        ? {
            enabled:          true,
            state:            metrics.embeddingState || 'running',
            total:            metrics.embeddingTotal || 0,
            cached_already:   metrics.embeddingCachedAlready    || 0,
            need_computation: metrics.embeddingNeedComputation  || 0,
            computed:         metrics.embeddingComputed         || 0,
            remaining:        metrics.embeddingRemaining        || 0,
          }
        : { enabled: false, state: 'not_required' },
      synthetic: true,
    };
  });
}

function parsePredictionTargets(predictionType) {
  if (!predictionType || typeof predictionType !== 'string') return [];
  return predictionType.split('+').map(p => p.trim()).filter(Boolean);
}

function legacyMethodForTarget(target, jobStatus) {
  if (target === 'kcat')    return jobStatus?.kcat_method    || '';
  if (target === 'Km')      return jobStatus?.km_method      || '';
  if (target === 'kcat/Km') return jobStatus?.kcat_km_method || '';
  return '';
}

function legacyStageStatusForIndex(globalStatus, index) {
  if (globalStatus === 'completed') return 'completed';
  if (globalStatus === 'failed')    return index === 0 ? 'failed'  : 'pending';
  if (globalStatus === 'running')   return index === 0 ? 'running' : 'pending';
  return 'pending';
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}

function formatDuration(duration) {
  if (!duration) return '';
  const h = Math.floor(duration.asHours());
  return `${h}h ${pad(duration.minutes())}m ${pad(duration.seconds())}s`;
}

function pad(n) { return String(n).padStart(2, '0'); }

const STORAGE_KEY = 'trackJob:lastPublicId';

function readStoredTrackJobId() {
  if (typeof window === 'undefined') return '';
  try { const r = window.localStorage.getItem(STORAGE_KEY); return typeof r === 'string' ? r.trim() : ''; } catch { return ''; }
}

function writeStoredTrackJobId(value) {
  if (typeof window === 'undefined') return;
  const id = String(value || '').trim();
  if (!id) return;
  try { window.localStorage.setItem(STORAGE_KEY, id); } catch {}
}
