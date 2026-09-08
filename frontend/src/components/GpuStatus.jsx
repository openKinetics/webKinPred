import React, { useEffect, useState } from 'react';
import { Table, Container, Row, Col, Alert } from 'react-bootstrap';
import { ChevronDown, ChevronUp, Speedometer2 } from 'react-bootstrap-icons';
import apiClient from './appClient';
import '../styles/components/GpuStatus.css';

const BENCHMARK_DATA = [
  {
    method: 'DLKcat',
    uncachedCpu: '32 s',
    uncachedGpu: 'N/A',
    uncachedGpuTooltip: 'not implemented',
    cached: 'N/A',
    cachedTooltip: 'Does not use PLM embeddings',
  },
  {
    method: 'MMISA-KM',
    uncachedCpu: '4 min 22 s',
    uncachedGpu: 'N/A',
    uncachedGpuTooltip: 'not implemented',
    cached: 'N/A',
    cachedTooltip: 'not implemented because no embeddings and contact map is already fast enough',
  },
  { method: 'CatPred',   uncachedCpu: '14 min 0 s',   uncachedGpu: '5 min 54 s', cached: '23 s'      },
  { method: 'CatRange',  uncachedCpu: '14 min 43 s',  uncachedGpu: '1 min 38 s', cached: '1 min 9 s'  },
  {
    method: 'EITLEM',
    uncachedCpu: '18 min 13 s',
    uncachedGpu: '7 min 43 s',
    cached: 'N/A',
    cachedTooltip: 'uses full per-residue embeddings, thus not cached on server',
  },
  {
    method: 'OmniESI',
    uncachedCpu: '18 min 41 s',
    uncachedGpu: '7 min 34 s',
    cached: 'N/A',
    cachedTooltip: 'uses full per-residue embeddings, thus not cached on server',
  },
  { method: 'TurNup',    uncachedCpu: '19 min 36 s',  uncachedGpu: '3 min 53 s', cached: '2 min 12 s' },
  { method: 'CataPro',   uncachedCpu: '25 min 9 s',   uncachedGpu: '1 min 37 s', cached: '41 s'       },
  { method: 'UniKP',     uncachedCpu: '33 min 46 s',  uncachedGpu: '1 min 28 s', cached: '58 s'       },
  {
    method: 'IECata',
    uncachedCpu: '34 min 41 s',
    uncachedGpu: '11 min 11 s',
    cached: 'N/A',
    cachedTooltip: 'uses full per-residue embeddings, thus not cached on server',
  },
  { method: 'KinForm-L', uncachedCpu: '54 min 38 s',  uncachedGpu: '3 min 49 s', cached: '37 s'       },
  { method: 'KinForm-H', uncachedCpu: '56 min 10 s',  uncachedGpu: '3 min 42 s', cached: '36 s'       },
];

const FALLBACK_STATUS = { configured: false, online: false, mode: 'cpu' };

export default function GpuStatus({ layout = 'home' }) {
  const [status, setStatus] = useState(FALLBACK_STATUS);
  const [isCheckingStatus, setIsCheckingStatus] = useState(true);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const fetchStatus = async () => {
      try {
        const { data } = await apiClient.get('/v1/gpu/status/');
        if (!cancelled) setStatus(data || FALLBACK_STATUS);
      } catch (_) {
        if (!cancelled) setStatus(FALLBACK_STATUS);
      } finally {
        if (!cancelled) setIsCheckingStatus(false);
      }
    };
    fetchStatus();
    const timer = setInterval(fetchStatus, 15000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  const isOnline = !isCheckingStatus && status.configured && status.online;

  let remaining = null;
  if (isOnline) {
    const now = new Date();
    const cutoff = new Date(Date.UTC(
      now.getUTCFullYear(), now.getUTCMonth(),
      now.getUTCDate() + (now.getUTCHours() >= 8 ? 1 : 0),
      8, 0, 0, 0
    ));
    const msLeft = cutoff - now;
    const hLeft = Math.floor(msLeft / 3600000);
    const mLeft = Math.floor((msLeft % 3600000) / 60000);
    remaining = hLeft > 0 ? `${hLeft}h ${mLeft}m` : `${mLeft}m`;
  }

  return (
    <section className="gpu-status-shell">
      <Container className="gpu-status-container">
        <Row className="justify-content-center">
          <Col md={10} lg={layout === 'track' ? 9 : undefined}>
            <Alert variant="info" className="d-flex align-items-start gpu-status-alert mb-0">
              <Speedometer2 size={24} className="me-3 mt-1 gpu-status-icon" />
              <div className="gpu-status-content">
                <div className="gpu-status-line">
                  <strong className="gpu-status-title">{isOnline ? 'GPU Available' : 'CPU Mode'}</strong>
                  <span className={`gpu-status-value${isCheckingStatus ? ' gpu-status-value--loading' : ''}`}>
                    {isCheckingStatus ? (
                      <>
                        <span className="gpu-status-pulse" aria-hidden="true" />
                        Checking GPU availability...
                      </>
                    ) : (
                      isOnline ? `${remaining} remaining` : 'GPU currently unavailable'
                    )}
                  </span>
                </div>

                <p className="gpu-status-meta mb-2">
                  <strong>GPU Window</strong>
                  <span>Daily 5 PM - 8 AM GMT · 24h on weekends</span>
                </p>

                <button
                  type="button"
                  className="gpu-status-benchmark-toggle mb-1"
                  onClick={() => setOpen((v) => !v)}
                  aria-expanded={open}
                >
                  <strong>Runtime Benchmark</strong>
                  {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>

                {open && (
                  <>
                    <p className="gpu-status-intro">
                      PLM embeddings are cached server-side. Once a sequence is computed, it is reused across future jobs.
                      Compute time scales mainly with unique proteins, not total rows.
                    </p>
                    <p className="gpu-status-conditions">
                      <strong>Conditions:</strong>
                      <span className="gpu-status-cond-value">1,000 reactions</span>
                      <span className="gpu-status-cond-sep">·</span>
                      <span className="gpu-status-cond-value">100 unique proteins</span>
                      <span className="gpu-status-cond-sep">·</span>
                      <span className="gpu-status-cond-value">avg. 400 aa</span>
                    </p>

                    <Table responsive size="sm" className="benchmark-table mb-0">
                      <colgroup>
                        <col className="benchmark-col-method" />
                        <col className="benchmark-col-gpu" />
                        <col className="benchmark-col-cpu" />
                        <col className="benchmark-col-cached" />
                      </colgroup>
                      <thead>
                        <tr>
                          <th rowSpan={2} className="benchmark-th-method">Method</th>
                          <th colSpan={2} className="benchmark-th-group">PLM embeddings not cached</th>
                          <th className="benchmark-th-group benchmark-th-cached">PLM embeddings cached</th>
                        </tr>
                        <tr>
                          <th className="benchmark-th-sub">GPU</th>
                          <th className="benchmark-th-sub">CPU</th>
                          <th className="benchmark-th-sub benchmark-th-sub-placeholder" aria-hidden="true">&nbsp;</th>
                        </tr>
                      </thead>
                      <tbody>
                        {BENCHMARK_DATA.map(({ method, uncachedGpu, uncachedCpu, cached, cachedTooltip, uncachedGpuTooltip }) => (
                          <tr key={method}>
                            <td className="benchmark-method">{method}</td>
                            <td
                              className={`benchmark-time ${uncachedGpu === 'N/A' ? 'benchmark-na' : uncachedGpu ? '' : 'benchmark-empty'}`}
                              title={uncachedGpuTooltip || undefined}
                            >
                              {uncachedGpu || '—'}
                            </td>
                            <td className="benchmark-time">{uncachedCpu ?? '—'}</td>
                            <td
                              className={`benchmark-time ${cached === 'N/A' ? 'benchmark-na' : cached ? '' : 'benchmark-empty'}`}
                              title={cachedTooltip || undefined}
                            >
                              {cached ?? '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </>
                )}
              </div>
            </Alert>
          </Col>
        </Row>
      </Container>
    </section>
  );
}
