import React, { useEffect, useRef } from 'react';
import './LiveLogOverlay.css';

export default function LiveLogOverlay({
  show,
  logs = [],
  connected = false,
  autoScroll = true,
  setAutoScroll = () => {},
  onClose = null,
  onCancel = null,      // NEW
  title = 'Validating Inputs and Running MMseqs2…',
}) {
  const panelRef = useRef(null);

  useEffect(() => {
    if (!autoScroll) return;
    const el = panelRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs, autoScroll]);

  const copyAll = async () => {
    try { await navigator.clipboard.writeText(logs.join('\n')); } catch {}
  };

  if (!show) return null;

  return (
    <div className="wkp-overlay">
      <div className="wkp-terminal-card">
        <div className="wkp-header">
          <div className="wkp-lights">
            {/* Red button doubles as "cancel & close" */}
            <button
              type="button"
              className="wkp-light red wkp-light-btn"
              aria-label="Cancel validation"
              title="Cancel validation"
              onClick={onCancel || onClose}
            />
          </div>

          <div className="wkp-title">{title}</div>

          <div className="wkp-actions">
            <span className={`wkp-pill ${connected ? 'ok' : 'warn'}`}>
              {connected ? 'Live' : 'Reconnecting…'}
            </span>
            <button className="wkp-btn" onClick={() => setAutoScroll(!autoScroll)}>
              {autoScroll ? 'Pause Scroll' : 'Auto-scroll'}
            </button>
            <button className="wkp-btn" onClick={copyAll}>Copy</button>
            {onClose && (
              <button className="wkp-btn wkp-btn-ghost" onClick={onClose}>Hide</button>
            )}
          </div>
        </div>

        <div className="wkp-progress">
          <div className="wkp-progress-bar" />
        </div>

        <div className="wkp-terminal" ref={panelRef} aria-live="polite">
          {logs.length === 0 ? (
            <div className="wkp-empty">
              <div className="spinner-border" role="status" />
              <div>Connecting to live logs…</div>
            </div>
          ) : (
            logs.map((line, idx) => (
              <pre className="wkp-line" key={idx}>
                <span className="wkp-lineno">{idx + 1}</span>
                <code className="wkp-code">{line}</code>
              </pre>
            ))
          )}
        </div>

        <div className="wkp-footer">
          <div className="wkp-note">
            Please keep this tab open. This may take a few minutes for large inputs.
          </div>
        </div>
      </div>
    </div>
  );
}
