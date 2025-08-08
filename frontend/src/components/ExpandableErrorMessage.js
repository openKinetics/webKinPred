import React, { useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { Alert, Button } from 'react-bootstrap';
import {
  ChevronDown,
  ChevronUp,
  Clipboard,
  ClipboardCheck,
  FileEarmarkText
} from 'react-bootstrap-icons';
import '../styles/components/ExpandableErrorMessage.css';

export default function ExpandableErrorMessage({
  errorMessage,
  title = 'Skipped Rows',
  collapsedLines = 6,
  showFileIcon = true,
}) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const normalised = useMemo(() => (typeof errorMessage === 'string'
    ? errorMessage.trimEnd()
    : JSON.stringify(errorMessage, null, 2)), [errorMessage]);

  const totalLines = useMemo(() => normalised.split(/\r?\n/).length, [normalised]);
  const isCollapsible = totalLines > collapsedLines;

  const preview = useMemo(() => {
    if (!isCollapsible) return normalised;
    return normalised.split(/\r?\n/).slice(0, collapsedLines).join('\n') + '\n…';
  }, [normalised, isCollapsible, collapsedLines]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(normalised);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // ignore
    }
  };

  return (
    <Alert variant="dark" className="expmsg-card section-container p-0">
      <div className="expmsg-header d-flex align-items-center justify-content-between">
        <div className="d-flex align-items-center gap-2">
          {showFileIcon && <FileEarmarkText size={18} className="text-primary" aria-hidden="true" />}
          <span className="expmsg-title">{title}</span>
        </div>

        <div className="d-flex align-items-center gap-2">
          <Button
            size="sm"
            variant="outline-light"
            className="expmsg-btn"
            onClick={handleCopy}
            aria-label="Copy details to clipboard"
          >
            {copied ? <ClipboardCheck size={16} /> : <Clipboard size={16} />}
          </Button>

          {isCollapsible && (
            <Button
              size="sm"
              className="btn btn-custom-subtle expmsg-toggle"
              onClick={() => setExpanded((s) => !s)}
              aria-expanded={expanded}
              aria-controls="expmsg-content"
            >
              {expanded ? <ChevronUp className="me-1" /> : <ChevronDown className="me-1" />}
              {expanded ? 'Collapse' : 'Expand'}
            </Button>
          )}
        </div>
      </div>

      <div
        id="expmsg-content"
        className={`expmsg-body ${expanded || !isCollapsible ? 'is-open' : 'is-closed'}`}
        role="region"
        aria-label="Error details"
      >
        <pre className="expmsg-pre">
{expanded || !isCollapsible ? normalised : preview}
        </pre>
      </div>
    </Alert>
  );
}

ExpandableErrorMessage.propTypes = {
  errorMessage: PropTypes.oneOfType([PropTypes.string, PropTypes.object]).isRequired,
  title: PropTypes.string,
  collapsedLines: PropTypes.number,
  showFileIcon: PropTypes.bool,
};
