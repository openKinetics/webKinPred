import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import { Button, Badge, OverlayTrigger, Popover } from 'react-bootstrap';

// Helper function to group items by their 'reason' for being invalid.
function groupByReason(items) {
  const map = new Map();
  for (const it of items || []) {
    const key = it?.reason || 'Unspecified';
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(it);
  }
  return Array.from(map.entries())
    .map(([reason, arr]) => ({
      reason,
      items: arr.sort((a, b) => (a.row ?? 0) - (b.row ?? 0)),
    }))
    .sort((a, b) => a.reason.localeCompare(b.reason));
}

// Helper component to render the offending content inside the popover.
function OffendingContent({ value }) {
  if (Array.isArray(value)) {
    return (
      <ul className="mb-0 ps-3">
        {value.map((v, i) => (
          <li key={i}>
            <code style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>{String(v)}</code>
          </li>
        ))}
      </ul>
    );
  }
  return (
    <code style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
      {value == null ? '' : String(value)}
    </code>
  );
}

export default function InvalidItems({ title, items }) {
  const groups = useMemo(() => groupByReason(items), [items]);
  const total = items?.length || 0;

  // Create a lookup map to get the full item details from a row number.
  const rowToItem = useMemo(() => {
    const m = new Map();
    for (const it of items || []) {
      if (!m.has(it.row)) m.set(it.row, it);
    }
    return m;
  }, [items]);

  return (
    <div className="invalid-list mb-4">
      {/* The title is now styled as a consistent tab section header */}
      <h5 className="tab-section-header">
        {title} <small className="text-secondary">({total} total)</small>
      </h5>

      {/* The main content is always visible since the Collapse wrapper was removed */}
      <div>
        {groups.map(({ reason, items }) => {
          const rows = items.map((it) => it.row);
          return (
            <div className="invalid-group mb-3" key={reason}>
              <div className="d-flex justify-content-between align-items-center mb-1">
                <div className="invalid-reason">
                  {reason} <span className="text-secondary">({rows.length})</span>
                </div>
                <Button
                  variant="outline-secondary"
                  size="sm"
                  onClick={async () => {
                    try {
                      // Note: Using the older execCommand for wider iFrame compatibility.
                      const textArea = document.createElement('textarea');
                      textArea.value = rows.join(', ');
                      document.body.appendChild(textArea);
                      textArea.select();
                      document.execCommand('copy');
                      document.body.removeChild(textArea);
                    } catch (err) {
                      console.error('Failed to copy rows:', err);
                    }
                  }}
                >
                  Copy rows
                </Button>
              </div>

              <div className="chips-scroll">
                {rows.map((r) => {
                  const it = rowToItem.get(r);
                  const value = it?.value;

                  const popover = (
                    <Popover id={`invalid-popover-${r}`} className="invalid-popover">
                      <Popover.Body>
                        <OffendingContent value={value} />
                      </Popover.Body>
                    </Popover>
                  );

                  return (
                    <OverlayTrigger
                      key={r}
                      trigger="click"
                      placement="top"
                      overlay={popover}
                      rootClose
                    >
                      <Badge bg="" className="chip-row me-2 mb-2" role="button" tabIndex={0}>
                        Row {r}
                      </Badge>
                    </OverlayTrigger>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

InvalidItems.propTypes = {
  title: PropTypes.string.isRequired,
  items: PropTypes.arrayOf(
    PropTypes.shape({
      row: PropTypes.number.isRequired,
      reason: PropTypes.string,
      value: PropTypes.oneOfType([PropTypes.string, PropTypes.array]),
    })
  ),
};