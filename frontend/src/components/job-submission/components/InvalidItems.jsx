import React, { useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { Button, Badge, Collapse, OverlayTrigger, Popover } from 'react-bootstrap';

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
  const [open, setOpen] = useState(true);
  const groups = useMemo(() => groupByReason(items), [items]);
  const total = items?.length || 0;

  // Row -> full item lookup (so we can get both row and offending value)
  const rowToItem = useMemo(() => {
    const m = new Map();
    for (const it of items || []) {
      if (!m.has(it.row)) m.set(it.row, it);
    }
    return m;
  }, [items]);

  return (
    <div className="invalid-list mb-4">
      <div className="d-flex justify-content-between align-items-center">
        <h5 className="mb-2">
          {title} <small className="text-secondary">({total})</small>
        </h5>
        <Button variant="outline-secondary" size="sm" onClick={() => setOpen((v) => !v)}>
          {open ? 'Hide' : 'Show'}
        </Button>
      </div>

      <Collapse in={open}>
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
                        await navigator.clipboard.writeText(rows.join(', '));
                      } catch {}
                    }}
                  >
                    Copy rows
                  </Button>
                </div>

                <div className="chips-scroll">
                  {rows.map((r) => {
                    const it = rowToItem.get(r);
                    const value = it?.value;

                    const pop = (
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
                        overlay={pop}
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
      </Collapse>
    </div>
  );
}

InvalidItems.propTypes = {
  title: PropTypes.string.isRequired,
  items: PropTypes.arrayOf(
    PropTypes.shape({
      row: PropTypes.number.isRequired,
      reason: PropTypes.string,
      value: PropTypes.oneOfType([PropTypes.string, PropTypes.array]), // string or array of strings
    })
  ),
};
