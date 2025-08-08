import React from 'react';
import PropTypes from 'prop-types';
import { Form, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { InfoCircleFill } from 'react-bootstrap-icons';

const ExpTooltip = (
  <Tooltip id="exp-tooltip" className="exp-tooltip">
    Prefer curated values from BRENDA, SABIO-RK and UniProt. When a reliable match is found,
    that value is returned instead of a model prediction.
  </Tooltip>
);

export default function ExperimentalSwitch({ checked, onChange }) {
  return (
    <Form.Group controlId="useExperimental" className="exp-switch-group">
      <div className="d-flex align-items-center gap-2">
        <Form.Check
          type="switch"
          label="Prefer experimental data"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="exp-switch"
        />

        <OverlayTrigger placement="right" overlay={ExpTooltip} delay={{ show: 150, hide: 0 }} trigger={['hover', 'focus']}>
          <button
            type="button"
            className="exp-info-btn"
            aria-label="What does ‘Prefer experimental data’ do?"
          >
            <InfoCircleFill size={16} aria-hidden="true" />
          </button>
        </OverlayTrigger>
      </div>
    </Form.Group>
  );
}

ExperimentalSwitch.propTypes = {
  checked: PropTypes.bool.isRequired,
  onChange: PropTypes.func.isRequired,
};
