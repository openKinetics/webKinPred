import React from 'react';
import PropTypes from 'prop-types';
import { Form, OverlayTrigger, Tooltip } from 'react-bootstrap';

const renderTip = (props) => (
  <Tooltip id="exp-tip" className="exp-tip" {...props}>
    If ticked, we look up any matching k<sub>cat</sub> or K<sub>M</sub> measurements in
    BRENDA, SABIO-RK or UniProt and use them instead of model predictions.
  </Tooltip>
);

export default function ExperimentalSwitch({ checked, onChange }) {
  return (
    <Form.Group controlId="useExperimental" className="mt-3">
      <div className="exp-switch-wrapper me-3">
        <Form.Check
          id="useExperimental"
          type="switch"
          label="Prefer experimental data"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="exp-switch"
        />
        <OverlayTrigger placement="right" overlay={renderTip}>
          <span className="info-icon ms-1" role="button" tabIndex={0}>i</span>
        </OverlayTrigger>
      </div>
    </Form.Group>
  );
}

ExperimentalSwitch.propTypes = {
  checked: PropTypes.bool.isRequired,
  onChange: PropTypes.func.isRequired,
};
