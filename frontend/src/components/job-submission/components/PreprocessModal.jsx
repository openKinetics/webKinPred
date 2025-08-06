import React from 'react';
import PropTypes from 'prop-types';
import { Modal, Button } from 'react-bootstrap';

export default function PreprocessModal({
  show,
  onHide,
  onRunValidation,
  isValidating,
}) {
  return (
    <Modal show={show} onHide={onHide} size="xl">
      <Modal.Header closeButton>
        <Modal.Title>Preprocess Before Prediction?</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p>Would you like to validate your input data before running predictions?</p>
        <p>
          This will identify invalid SMILES/InChIs and protein sequences, and perform sequence similarity checks
          (using mmseqs2) against the training datasets of the methods.
        </p>
        <p><strong>Note:</strong> Even if you skip this step, invalid rows will be automatically excluded during prediction and will not produce results.</p>
        <p className="fw-bold">Recommended if you’re unsure about input quality.</p>
      </Modal.Body>
      <Modal.Footer>
        <Button className="btn kave-btn-run-val" onClick={onHide} disabled={isValidating}>
          Cancel
        </Button>
        <Button className="btn kave-btn-run-val" onClick={onRunValidation} disabled={isValidating}>
          {isValidating ? 'Validating…' : 'Run Validation'}
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

PreprocessModal.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  onRunValidation: PropTypes.func.isRequired,
  isValidating: PropTypes.bool.isRequired,
};
