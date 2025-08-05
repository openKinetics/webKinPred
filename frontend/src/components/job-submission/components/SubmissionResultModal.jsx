import React from 'react';
import PropTypes from 'prop-types';
import { Modal, Button } from 'react-bootstrap';

export default function SubmissionResultModal({ show, onHide, message, publicId }) {
  return (
    <Modal show={show} onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>Job Successfully Submitted</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h5>{message}</h5>
        <p>Job ID: {publicId}</p>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
        <Button
          variant="primary"
          onClick={() => {
            onHide();
            window.location.href = `/track-job/${publicId}`;
          }}
        >
          Track Job
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

SubmissionResultModal.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  message: PropTypes.string,
  publicId: PropTypes.string,
};
