import React from 'react';
import PropTypes from 'prop-types';
import { Modal, Button } from 'react-bootstrap';

export default function SubmissionResultModal({ show, onHide, message, publicId }) {
  return (
    <Modal
      show={show}
      onHide={onHide}
      backdrop="static" // Prevents closing on backdrop click
      keyboard={false}   // Prevents closing with the Escape key
    >
      <Modal.Header closeButton>
        <Modal.Title>Job Successfully Submitted</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <h5>{message}</h5>
        <p>Job ID: {publicId}</p>
      </Modal.Body>
      <Modal.Footer>
        <Button className='btn kave-btn-run-val' onClick={onHide}>
          Close
        </Button>
        <Button
          className='btn kave-btn-run-val'
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