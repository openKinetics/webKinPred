import React from 'react';
import PropTypes from 'prop-types';
import { Card, Form, Alert, Button } from 'react-bootstrap';

export default function CsvUpload({
  csvFormatValid,
  csvFormatInfo,
  csvFormatError,
  onFileSelected,
  onClickValidate,
  fileName,
}) {
  return (
    <Card className="section-container section-reaction-info">
      <Card.Body>
        <h3>Upload Reaction Information</h3>
        <p>Please upload a CSV file with the columns mentioned above.</p>
        <Form>
          <Form.Group controlId="csvFile" className="mt-3">
            <div className="file-upload">
              <Form.Control
                type="file"
                accept=".csv"
                onChange={(e) => onFileSelected(e.target.files[0])}
                style={{ display: 'none' }}
                required
              />
              <label htmlFor="csvFile" className="custom-file-upload">Choose File</label>
              <span id="file-selected">{fileName}</span>
            </div>
          </Form.Group>
        </Form>

        {csvFormatValid && csvFormatInfo?.csv_type && (
          <Alert variant="success" className="mt-3">
            ✅ Detected a <strong>{csvFormatInfo.csv_type === 'multi' ? 'multi-substrate' : 'single-substrate'}</strong> CSV. You may now choose compatible methods.
          </Alert>
        )}
        {!csvFormatValid && csvFormatError && (
          <Alert variant="danger" className="mt-3">
            ❌ Invalid CSV: {csvFormatError}
          </Alert>
        )}

        {csvFormatValid && (
          <div className="mt-4 d-flex justify-content-end">
            <Button variant="secondary" onClick={onClickValidate}>
              Validate Inputs (Optional)
            </Button>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}

CsvUpload.propTypes = {
  csvFormatValid: PropTypes.bool.isRequired,
  csvFormatInfo: PropTypes.object,
  csvFormatError: PropTypes.string,
  onFileSelected: PropTypes.func.isRequired,
  onClickValidate: PropTypes.func.isRequired,
  fileName: PropTypes.string.isRequired,
};
