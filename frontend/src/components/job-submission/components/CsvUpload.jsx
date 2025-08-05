import React, { useRef, useState } from 'react';
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
const fileRef = useRef(null);
const [fileInputKey, setFileInputKey] = useState(0);
const hasFile = !!fileName && fileName !== 'No file chosen';
  return (
    <Card className="section-container section-reaction-info mb-4">
      <Card.Body>
        <h3>Upload Reaction Information</h3>
        <p>Please upload a CSV file with the columns mentioned above.</p>
        <Form>
          <Form.Group controlId="csvFile" className="mt-3">
            <div className="file-upload">
              <Form.Control
              key={fileInputKey}
              ref={fileRef}
              id="csvFile"
              type="file"
              accept=".csv"
              onChange={(e) => onFileSelected(e.target.files?.[0] || null)}
              style={{ display: 'none' }}
              required
              />
                <button
                    type="button"
                    className={`btn kave-btn-upload ${hasFile ? 'has-file' : ''}`}
                    onClick={() => {
                    if (fileRef.current) fileRef.current.value = ''; // allow same-file selection
                    fileRef.current?.click();
                    }}
                    aria-label={hasFile ? 'Change CSV file' : 'Choose CSV file'}
                >
                {/* Icon changes when a file is present */}
                {hasFile ? (
                  // check icon
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                       viewBox="0 0 16 16" aria-hidden="true" style={{ marginRight: 8 }}>
                    <path fill="currentColor" d="M13.485 1.929 6.5 8.914 4.222 6.636l-1.414 1.414L6.5 11.742l8.4-8.4-1.415-1.414z"/>
                  </svg>
                ) : (
                  // upload icon
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                       viewBox="0 0 16 16" aria-hidden="true" style={{ marginRight: 8 }}>
                    <path fill="currentColor" d="M.5 13a2.5 2.5 0 0 0 2.5 2.5h10A2.5 2.5 0 0 0 15.5 13v-3h-1v3a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 1.5 13v-3h-1V13zM8.5 8.207V1.5h-1v6.707L5.354 5.061l-.708.707L8 9.121l3.354-3.353-.708-.707L8.5 8.207z"/>
                  </svg>
                )}
                {hasFile ? 'Change CSV' : 'Choose CSV'}
              </button>

              <span
                className={`file-selected ${hasFile ? 'has-file' : ''}`}
                aria-live="polite"
              >
                {fileName}
              </span>
  
              {hasFile && (
              <button
                  type="button"
                  className="btn kave-btn-upload-clear"
                  onClick={() => {
                  if (fileRef.current) fileRef.current.value = '';
                  setFileInputKey((k) => k + 1);          // force remount
                  onFileSelected(null);                    // your hook will set 'No file chosen'
                  }}
                  aria-label="Clear selected file"
              >
                  Clear
              </button>
              )}
            </div>
          </Form.Group>
        </Form>

        {csvFormatValid && csvFormatInfo?.csv_type && (
          <Alert variant="success" className="mt-3">
          Detected a <strong>{csvFormatInfo.csv_type === 'multi' ? 'multi-substrate' : 'single-substrate'}</strong> CSV with {csvFormatInfo.num_rows} rows. You may now choose compatible methods.
          </Alert>
        )}
        {!csvFormatValid && csvFormatError && (
          <Alert variant="danger" className="mt-3">
            ❌ Invalid CSV: {csvFormatError}
          </Alert>
        )}

        {csvFormatValid && (
          <div className="mt-4 d-flex justify-content-end">
            <Button
              className="kave-btn kave-btn-secondary"
              onClick={onClickValidate}
            >
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
