import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';

import useJobSubmission from './hooks/useJobSubmission';
import HowToUseCard from './components/HowToUseCard';
import PredictionTypeSelect from './components/PredictionTypeSelect';
import CsvUpload from './components/CsvUpload';
import MethodPicker from './components/MethodPicker';
import ValidationResults from './components/ValidationResults';
import PreprocessModal from './components/PreprocessModal';
import SubmissionResultModal from './components/SubmissionResultModal';

import './JobSubmissionForm.css';

export default function JobSubmissionForm() {
  const state = useJobSubmission();

  return (
    <Container className="mt-5 pb-5">
      <Row className="justify-content-center">
        <Col md={10}>
          <HowToUseCard />

          <PredictionTypeSelect
            value={state.predictionType}
            onChange={state.setPredictionType}
          />

          {state.predictionType && (
            <CsvUpload
              csvFormatValid={state.csvFormatValid}
              csvFormatInfo={state.csvFormatInfo}
              csvFormatError={state.csvFormatError}
              onFileSelected={state.onFileSelected}
              onClickValidate={() => state.setShowPreprocessPrompt(true)}
              fileName={state.fileName}
            />
          )}

          {state.submissionResult && (
            <ValidationResults
              submissionResult={state.submissionResult}
              showValidationResults={state.showValidationResults}
              setShowValidationResults={state.setShowValidationResults}
              handleLongSeqs={state.handleLongSeqs}
              setHandleLongSeqs={state.setHandleLongSeqs}
              similarityData={state.similarityData}
            />
          )}

          {state.predictionType && state.csvFile && state.csvFormatValid && (
            <MethodPicker
              predictionType={state.predictionType}
              allowedKcatMethods={state.allowedKcatMethods}
              kcatMethod={state.kcatMethod}
              setKcatMethod={state.setKcatMethod}
              kmMethod={state.kmMethod}
              setKmMethod={state.setKmMethod}
              csvFormatInfo={state.csvFormatInfo}
              useExperimental={state.useExperimental}
              setUseExperimental={state.setUseExperimental}
              onSubmit={state.submitJob}
              isSubmitting={state.isSubmitting}
            />
          )}

          {/* Modals */}
          <PreprocessModal
            show={state.showPreprocessPrompt}
            onHide={() => state.setShowPreprocessPrompt(false)}
            onRunValidation={async () => {
              state.setShowPreprocessPrompt(false);
              await state.runValidation();
            }}
            isValidating={state.isValidating}
          />

          <SubmissionResultModal
            show={state.showModal}
            onHide={() => state.setShowModal(false)}
            message={state.submissionResult?.message}
            publicId={state.submissionResult?.public_id}
          />

          {/* Full-screen validating overlay */}
          {state.isValidating && (
            <div style={{
              position: 'fixed',
              top: 0, left: 0, right: 0, bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.85)',
              zIndex: 9999,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              color: 'white',
              fontSize: '1.5rem',
              flexDirection: 'column',
            }}>
              <div className="spinner-border text-light mb-3" role="status">
                <span className="visually-hidden">Loading…</span>
              </div>
              <div>Validating Inputs and Running MMseqs2…</div>
            </div>
          )}
        </Col>
      </Row>
    </Container>
  );
}
