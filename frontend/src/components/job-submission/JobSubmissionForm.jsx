// /home/saleh/webKinPred/frontend/src/components/job-submission/JobSubmissionForm.jsx
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
import LiveLogOverlay from './components/LiveLogOverlay';

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
        </Col>
      </Row>

      {/* New Live Progress Overlay */}
      <LiveLogOverlay
        show={state.isValidating}
        logs={state.liveLogs}
        connected={state.streamConnected}
        autoScroll={state.autoScroll}
        setAutoScroll={state.setAutoScroll}
        onCancel={state.cancelValidation}
      />
    </Container>
  );
}
