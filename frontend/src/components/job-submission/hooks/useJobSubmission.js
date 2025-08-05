import { useState, useMemo } from 'react';
import kcatMethodsByCsvType from '../constants/kcatMethodsByCsvType';
import {
  detectCsvFormat,
  validateCsv,
  fetchSequenceSimilaritySummary,
  submitJob as submitJobApi,
} from '../services/api';

export default function useJobSubmission() {
  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [showPreprocessPrompt, setShowPreprocessPrompt] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [showValidationResults, setShowValidationResults] = useState(false);

  // Domain state
  const [predictionType, setPredictionType] = useState('');
  const [kcatMethod, setKcatMethod] = useState('');
  const [kmMethod, setKmMethod] = useState('');
  const [csvFile, setCsvFile] = useState(null);
  const [fileName, setFileName] = useState('No file chosen');
  const [useExperimental, setUseExperimental] = useState(false);
  const [handleLongSeqs, setHandleLongSeqs] = useState('truncate');

  // Derived server feedback
  const [csvFormatInfo, setCsvFormatInfo] = useState(null);
  const [csvFormatValid, setCsvFormatValid] = useState(false);
  const [csvFormatError, setCsvFormatError] = useState('');
  const [similarityData, setSimilarityData] = useState(null);
  const [submissionResult, setSubmissionResult] = useState(null); // { message, public_id, invalid_* , length_violations }

  // Allowed methods derived from format
  const allowedKcatMethods = useMemo(() => {
    if (!csvFormatInfo?.csv_type) return [];
    return kcatMethodsByCsvType[csvFormatInfo.csv_type] || [];
  }, [csvFormatInfo]);

  // Handlers
  const resetMethods = () => {
    setKcatMethod('');
    setKmMethod('');
  };

  const onChangePredictionType = (val) => {
    setPredictionType(val);
    resetMethods();
    setSimilarityData(null);
    setSubmissionResult(null);
    setShowValidationResults(false);
  };

  const onFileSelected = async (file) => {
    setCsvFile(file);
    setFileName(file?.name || 'No file chosen');
    resetMethods();
    setSimilarityData(null);
    setSubmissionResult(null);
    setShowValidationResults(false);

    if (!file) {
      setCsvFormatInfo(null);
      setCsvFormatValid(false);
      setCsvFormatError('');
      return;
    }

    try {
      const data = await detectCsvFormat(file);
      if (data.status === 'valid') {
        setCsvFormatInfo(data);
        setCsvFormatValid(true);
        setCsvFormatError('');
      } else {
        setCsvFormatInfo(null);
        setCsvFormatValid(false);
        setCsvFormatError(Array.isArray(data.errors) ? data.errors.join('; ') : 'Invalid CSV format.');
      }
    } catch (err) {
      setCsvFormatInfo(null);
      setCsvFormatValid(false);
      setCsvFormatError(err?.response?.data?.error || 'Error detecting CSV format.');
    }
  };

  const runValidation = async () => {
    if (!csvFile) return;
    setIsValidating(true);
    try {
      const validation = await validateCsv({
        file: csvFile,
        predictionType,
        kcatMethod,
        kmMethod,
      });
      const sim = await fetchSequenceSimilaritySummary({
        file: csvFile,
        useExperimental,
      });
      setSimilarityData(sim);
      const { invalid_substrates, invalid_proteins, length_violations } = validation;
      setSubmissionResult({ invalid_substrates, invalid_proteins, length_violations });
      setShowValidationResults(true);
    } catch (err) {
      // Surface for now; in production, prefer a toast system
      alert('Validation failed. Please try again. ' + (err?.message || ''));
      // leave state visible for debugging
    } finally {
      setIsValidating(false);
    }
  };

  const submitJob = async () => {
    if (!csvFile) return;
    setIsSubmitting(true);
    try {
      const data = await submitJobApi({
        predictionType,
        kcatMethod,
        kmMethod,
        file: csvFile,
        handleLongSequences: handleLongSeqs,
        useExperimental,
      });
      setSubmissionResult((prev) => ({
        ...prev,
        message: data.message,
        public_id: data.public_id,
      }));
      setShowModal(true);
    } catch (error) {
      const msg = error?.response?.data?.error || 'Failed to submit job';
      alert('Failed to submit job\n' + msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return {
    // state
    isSubmitting,
    isValidating,
    showPreprocessPrompt,
    setShowPreprocessPrompt,
    showModal,
    setShowModal,
    showValidationResults,
    setShowValidationResults,

    predictionType,
    setPredictionType: onChangePredictionType,
    kcatMethod,
    setKcatMethod,
    kmMethod,
    setKmMethod,
    csvFile,
    fileName,
    csvFormatInfo,
    csvFormatValid,
    csvFormatError,
    useExperimental,
    setUseExperimental,
    handleLongSeqs,
    setHandleLongSeqs,
    similarityData,
    submissionResult,

    allowedKcatMethods,

    // actions
    onFileSelected,
    runValidation,
    submitJob,
  };
}
