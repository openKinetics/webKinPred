// /home/saleh/webKinPred/frontend/src/components/job-submission/hooks/useJobSubmission.js
import { useState, useMemo, useRef, useEffect } from 'react';
import kcatMethodsByCsvType from '../constants/kcatMethodsByCsvType';
import {
  detectCsvFormat,
  validateCsv,
  fetchSequenceSimilaritySummary,
  submitJob as submitJobApi,
  openProgressStream,
  cancelValidationApi
} from '../services/api';

function makeSessionId() {
  // Simple UUID-ish
  return 'vs_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

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
  const [submissionResult, setSubmissionResult] = useState(null);

  // New: live log state
  const [validationSessionId, setValidationSessionId] = useState('');
  const userCancelledRef = useRef(false);
  const [liveLogs, setLiveLogs] = useState([]);
  const [streamConnected, setStreamConnected] = useState(false);
  const eventSourceRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Allowed methods derived from format
  const allowedKcatMethods = useMemo(() => {
    if (!csvFormatInfo?.csv_type) return [];
    return kcatMethodsByCsvType[csvFormatInfo.csv_type] || [];
  }, [csvFormatInfo]);

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

  const openStream = (sid) => {
    if (eventSourceRef.current) {
        try { eventSourceRef.current.close(); } catch {}
    }

    const es = openProgressStream(sid);
    eventSourceRef.current = es;
    setLiveLogs([]);
    setStreamConnected(true);

    es.onmessage = (evt) => {
        if (!evt?.data) return;
        setLiveLogs((prev) => [...prev, evt.data]);
    };

    es.onerror = (err) => {
        // Log any errors
        setStreamConnected(false);
    };
    };

  const closeStream = () => {
    if (eventSourceRef.current) {
      try { eventSourceRef.current.close(); } catch {}
      eventSourceRef.current = null;
    }
    setStreamConnected(false);
  };

  useEffect(() => {
    // Clean up on unmount
    return () => closeStream();
  }, []);

  const runValidation = async () => {
    if (!csvFile) return;
    const sid = makeSessionId();
    setValidationSessionId(sid);
    userCancelledRef.current = false;
    openStream(sid);
    setIsValidating(true);
    try {
        const validation = await validateCsv({
        file: csvFile,
        predictionType,
        kcatMethod,
        kmMethod,
        });
        const simPromise = fetchSequenceSimilaritySummary({ file: csvFile, useExperimental, validationSessionId: sid });
        const sim = await simPromise;
        if (userCancelledRef.current) return
        // Handle the results after both promises resolve
        setSimilarityData(sim);
        const { invalid_substrates, invalid_proteins, length_violations } = validation;
        setSubmissionResult({ invalid_substrates, invalid_proteins, length_violations });
        setShowValidationResults(true);
    } catch (err) {
      if (!userCancelledRef.current) {
        alert('Validation failed. Please try again. ' + (err?.message || ''));
      }
    } finally {
        setTimeout(() => closeStream(), 1000);
        setIsValidating(false);
    }
    };

  const cancelValidation = async () => {
    if (!validationSessionId) {
      // just close UI if nothing started
      setIsValidating(false);
      closeStream();
      return;
    }
    userCancelledRef.current = true;
    try {
      await cancelValidationApi(validationSessionId);
    } catch { /* swallow */ }
    setIsValidating(false);
    closeStream();
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

    // logs
    liveLogs,
    streamConnected,
    autoScroll,
    setAutoScroll,

    // actions
    onFileSelected,
    runValidation,
    submitJob,
    cancelValidation,
  };
}
