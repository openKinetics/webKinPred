// App.js
import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import JobSubmissionForm from './components/job-submission/JobSubmissionForm';
import JobStatus from './components/JobStatus';
import About from './components/About';
import ApiDocs from './components/ApiDocs/ApiDocs';
import Contribute from './components/Contribute';
import NotFound from './components/NotFound';
import Header from './components/Header';
import GpuStatus from './components/GpuStatus';
import ProteinBackground from './components/ProteinBackground';
import Footer from './components/Footer';
import apiClient from './components/appClient';
import './styles/App.css';

function MaintenanceNotice() {
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => apiClient.get('/health/', {
      signal: controller.signal,
      timeout: 10000,
    }).then(({ data }) => {
      setPaused(data.submissions_paused === true);
    }).catch(() => {});
    refresh();
    const timer = setInterval(refresh, 30000);
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, []);

  return paused ? (
    <div className="alert alert-warning" role="status">
      <strong>Maintenance in progress.</strong> New submissions are paused.
      {' '}Existing jobs are saved, and status pages and completed downloads remain available.
    </div>
  ) : null;
}

function AppContent() {
  const location = useLocation();
  const isTrackPage = location.pathname.startsWith('/track-job');
  const showGpuStatus = location.pathname === '/' || location.pathname.startsWith('/track-job');
  const gpuLayout = isTrackPage ? 'track' : 'home';

  return (
    <>
      <ProteinBackground />
      <div className={`app-container${isTrackPage ? ' app-container--track' : ''}`}>
        <Header />
        <main className={`main-content${isTrackPage ? ' main-content--track' : ''}`}>
          <MaintenanceNotice />
          {showGpuStatus && <GpuStatus layout={gpuLayout} />}
          <Routes>
            <Route path="/" element={<JobSubmissionForm />} />
            <Route path="/track-job/:public_id" element={<JobStatus />} />
            <Route path="/track-job" element={<JobStatus />} />
            <Route path="/about" element={<About />} />
            <Route path="/api-docs" element={<ApiDocs />} />
            <Route path="/contribute" element={<Contribute />} />
            <Route path="*" element={<NotFound />} />
            {/* <Route path="/evaluation" element={<Evaluation />} /> */}
          </Routes>
        </main>
        <Footer />
      </div>
    </>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}

export default App;
