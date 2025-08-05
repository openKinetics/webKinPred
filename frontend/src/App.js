// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import JobSubmissionForm from './components/job-submission/JobSubmissionForm';
import JobStatus from './components/JobStatus';
import About from './components/About';
import Evaluation from './components/Evaluation';
import Header from './components/Header';
import ProteinBackground from './components/ProteinBackground';
import Footer from './components/Footer';
import './App.css'; // Ensure this import is present

function App() {
  return (
    <Router>
      <ProteinBackground />
      <div className="app-container">
        <Header />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<JobSubmissionForm />} />
            <Route path="/track-job/:public_id" element={<JobStatus />} />
            <Route path="/track-job" element={<JobStatus />} />
            <Route path="/about" element={<About />} />
            <Route path="/evaluation" element={<Evaluation />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
