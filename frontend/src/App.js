// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import JobSubmissionForm from './components/JobSubmissionForm';
import JobStatus from './components/JobStatus';
import About from './components/About';
import FAQ from './components/FAQ';
import Evaluation from './components/Evaluation';
import Header from './components/Header';
import Footer from './components/Footer';
import './App.css'; // Ensure this import is present

function App() {
  return (
    <Router>
      <div className="app-container">
        <Header />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<JobSubmissionForm />} />
            <Route path="/track-job/:jobId" element={<JobStatus />} />
            <Route path="/track-job" element={<JobStatus />} />
+           <Route path="/about" element={<About />} />
+           <Route path="/faq" element={<FAQ />} />
+           <Route path="/evaluation" element={<Evaluation />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
