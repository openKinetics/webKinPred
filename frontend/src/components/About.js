// src/components/About.js
import React, { useState } from 'react';
import { Button, Form } from 'react-bootstrap';

const About = () => {
  const [copied, setCopied] = useState(false);
  const citationText = "KineticXPredictor: ....";

  const copyCitation = () => {
    navigator.clipboard.writeText(citationText)
      .then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
      })
      .catch(err => console.error('Failed to copy: ', err));
  };

  return (
    <div className="container mt-4">
      <h2>About KineticXPredictor</h2>
      <section>
        <p>
          {/* Add your detailed description here */}
        </p>
      </section>
      <section className="mt-4">
        <h4>Citation</h4>
        <Form.Group controlId="citationText">
          <Form.Control as="textarea" rows={3} readOnly value={citationText} />
        </Form.Group>
        <Button variant="secondary" className="mt-2" onClick={copyCitation}>
          {copied ? "Copied!" : "Copy Citation"}
        </Button>
      </section>
    </div>
  );
};

export default About;
