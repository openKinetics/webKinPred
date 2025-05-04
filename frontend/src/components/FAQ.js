// src/components/FAQ.js
import React from 'react';
import Accordion from 'react-bootstrap/Accordion';

const FAQ = () => {
  const faqItems = [
    {
      question: "What is KineticXPredictor?",
      answer: "KineticXPredictor is an application designed to predict kinetics performance based on various input parameters."
    },
    {
      question: "How do I submit a job?",
      answer: "Jobs can be submitted from the homepage using the provided submission form."
    },
    {
      question: "How can I track my job status?",
      answer: "You can track your submitted jobs on the 'Track Job' page."
    }
    // Add more FAQ items as needed
  ];

  return (
    <div className="container mt-4">
      <h2>Frequently Asked Questions</h2>
      <Accordion defaultActiveKey="0">
        {faqItems.map((item, index) => (
          <Accordion.Item eventKey={index.toString()} key={index}>
            <Accordion.Header>{item.question}</Accordion.Header>
            <Accordion.Body>{item.answer}</Accordion.Body>
          </Accordion.Item>
        ))}
      </Accordion>
    </div>
  );
};

export default FAQ;
