// src/components/FAQ.js
import React from 'react';
import Accordion from 'react-bootstrap/Accordion';

const FAQ = () => {
  const faqItems = [
    {
      question: "How to know which model to use?",
      answer: ""
    },
    {
      question: "How to make predictions for full reactions, not just protein-substrate pairs?",
        answer: "" // depending on method
    },
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
