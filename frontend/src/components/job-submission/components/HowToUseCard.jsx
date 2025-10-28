// src/components/HowToUseCard.js

import React from 'react';
import { Card, Row, Col, Alert, ListGroup, Button, Badge } from 'react-bootstrap';
import { BoxArrowInDown, Bullseye, CloudUpload, Cpu, Github } from 'react-bootstrap-icons';
import '../../../styles/components/HowToUseCard.css';
import methodDetails from '../constants/methodDetails';


export default function HowToUseCard() {
  return (
    <Card className="section-container how-to-use-card mb-4">
      <Card.Header as="h3" className="text-center">
        How to Use This Tool
      </Card.Header>
      <Card.Body>
        <p className="lead text-center mb-4">
          Predict kinetic parameters (k<sub>cat</sub> and/or K<sub>M</sub>) for enzyme-catalysed reactions using various machine learning models.
        </p>
        <Alert variant="info" className="d-flex align-items-center">
          <Bullseye size={24} className="me-3" />
          <div>
            Ticking <strong>“Prefer experimental data”</strong> will first search BRENDA, SABIO-RK, and UniProt for known values. If found, these are used instead of model predictions.
          </div>
        </Alert>

        <Row className="text-center">
          <Col md={4} className="step-col">
            <div className="step-icon"><Bullseye size={30} /></div>
            <h5>Step 1: Select Prediction</h5>
            <p>Choose what you want to predict: k<sub>cat</sub>, K<sub>M</sub>, or both.</p>
          </Col>
          <Col md={4} className="step-col">
            <div className="step-icon"><CloudUpload size={30} /></div>
            <h5>Step 2: Upload Data</h5>
            <p>Provide your reaction data by uploading a formatted CSV file.</p>
          </Col>
          <Col md={4} className="step-col">
            <div className="step-icon"><Cpu size={30} /></div>
            <h5>Step 3: Choose Method</h5>
            <p>Select your desired prediction model(s) after optional validation.</p>
          </Col>
        </Row>

        <hr className="my-4" />

        <h4 className="text-center mb-4">Input Data Format</h4>
        <Row>
          <Col md={6} className="mb-3">
            <Card className="h-100 format-card">
              <Card.Body>
                <Card.Title>Single-Substrate Models</Card.Title>
                <Card.Subtitle className="mb-2 text-white-70">
                  DLKcat, EITLEM, UniKP, KinForm-H, KinForm-L
                </Card.Subtitle>
                <ListGroup variant="flush">
                  <ListGroup.Item>
                    <span className="csv-col">Protein Sequence</span> — Full amino-acid sequence.
                  </ListGroup.Item>
                  <ListGroup.Item>
                    <span className="csv-col">Substrate</span> — One <code>SMILES</code> or <code>InChI</code> string.
                  </ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6} className="mb-3">
            <Card className="h-100 format-card">
              <Card.Body>
                <Card.Title>Multi-Substrate Model</Card.Title>
                <Card.Subtitle className="mb-2 text-white-70">TurNup</Card.Subtitle>
                <ListGroup variant="flush">
                  <ListGroup.Item>
                    <span className="csv-col">Protein Sequence</span> — Full amino-acid sequence.
                  </ListGroup.Item>
                  <ListGroup.Item>
                    <span className="csv-col">Substrates</span> — Semicolon-separated list of <code>SMILES</code> or <code>InChI</code>.
                  </ListGroup.Item>
                  <ListGroup.Item>
                    <span className="csv-col">Products</span> — Semicolon-separated list of <code>SMILES</code> or <code>InChI</code>.
                  </ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        <p className="text-center text-white-70">
            Multi-substrate CSVs can also be used for K<sub>M</sub> predictions. Each entry in the 'Substrates' column will receive its own K<sub>M</sub> value (separated by semicolons).
        </p>
        <hr className="my-4" />
        <h4 className="text-center mb-3">Example Templates</h4>
        <div className="d-grid gap-2 d-md-flex justify-content-md-center">
            <Button
            href="/templates/single_substrate_template.csv"
            download
            className="btn btn-custom-subtle"
            >
            <BoxArrowInDown className="me-2" />
            Single-Substrate Template
            </Button>

            <Button
            href="/templates/multi_substrate_template.csv"
            download
            className="btn btn-custom-subtle"
            >
            <BoxArrowInDown className="me-2" />
            Multi-Substrate Template
            </Button>
        </div>

        <hr className="my-4" />
        <h4 className="text-center mb-4">Available Prediction Methods</h4>
        <Row className="g-3">
          {Object.entries(methodDetails).map(([methodName, details]) => (
            <Col key={methodName} md={6} lg={4}>
              <Card className="method-card h-100">
                <Card.Body className="d-flex flex-column">
                  <div className="d-flex justify-content-between align-items-start mb-2">
                    <Card.Title className="method-title mb-0">{methodName}</Card.Title>
                    <a
                      href={details.repoUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="github-link"
                      title="View on GitHub"
                    >
                      <Github size={20} />
                    </a>
                  </div>
                  <Card.Text className="method-description flex-grow-1">
                    {details.description}
                  </Card.Text>
                  <div className="method-publication">
                    <a
                      href={details.citationUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="publication-link"
                    >
                      <small className="publication-title">{details.publicationTitle}</small>
                    </a>
                    <small className="text-muted d-block mt-1">{details.authors}</small>
                  </div>
                  {details.moreInfo && (
                    <div className="mt-2">
                      <Badge bg="secondary" className="more-info-badge">
                        {details.moreInfo}
                      </Badge>
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
      </Card.Body>
    </Card>
  );
}