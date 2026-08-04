// src/components/HowToUseCard.js

import { useState } from 'react';
import { Card, Row, Col, Alert, Button } from 'react-bootstrap';
import { BoxArrowInDown, Bullseye, CloudUpload, Cpu, Github, ChevronDown, BoxArrowUpRight, ExclamationTriangle } from 'react-bootstrap-icons';
import { Link } from 'react-router-dom';
import '../../../styles/components/HowToUseCard.css';

const FORMAT_SPECS = [
  {
    key: 'single',
    name: 'Single Substrate',
    methods: ['CataPro', 'CatPred (Km)', 'CatRange', 'DLKcat', 'EITLEM', 'IECata', 'KinForm-H', 'KinForm-L', 'MMISA-KM', 'OmniESI', 'UniKP'],
    columns: [
      { col: 'Protein Sequence', desc: 'full amino-acid sequence' },
      { col: 'Substrate',        desc: 'SMILES or InChI — one per row' },
    ],
  },
  {
    key: 'multi',
    name: 'Multi-Substrate',
    methods: ['CatPred (kcat)'],
    columns: [
      { col: 'Protein Sequence', desc: 'full amino-acid sequence' },
      { col: 'Substrates',        desc: 'semicolon-separated SMILES or InChI' },
    ],
  },
  {
    key: 'full',
    name: 'Full Reaction',
    methods: ['TurNup (kcat)'],
    columns: [
      { col: 'Protein Sequence', desc: 'full amino-acid sequence' },
      { col: 'Substrates',       desc: 'semicolon-separated SMILES or InChI' },
      { col: 'Products',         desc: 'semicolon-separated SMILES or InChI' },
    ],
  },
];


export default function HowToUseCard({ methods = {} }) {
  const [openKeys, setOpenKeys] = useState(new Set());

  const toggle = (key) => setOpenKeys(prev => {
    const next = new Set(prev);
    next.has(key) ? next.delete(key) : next.add(key);
    return next;
  });

  const methodEntries = Object.entries(methods).sort(([, a], [, b]) =>
    (a.displayName || '').localeCompare(b.displayName || '')
  );
  const cols = [[], [], []];
  methodEntries.forEach((entry, i) => cols[i % 3].push(entry));

  const targetLabel = {
    kcat: 'kcat',
    Km: 'Km',
    'kcat/Km': 'kcat/Km',
  };

  return (
    <Card className="section-container how-to-use-card mb-4">
      <Card.Header as="h3" className="text-center">
        How to Use This Tool
      </Card.Header>
      <Card.Body>
        <p className="lead text-center mb-4">
          Predict kinetic parameters (k<sub>cat</sub>, K<sub>M</sub>, and k<sub>cat</sub>/K<sub>M</sub>) for enzyme-catalysed reactions using various machine learning models.
        </p>
        <Alert variant="warning" className="d-flex align-items-center howto-limit-alert">
          <ExclamationTriangle size={24} className="me-3 howto-limit-icon" />
          <div>
            Default usage is limited to <strong>20,000 reactions per day</strong>. If you need a higher limit, please{' '}
            <Link to="/about#contact" className="howto-limit-link">contact us</Link>.
          </div>
        </Alert>
        <Row className="text-center">
          <Col md={4} className="step-col">
            <div className="step-icon"><Bullseye size={30} /></div>
            <h5>Step 1: Select Prediction</h5>
            <p>Choose one or more targets: k<sub>cat</sub>, K<sub>M</sub>, and/or k<sub>cat</sub>/K<sub>M</sub>.</p>
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

        <h4 className="text-center mb-3">Available Predictors</h4>
        <div className="mpill-grid">
          {cols.map((col, ci) => (
            <div key={ci} className="mpill-col">
              {col.map(([key, details]) => {
                const isOpen = openKeys.has(key);
                return (
                  <div key={key} className={`mpill${isOpen ? ' mpill--open' : ''}`}>
                    <button
                      type="button"
                      className="mpill-header"
                      onClick={() => toggle(key)}
                      aria-expanded={isOpen}
                    >
                      <span className="mpill-name">{details.displayName}</span>
                      <div className="mpill-right">
                        <div className="mpill-chips">
                          {(details.supports || []).map((target) => (
                            <span key={target} className="mpill-chip">
                              {targetLabel[target] || target}
                            </span>
                          ))}
                        </div>
                        <ChevronDown size={13} className="mpill-chevron" />
                      </div>
                    </button>

                    <div className="mpill-body">
                      <div className="mpill-body-inner">
                        <div className="mpill-pub">
                          {details.citationUrl ? (
                            <a
                              href={details.citationUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="mpill-pub-link"
                            >
                              {details.publicationTitle}
                              <BoxArrowUpRight size={10} className="mpill-ext-icon" />
                            </a>
                          ) : (
                            <span className="mpill-pub-title">{details.publicationTitle}</span>
                          )}
                        </div>

                        {details.authors && (
                          <p className="mpill-authors">{details.authors}</p>
                        )}

                        {details.moreInfo && (
                          <p className="mpill-note">
                            <span className="mpill-note-kw">Note</span>
                            {details.moreInfo}
                          </p>
                        )}

                        {details.repoUrl && (
                          <a
                            href={details.repoUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mpill-github"
                          >
                            <Github size={13} />
                            View on GitHub
                          </a>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <hr className="my-4" />
        <h4 className="text-center mb-3">Input Data Format</h4>
        <div className="fmt-grid">
          {FORMAT_SPECS.map(({ key, name, methods, columns }) => (
            <div key={key} className={`fmt-card fmt-card--${key}`}>
              <div className="fmt-card-name">{name}</div>
              <div className="fmt-methods">
                {methods.map(m => (
                  <span key={m} className="fmt-method-chip">{m}</span>
                ))}
              </div>
              <div className="fmt-columns">
                {columns.map(({ col, desc }) => (
                  <div key={col} className="fmt-col-row">
                    <code className="fmt-col-code">{col}</code>
                    <span className="fmt-col-desc">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="howto-note mt-3">
          <span className="howto-note-tag">Note</span>
          <div className="howto-note-body">
            <p className="howto-note-line">
              <strong>Single-substrate methods also accept Multi-Substrate and Full-Reaction files.</strong>{' '}
              They predict every semicolon-separated substrate independently: k<sub>cat</sub> is set to the maximum prediction.
            </p>
            <p className="howto-note-line">
              <code>Protein Sequence</code> may also contain <code>;</code>-separated candidate sequences; output stays one row, using max k<sub>cat</sub>, min K<sub>M</sub>, and max k<sub>cat</sub>/K<sub>M</sub>.
            </p>
          </div>
        </div>

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

          <Button
            href="/templates/full_reaction_template.csv"
            download
            className="btn btn-custom-subtle"
          >
            <BoxArrowInDown className="me-2" />
            Full-Reaction Template
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
}
