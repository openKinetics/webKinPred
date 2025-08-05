import React from 'react';
import { Card } from 'react-bootstrap';

export default function HowToUseCard() {
  return (
    <Card className="section-container section-how-to-use mb-4">
      <Card.Body>
        <h3>How to Use This Tool</h3>
        <p>
          This tool predicts kinetic parameters (k<sub>cat</sub> and/or K<sub>M</sub>) for enzyme-catalysed reactions using various ML models.
        </p>
        <p>
          If you tick <strong>“Prefer experimental data”</strong>, the system will first check BRENDA, SABIO-RK,
          and UniProt for experimental values for your protein–substrate pair. If a match is found, that value
          will be returned instead of a model prediction.
        </p>
        <p><strong>Steps:</strong></p>
        <ol>
          <li>Select what you want to predict (k<sub>cat</sub>, K<sub>M</sub>, or both).</li>
          <li>Upload your reaction data as a CSV file.</li>
          <li>Choose prediction method(s) (after optional validation).</li>
        </ol>

        <h5 className="mt-4">CSV Input Headers &amp; Expected Cell Contents</h5>
        <ul className="list-unstyled ps-3">
          <li className="mb-2"><span className="csv-col">Protein Sequence</span> — the full amino-acid sequence.</li>
          <li className="mb-1 fw-bold">Single-substrate models (DLKcat / EITLEM / UniKP)</li>
          <ul className="ps-3 mb-2">
            <li><span className="csv-col">Substrate</span> — one <code>SMILES</code> or <code>InChI</code> string</li>
          </ul>
          <li className="mb-1 fw-bold">Multi-substrate model (TurNup)</li>
          <ul className="ps-3">
            <li><span className="csv-col">Substrates</span> — semicolon-separated <code>SMILES</code>/<code>InChI</code> list</li>
            <li><span className="csv-col">Products</span> — semicolon-separated <code>SMILES</code>/<code>InChI</code> list</li>
          </ul>
        </ul>

        <p className="ps-1">
          Multi-substrate CSVs can also be used for KM predictions. Each entry in
          <span className="csv-col mx-1">Substrates</span> receives its own KM value (semicolon-separated in the output).
        </p>

        <h6>📥 Example Templates:</h6>
        <ul>
          <li><a href="/templates/single_substrate_template.csv" download>Download single-substrate template</a></li>
          <li><a href="/templates/multi_substrate_template.csv" download>Download multi-substrate template</a></li>
        </ul>
      </Card.Body>
    </Card>
  );
}
