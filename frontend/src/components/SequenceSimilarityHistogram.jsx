import React, { useState, useMemo } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
import { Accordion, Row, Col } from 'react-bootstrap';
import { InfoCircle, Diagram3, GearFill } from 'react-bootstrap-icons';
import '../styles/components/SequenceSimilarityHistogram.css';

// Register Chart.js components
ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const GRANULARITY_BINS = [1, 2, 4, 5, 10, 20, 25, 50, 100];

function SequenceSimilarityHistogram({ similarityData }) {
  const models = similarityData ? Object.keys(similarityData) : [];
  const [activeModel, setActiveModel] = useState(models[0] || '');
  const [similarityType, setSimilarityType] = useState('max'); // 'max' or 'mean'
  
  const DEFAULT_BIN_INDEX = 3; // Default to 5 bins
  const [binIndex, setBinIndex] = useState(DEFAULT_BIN_INDEX);
  const numberOfBins = GRANULARITY_BINS[binIndex];

  const { labels, dataValues, countValues } = useMemo(() => {
    if (!similarityData || !activeModel || !similarityData[activeModel]) {
      return { labels: [], dataValues: [], countValues: [] };
    }

    const modelData = similarityData[activeModel];
    const rawPercentages = similarityType === 'mean' ? modelData.histogram_mean : modelData.histogram_max;
    const rawCounts = similarityType === 'mean' ? modelData.count_mean : modelData.count_max;
    
    if (!rawPercentages || !rawCounts) {
      return { labels: [], dataValues: [], countValues: [] };
    }
    
    if (numberOfBins >= 100) {
        const sortedLabels = Object.keys(rawPercentages).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
        return {
            labels: sortedLabels,
            dataValues: sortedLabels.map(label => rawPercentages[label]),
            countValues: sortedLabels.map(label => rawCounts[label])
        };
    }

    const newLabels = [];
    const newDataValues = [];
    const newCountValues = [];
    const binSize = 101 / numberOfBins;

    for (let i = 0; i < numberOfBins; i++) {
        const start = Math.floor(i * binSize);
        const end = Math.floor((i + 1) * binSize) - 1;
        
        const actualEnd = (i === numberOfBins - 1) ? 100 : end;

        let binPercentage = 0;
        let binCount = 0;
        
        for (let j = start; j <= actualEnd; j++) {
            const key = String(j);
            if (rawPercentages[key] !== undefined) {
                binPercentage += rawPercentages[key];
            }
            if (rawCounts[key] !== undefined) {
                binCount += rawCounts[key];
            }
        }
        
        newLabels.push(`${start}-${actualEnd}`);
        newDataValues.push(binPercentage);
        newCountValues.push(binCount);
    }
    
    return { labels: newLabels, dataValues: newDataValues, countValues: newCountValues };

  }, [activeModel, similarityData, similarityType, numberOfBins]);


  if (!similarityData) return null;

  const modelData = similarityData[activeModel] || {};
  const averageSimilarity = similarityType === 'mean'
    ? modelData.average_mean_similarity
    : modelData.average_max_similarity;

  const data = {
    labels,
    datasets: [{
      label: `% of input at similarity`,
      data: dataValues,
      backgroundColor: 'rgba(75,192,192,0.4)',
      borderColor: 'rgba(75,192,192,1)',
      borderWidth: 1,
      barPercentage: 1.0,
      categoryPercentage: 0.9
    }]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        callbacks: {
          title: (tooltipItems) => `Similarity Range: ${tooltipItems[0].label}%`,
          label: (context) => {
            const percentage = context.parsed.y;
            const count = countValues[context.dataIndex];
            return `${percentage.toFixed(1)}% of input (${count} sequence${count !== 1 ? 's' : ''}) found in this range`;
          }
        }
      }
    },
    scales: {
      x: {
        title: { display: true, text: `${similarityType === 'mean' ? 'Mean' : 'Max'} Sequence Similarity (%)`, color: 'white', font: { size: 14 } },
        ticks: { color: 'white', font: { size: 12 } },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      y: {
        title: { display: true, text: 'Frequency of Your Input Sequences (%)', color: 'white', font: { size: 14 } },
        ticks: { color: 'white', font: { size: 12 } },
        grid: { color: 'rgba(255,255,255,0.1)' },
        beginAtZero: true
      }
    }
  };

  return (
    <div>
      <h5 className="tab-section-header text-center mb-4">Sequence Similarity Histogram</h5>

      <Accordion defaultActiveKey="0" className="custom-accordion">
        <Accordion.Item eventKey="0">
          <Accordion.Header>
            <InfoCircle className="me-2" /> What does this chart show?
          </Accordion.Header>
          <Accordion.Body>
            This histogram displays the distribution of sequence similarities found when searching your input protein sequences against the training data of a selected model. This helps you understand how novel or similar your sequences are compared to the data the model was trained on, which can influence prediction confidence.
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="1">
          <Accordion.Header>
            <Diagram3 className="me-2" /> Understanding Similarity Types
          </Accordion.Header>
          <Accordion.Body>
            You can view the similarity distribution in two ways:
            <Row className="mt-2">
              <Col md={6}>
                <strong>Max Similarity</strong>
                <p className="text-white-50 small">The single highest percentage identity found for each of your sequences. This represents the "best match" in the training data.</p>
              </Col>
              <Col md={6}>
                <strong>Mean Similarity</strong>
                <p className="text-white-50 small">The average percentage identity calculated from all significant alignment hits found for each of your sequences.</p>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="2">
          <Accordion.Header>
            <GearFill className="me-2" /> Technical Details & Parameters
          </Accordion.Header>
          <Accordion.Body>
            <p>If no significant hits are found for a sequence, both its mean and max similarity are set to 0%. All similarity values are rounded to the nearest integer for binning.</p>
            <strong>MMseqs2 Parameters Used:</strong>
            <ul className="list-unstyled ps-3 mt-1 parameter-list">
              <li><code>-s</code> (sensitivity): <code>7.5</code></li>
              <li><code>-e</code> (E-value): <code>0.001</code></li>
              <li><code>--max-seqs</code>: <code>5000</code></li>
            </ul>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>

      <div className="mt-4">
        <h5 className="text-white-50 text-center mb-3">Chart Controls</h5>
        <div className="p-3 rounded" style={{ backgroundColor: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <div className="row justify-content-between align-items-center mb-4">
            <div className="col-md-auto mb-3 mb-md-0">
              <label className="form-label d-block mb-2 small text-white-50">Dataset</label>
              <div className="col-md-auto mb-3 mb-md-0">
                            <div>
                              {models.map(model => (
                                <button 
                                  key={model} 
                                  onClick={() => setActiveModel(model)} 
                                  // This is the only line that changes
                                  className={`btn btn-kave-toggle ${activeModel === model ? 'active' : ''} me-2`}
                                >
                                  {model}
                                </button>
                              ))}
                            </div>
              </div>
            </div>
            <div className="col-md-auto">
              <label className="form-label d-block mb-2 small text-white-50">View Type</label>
              <div className="btn-group" role="group">
                <button type="button" className={`btn ${similarityType === 'max' ? 'btn-light text-dark fw-bold' : 'btn-outline-light'}`} onClick={() => setSimilarityType('max')}>
                  Max
                </button>
                <button type="button" className={`btn ${similarityType === 'mean' ? 'btn-light text-dark fw-bold' : 'btn-outline-light'}`} onClick={() => setSimilarityType('mean')}>
                  Mean
                </button>
              </div>
            </div>
          </div>

          <div className="row justify-content-center">
            <div className="col-lg-7 d-flex align-items-center text-white">
              <label htmlFor="granularity-slider" className="form-label me-3 mb-0 small text-white-50">Granularity</label>
              <input
                type="range"
                className="form-range"
                id="granularity-slider"
                min="0"
                max={GRANULARITY_BINS.length - 1}
                value={binIndex}
                onChange={(e) => setBinIndex(parseInt(e.target.value, 10))}
              />
              <span className="fw-bold ms-3" style={{ minWidth: '70px', textAlign: 'right' }}>{numberOfBins} Bins</span>
            </div>
          </div>
        </div>
      </div>
      
      {averageSimilarity !== null && (
        <div className="text-center my-4">
          <p className="text-white mb-0">Input sequences vs. <strong>{activeModel}</strong> training data ({similarityType} similarity)</p>
        </div>
      )}
      <Bar data={data} options={options} />
    </div>
  );
}

export default SequenceSimilarityHistogram;