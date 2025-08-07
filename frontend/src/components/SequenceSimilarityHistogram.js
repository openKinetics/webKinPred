// /frontend/src/components/job-submission/components/SequenceSimilarityHistogram.js
import React, { useState, useMemo } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

// Register Chart.js components.
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
        
        // Ensure the last bin goes all the way to 100
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
      legend: { position: 'top', labels: { color: 'white', font: { size: 14 } } },
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
    <div className="mt-4">
      <h5 className="text-center mt-3 mb-3">Sequence Similarity Histogram</h5>
      <div className="text-white">
        <p>
          Each input protein sequence is searched against the <i>k</i><sub>cat</sub> training data of each model using
          MMseqs2. The histogram displays the distribution of sequence similarities, which can be viewed in two ways:
        </p>
        <ul className="ms-4">
          <li><strong>Max Similarity</strong> – the single highest percentage identity found for each of your sequences against the training data. This is the largest similarity value among all alignment hits.</li>
          <li className="mt-2"><strong>Mean Similarity</strong> – the average percentage identity calculated from all significant alignment hits found for each sequence. This is the mean of the identities from those hits, not from the entire target database.</li>
        </ul>
        <p>Both values are rounded to the nearest percent. If no hits are found for a sequence, both the mean and max similarity are set to 0%. This histogram shows the frequency of each similarity value (0–100%).</p>
        <p><strong>MMseqs2 Parameters:</strong></p>
        <ul className="ms-4">
          <li><code>-s</code> (sensitivity): <code>7.5</code></li>
          <li><code>-e</code> (significance cutoff, E-value): <code>0.001</code></li>
          <li><code>--max-seqs</code> (maximum number of target sequences to consider per query): <code>5000</code></li>
        </ul>
      </div>

      <div className="p-3 mt-4 rounded" style={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}>
        {/* Top row for Model and View Type selection */}
        <div className="row justify-content-between align-items-center mb-4">
          
          {/* Left Column: Model Selection */}
          <div className="col-md-auto mb-3 mb-md-0">
            <label className="form-label d-block mb-2 small text-white-50">Dataset</label>
            <div>
              {models.map(model => (
                <button key={model} onClick={() => setActiveModel(model)} className={`btn ${activeModel === model ? 'btn-primary' : 'btn-secondary'} me-2`}>
                  {model}
                </button>
              ))}
            </div>
          </div>

          {/* Right Column: View Type */}
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

        {/* Bottom Row for Granularity Slider */}
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

      {averageSimilarity !== null && (
        <div className="text-center my-4">
          <p className="text-white mb-0">
            Input sequences vs. <strong>{activeModel}</strong> training data ({similarityType} similarity)
          </p>
        </div>
      )}

      <Bar data={data} options={options} />
    </div>
  );
}

export default SequenceSimilarityHistogram;