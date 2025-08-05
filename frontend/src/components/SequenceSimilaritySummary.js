import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

// Register Chart.js components.
ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

function SequenceSimilarityHistogram({ similarityData }) {
  const models = similarityData ? Object.keys(similarityData) : [];
  const [activeModel, setActiveModel] = useState(models[0] || '');
  const [similarityType, setSimilarityType] = useState('max'); // or 'mean'

  if (!similarityData) return null;

  const modelData = similarityData[activeModel] || {};
  const histogram = similarityType === 'mean'
    ? modelData.histogram_mean || {}
    : modelData.histogram_max || {};
  
  const histogramCounts = similarityType === 'mean'
  ? modelData.count_mean || {}
  : modelData.count_max || {};

  const averageSimilarity = similarityType === 'mean'
    ? modelData.average_mean_similarity
    : modelData.average_max_similarity;
  
  // Sort identity bins numerically
  const labels = Object.keys(histogram)
    .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));

  const dataValues = labels.map(percent => histogram[percent]);
  const countValues = labels.map(percent => histogramCounts[percent]);

  const data = {
    labels,
    datasets: [{
      label: `% of input at similarity `,
      data: dataValues,
      count: countValues,
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
      legend: {
        position: 'top',
        labels: {
          color: 'white',
          font: {
            size: 14
          }
        }
      },
      tooltip: {
        enabled: true,
        callbacks: {
          title: (tooltipItems) =>
            tooltipItems.map(item => `${item.label}% Similarity`),
          label: (context) => {
            const percentage = context.parsed.y;
            const count = countValues[context.dataIndex]; // Get the count from the countValues array
            return `${percentage.toFixed(1)}% of input (${count} sequence${count !== 1 ? 's' : ''}) is at this similarity`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: `${similarityType === 'mean' ? 'Mean' : 'Max'} Sequence Similarity (%)`,

          color: 'white',
          font: { size: 14 }
        },
        ticks: {
          color: 'white',
          font: { size: 12 }
        },
        grid: {
          color: 'rgba(255,255,255,0.1)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Frequency of Your Input Sequences (%)',
          color: 'white',
          font: { size: 14 }
        },
        ticks: {
          color: 'white',
          font: { size: 12 }
        },
        grid: {
          color: 'rgba(255,255,255,0.1)'
        },
        beginAtZero: true
      }
    }
  };

  return (
    <div className="mt-4">
      <h5 className="text-center mt-3 mb-3">Sequence Similarity Histogram</h5>
      <p className="text-white">
        Each input protein sequence is searched against the <i>k</i><sub>cat</sub> training data of each model using MMseqs2. The histogram displays the distribution of sequence similarities, which can be viewed in two ways:
        <br />
        <ul className="ms-4">
          <li>
            <strong>Max Similarity</strong> – the single highest percentage identity found for each of your sequences against the training data. This is the largest similarity value among all alignment hits.
          </li>
          <li className="mt-2">
            <strong>Mean Similarity</strong> – the average percentage identity calculated from all significant alignment hits found for each sequence. This is the mean of the identities from those hits, not from the entire target database. 
          </li>
        </ul>
        Both values are rounded to the nearest percent. If no hits are found for a sequence, both the mean and max similarity are set to 0%. This histogram shows the frequency of each similarity value (0–100%).
        <br />
        <strong>MMseqs2 Parameters:</strong>
        <ul className="ms-4">
          <li><code>-s</code> (sensitivity): <code>7.5</code></li>
          <li><code>-e</code> (significance cutoff, E-value): <code>0.001</code></li>
          <li><code>--max-seqs</code> (maximum number of target sequences to consider per query): <code>5000</code></li>
        </ul>
        Click a model below to toggle between the training datasets.
      </p>
      <div className="mb-3 d-flex align-items-center">
        <div className="me-3">
          {models.map(model => (
            <button
              key={model}
              onClick={() => setActiveModel(model)}
              className={`btn ${activeModel === model ? 'btn-primary' : 'btn-secondary'} me-2`}
            >
              {model}
            </button>
          ))}
        </div>
        <div className="btn-group" role="group" aria-label="Similarity Type Toggle">
          <button
            type="button"
            className={`btn ${similarityType === 'max' ? 'btn-light text-dark fw-bold' : 'btn-outline-light'}`}
            onClick={() => setSimilarityType('max')}
          >
            Max
          </button>
          <button
            type="button"
            className={`btn ${similarityType === 'mean' ? 'btn-light text-dark fw-bold' : 'btn-outline-light'}`}
            onClick={() => setSimilarityType('mean')}
          >
            Mean
          </button>
        </div>
      </div>

      {averageSimilarity !== null && (
        <div className="text-white mb-3">
          <strong>
            {similarityType === 'mean' ? 'Mean' : 'Max'} similarity distribution of your input sequences 
            against {activeModel}'s training data
          </strong>
        </div>
      )}

      <Bar data={data} options={options} />
    </div>
  );
}

export default SequenceSimilarityHistogram;
