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
  
  const averageSimilarity = similarityType === 'mean'
    ? modelData.average_mean_similarity
    : modelData.average_max_similarity;
  
  // Sort identity bins numerically
  const labels = Object.keys(histogram)
    .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));

  const dataValues = labels.map(percent => histogram[percent]);

  const data = {
    labels,
    datasets: [{
      label: `% of input at similarity `,
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
            const value = context.parsed.y;
            return value !== null ? `${value.toFixed(1)}% of input is at this similarity` : '';
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
      <h5 className="text-white">Sequence Similarity Histogram</h5>
      <p className="text-white">
        Each input sequence is searched against the pre-created model training databases using mmseqs2.<br />
        The mean/max (toggle) sequence similarity for each query is rounded to the nearest percent (0–100%) and this histogram shows 
        the percentage frequency for each similarity value.
        <br />
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
          <strong>Average {similarityType === 'mean' ? 'Mean' : 'Max'} Similarity with {activeModel} Training Data:</strong> {averageSimilarity}%
        </div>
      )}

      <Bar data={data} options={options} />
    </div>
  );
}

export default SequenceSimilarityHistogram;
