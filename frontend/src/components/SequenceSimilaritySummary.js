import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

// Register Chart.js components.
ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

function SequenceSimilarityHistogram({ similarityData }) {
  const models = similarityData ? Object.keys(similarityData) : [];
  const [activeModel, setActiveModel] = useState(models[0] || '');

  if (!similarityData) return null;

  // Extract histogram and average for the active model
  const histogram = similarityData[activeModel]?.histogram || {};
  const averageSimilarity = similarityData[activeModel]?.average_similarity ?? null;

  // Sort identity bins numerically
  const labels = Object.keys(histogram)
    .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));

  const dataValues = labels.map(percent => histogram[percent]);

  const data = {
    labels,
    datasets: [{
      label: `% of input at this Identity`,
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
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += parseFloat(context.parsed.y).toFixed(1) + '%';
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Max Sequence Similarity (%)',
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
        Each input sequence is searched against the pre-created model training databases using MMseqs2.
        The highest sequence identity for each query is rounded to the nearest percent (0–100%) and this histogram shows 
        the percentage frequency for each identity value.
        <br />
        Click a model below to toggle between the training datasets.
      </p>

      <div className="mb-3">
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

      {averageSimilarity !== null && (
        <div className="text-white mb-3">
          <strong>Average Max Identity with {activeModel} Training Data:</strong> {averageSimilarity}%
        </div>
      )}

      <Bar data={data} options={options} />
    </div>
  );
}

export default SequenceSimilarityHistogram;
