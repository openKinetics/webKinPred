// src/components/About.js
import { useEffect, useState } from 'react';
import { Check2, Clipboard, Download, Envelope } from 'react-bootstrap-icons';
import { useLocation } from 'react-router-dom';
import apiClient from './appClient';
import './ApiDocs/ApiDocs.css';

const teamInstitutions = [
  {
    institution: 'Digital Metabolic Twin Center, University of Galway',
    location: 'Galway, Ireland',
    members: ['Saleh Alwer', 'Chidi Egwu', 'Farid Zare', 'Jack McGoldrick', 'Ronan Fleming'],
  },
  {
    institution: 'Faculty of Science, Technology and Medicine, University of Luxembourg',
    location: 'Belvaux, Luxembourg',
    members: ['Thomas Sauter', 'Hugues Escoffier'],
  },
  {
    institution: 'Ratul Chowdhury Lab, Nanovaccine Institute, Department of Chemical & Biological Engineering, Iowa State University, Ames National Laboratory, Critical Mineral Innovation Hub',
    location: 'Ames, IA, USA',
    members: ['Supantha Dey', 'Vaishnavey SR', 'Ratul Chowdhury'],
  },
  {
    institution: 'Systems and Synthetic Biology Laboratory, College of Engineering, University of Nebraska–Lincoln',
    location: 'Lincoln, NE, USA',
    members: ['Abraham Osinuga', 'Rajib Saha'],
  },
  {
    institution: 'Systems Biology, Department of Life Sciences, Chalmers University of Technology',
    location: 'Gothenburg, Sweden',
    members: ['Eduard Kerkhoven'],
  },
  {
    institution: 'Luo Laboratory, Center for Synthetic Biochemistry, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences',
    location: 'Shenzhen, China',
    members: ['Han Yu', 'Xiaozhou Luo'],
  },
  {
    institution: 'Maranas Group, Department of Chemical Engineering, The Pennsylvania State University',
    location: 'University Park, PA, USA',
    members: ['Costas D. Maranas', 'Veda Boorla', 'Somtirtha Santra'],
  },
  {
    institution: 'Shanghai Zelixir Biotech Co. Ltd',
    location: 'Shanghai, China',
    members: ['Liangzhen Zheng'],
  },
  {
    institution: 'College of Computing and Data Science, Nanyang Technological University',
    location: 'Singapore',
    members: ['Zechen Wang'],
  },
  {
    institution: 'Töpfer Lab, Institute for Plant Sciences, University of Cologne',
    location: 'Cologne, Germany',
    members: ['Nadine Töpfer', 'Jan-Niklas Weder', 'Karim Taha'],
  },
];

const METRIC_CARDS = [
  { key: 'jobs_completed', label: 'Jobs' },
  { key: 'reactions_completed', label: 'Reactions' },
  { key: 'unique_protein_sequences', label: 'Distinct proteins' },
];

const PARAMETER_BREAKDOWN = [
  {
    key: 'kcat_predictions_completed',
    tone: 'kcat',
    label: (
      <>
        <span className="about-math">k<sub>cat</sub></span>
      </>
    ),
  },
  {
    key: 'km_predictions_completed',
    tone: 'km',
    label: (
      <>
        <span className="about-math">K<sub>M</sub></span>
      </>
    ),
  },
  {
    key: 'kcat_km_predictions_completed',
    tone: 'kcat-km',
    label: (
      <>
        <span className="about-math-frac" aria-label="k sub cat over K sub M">
          <span className="about-math-frac__num">k<sub>cat</sub></span>
          <span className="about-math-frac__den">K<sub>M</sub></span>
        </span>
      </>
    ),
  },
];

const numberFormatter = new Intl.NumberFormat('en-US');
const ABOUT_STATS_STORAGE_KEY = 'about_stats_payload_v1';
const CITATION_BIB_PATH = '/citations/openkineticspredictor.bib';
const citationText = String.raw`@unpublished{alwer2026accessing,
  author = {Alwer, Saleh and Escoffier, Hugues and Taha, Karim and Boorla, Veda and Yu, Han and Santra, Somtirtha and Wang, Zechen and Egwu, Chidi and Osinuga, Abraham and Dey, Supantha and Srinivasan Raghunath, Vaishnavey and Zare, Farid and McGoldrick, Jack and Weder, Jan-Niklas and Kerkhoven, Eduard and Luo, Xiaozhou and Maranas, Costas D. and Zheng, Liangzhen and Wittig, Ulrike and Chowdhury, Ratul and Saha, Rajib and T{\"o}pfer, Nadine and Sauter, Thomas and Fleming, Ronan M. T.},
  title = {{Accessing Enzyme Kinetic Data and Prediction Methods at Scale}},
  note = {Unpublished manuscript},
  year = {2026}
}`;

const About = () => {
  const location = useLocation();
  const [copied, setCopied] = useState(false);
  const [stats, setStats] = useState(() => {
    try {
      const raw = window.localStorage.getItem(ABOUT_STATS_STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    let isMounted = true;

    apiClient.get('/about-stats/')
      .then((response) => {
        if (!isMounted) return;
        const payload = response.data || {};
        setStats(payload);
        try {
          window.localStorage.setItem(ABOUT_STATS_STORAGE_KEY, JSON.stringify(payload));
        } catch {
          // Best effort only.
        }
      })
      .catch(() => {
        if (!isMounted) return;
        setStats(prev => prev ?? {});
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (location.hash !== '#contact') return;

    window.requestAnimationFrame(() => {
      document.getElementById('contact')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    });
  }, [location.hash]);

  const getMetricValue = (key) => {
    const value = stats?.[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  };

  const formatMetric = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    return numberFormatter.format(value);
  };

  const formatPercent = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    if (value > 0 && value < 0.1) return '<0.1%';
    if (value < 10) return `${value.toFixed(1)}%`;
    return `${Math.round(value)}%`;
  };

  const parameterBreakdown = PARAMETER_BREAKDOWN.map((metric) => ({
    ...metric,
    value: getMetricValue(metric.key),
  }));

  const parameterBreakdownTotal = parameterBreakdown.every(metric => metric.value !== null)
    ? parameterBreakdown.reduce((sum, metric) => sum + metric.value, 0)
    : null;
  const parameterPredictionTotal =
    getMetricValue('parameter_predictions_completed') ?? parameterBreakdownTotal;
  const parameterBreakdownStats = parameterBreakdown.map((metric) => {
    const value = metric.value ?? 0;
    const percent = parameterBreakdownTotal ? (value / parameterBreakdownTotal) * 100 : null;

    return {
      ...metric,
      percent,
      barShare: value > 0 ? Math.max(percent ?? 0, 1) : 0,
    };
  });

  const copyCitation = () => {
    if (!navigator.clipboard) return;

    navigator.clipboard.writeText(citationText)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(err => console.error('Failed to copy: ', err));
  };

  return (
    <div className="about-page">
      <div className="about-container container">
        <header className="about-header">
          <h1>OpenKineticsPredictor</h1>
          <p className="about-hero-copy">
            We developed this platform to make kinetic parameter prediction methods more accessible in an open source setting, so it can continue to expand as more methods are published and introduced.
          </p>
        </header>

        <section className="about-section about-metrics-section" aria-label="Platform usage metrics">
          <div className="about-section-heading">
            <h2>Usage</h2>
          </div>

          <div className="about-metrics-grid">
            {METRIC_CARDS.map((metric) => (
              <article key={metric.key} className="about-metric-card">
                <span className="about-metric-label">{metric.label}</span>
                <strong className="about-metric-value">{formatMetric(getMetricValue(metric.key))}</strong>
              </article>
            ))}

            <article className="about-metric-card about-metric-card--parameter">
              <div className="about-parameter-total">
                <span className="about-metric-label">Parameter predictions</span>
                <strong className="about-metric-value">
                  {formatMetric(parameterPredictionTotal)}
                </strong>
              </div>
              <div className="about-parameter-breakdown" aria-label="Parameter prediction breakdown">
                {parameterBreakdownStats.map((metric) => (
                  <div key={metric.key} className="about-breakdown-row">
                    <div className="about-breakdown-main">
                      <span className={`about-breakdown-marker about-breakdown-marker--${metric.tone}`} aria-hidden="true" />
                      <span className="about-breakdown-label">{metric.label}</span>
                      <strong>{formatMetric(metric.value)}</strong>
                      <span className="about-breakdown-percent">{formatPercent(metric.percent)}</span>
                    </div>
                    <div className="about-breakdown-bar" aria-hidden="true">
                      <span
                        className={`about-breakdown-bar-fill about-breakdown-bar-fill--${metric.tone}`}
                        style={{ width: `${metric.barShare}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>

        <section className="about-section about-consortium-section" aria-labelledby="about-consortium-title">
          <div className="about-section-heading">
            <h2 id="about-consortium-title">Contributors</h2>
          </div>

          <div className="about-institution-grid">
            {teamInstitutions.map((entry) => (
              <article key={entry.institution} className="about-institution-card">
                <div className="about-institution-heading">
                  <span className="about-institution-marker" aria-hidden="true" />
                  <div>
                    <h3 className="about-institution-name">{entry.institution}</h3>
                    {entry.location && <p className="about-institution-location">{entry.location}</p>}
                  </div>
                </div>
                <p className="about-member-list">{entry.members.join(', ')}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="about-section about-details-grid" aria-label="Contact and citation">
          <article id="contact" className="about-detail-card about-contact-card">
            <div className="about-detail-heading">
              <h2>Contact</h2>
            </div>
            <p className="about-detail-text">
              For questions about the platform, collaborations, or contributing prediction methods.
            </p>
            <a href="mailto:s.alwer1@universityofgalway.ie" className="about-contact-link">
              <Envelope aria-hidden="true" />
              <span>s.alwer1@universityofgalway.ie</span>
            </a>
          </article>

          <article className="about-detail-card about-citation-card">
            <div className="about-detail-heading about-citation-heading">
              <h2>Citation</h2>
              <div className="about-citation-actions">
                <button type="button" className="about-citation-button" onClick={copyCitation}>
                  {copied ? <Check2 aria-hidden="true" /> : <Clipboard aria-hidden="true" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <a
                  className="about-citation-button"
                  href={CITATION_BIB_PATH}
                  download="openkineticspredictor.bib"
                >
                  <Download aria-hidden="true" />
                  <span>Download</span>
                </a>
              </div>
            </div>
            <pre className="about-citation-text">{citationText}</pre>
          </article>
        </section>
      </div>
    </div>
  );
};

export default About;
