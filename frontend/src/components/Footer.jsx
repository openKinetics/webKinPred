// src/components/Footer.js
import React from 'react';
import '../styles/components/Footer.css';
import recon4Logo from '../assets/recon4imd_logo.png';
import euLogo from '../assets/eu_logo.png';

const funders = [
  { name: 'EU Horizon Europe', grant: '#101080997' },
  { name: 'Swiss SERI', grant: '#23.00232' },
  { name: 'UKRI', grant: '#10083717 & #10080153' },
  { name: 'FNR', grant: 'PRIDE21/16763386/CANBIO2' },
  { name: 'Novo Nordisk Foundation', grant: '#NNF10CC1016517 & #NNF20CC0035580' },
  { name: 'Knut & Alice Wallenberg Foundation', grant: null },
  { name: 'EU Horizon 2020', grant: '#686070 & #814650' },
  { name: 'National Key R&D China', grant: '2025YFA0922700' },
  { name: 'German Research Foundation (DFG)', grant: 'EXC-2048/1 · project ID 390686111' },
  { name: 'DFG SFB 1644/1', grant: 'project no. 512328399' },
];

function Footer() {
  return (
    <footer className="custom-footer">
      <div className="container-fluid">
        <div className="footer-inner">

          {/* Brand */}
          <p className="footer-brand mb-0">OpenKineticsPredictor</p>

          {/* Funding badges */}
          <div className="funding-section">
            <span className="funding-label">Funded by</span>
            <div className="funding-badges">
              {funders.map(({ name, grant }) => (
                <span key={name} className="funding-badge">
                  <span className="badge-name">{name}</span>
                  {grant && <span className="badge-grant">{grant}</span>}
                </span>
              ))}
            </div>
          </div>

          {/* Partner logos */}
          <div className="footer-logos">
            <a href="http://recon4imd.org/" target="_blank" rel="noopener noreferrer">
              <img src={recon4Logo} alt="Recon4IMD" className="footer-logo footer-logo--recon4" />
            </a>
            <a href="https://cordis.europa.eu/project/id/101080997" target="_blank" rel="noopener noreferrer">
              <img src={euLogo} alt="Co-funded by the European Union" className="footer-logo footer-logo--eu" />
            </a>
          </div>

        </div>
      </div>
    </footer>
  );
}

export default Footer;
