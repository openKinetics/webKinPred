// src/components/Footer.js

import React from 'react';
import './Footer.css';
import euLogo from '../assets/eu_logo.png';
import reconLogo from '../assets/recon4imd_logo.png';

function Footer() {
  return (
    <footer className="custom-footer">
      <div className="footer-content">
        <p className="footer-text">
          &copy; {new Date().getFullYear()} KineticXPredictor
        </p>
        <p className="funding-statement">
          Recon4IMD is co-funded by the European Union's Horizon Europe Framework Programme (101080997).
        </p>
        <div className="footer-logos">
          <a href="https://www.recon4imd.org/" target="_blank" rel="noopener noreferrer">
            <img src={reconLogo} alt="Recon4IMD Logo" className="footer-logo recon-logo" />
          </a>
          <a href="https://cordis.europa.eu/project/id/101080997" target="_blank" rel="noopener noreferrer">
            <img src={euLogo} alt="European Union Logo" className="footer-logo eu-logo" />
          </a>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
