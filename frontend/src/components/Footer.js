// src/components/Footer.js
import React from 'react';
import euLogo from '../assets/eu_logo.png';
import reconLogo from '../assets/recon4imd_logo.png';

function Footer() {
  return (
    <footer className="custom-footer">
      <div className="container-fluid">
        {/* A single row to align all content, stacking on small screens */}
        <div className="row align-items-center text-center text-lg-start">

          {/* Column 1: App Name and Copyright */}
          <div className="col-lg-4 col-md-12 mb-3 mb-lg-0">
            <p className="footer-brand mb-0">KineticXPredictor</p>
            <p className="copyright-text mb-0">2025 University of Galway</p>
          </div>

          {/* Column 2: Funding Information and Logos */}
          <div className="col-lg-8 col-md-12">
            <div className="d-flex justify-content-center justify-content-lg-end align-items-center">
              <p className="funding-text me-4 mb-0">
                Co-funded by the European Union's Horizon Europe Framework Programme (101080997)
              </p>
              <div className="footer-logos">
                <a href="https://www.recon4imd.org/" target="_blank" rel="noopener noreferrer">
                  <img src={reconLogo} alt="Recon4IMD Logo" className="footer-logo" />
                </a>
                <a href="https://cordis.europa.eu/project/id/101080997" target="_blank" rel="noopener noreferrer">
                  <img src={euLogo} alt="European Union Logo" className="footer-logo" />
                </a>
              </div>
            </div>
          </div>

        </div>
      </div>
    </footer>
  );
}

export default Footer;