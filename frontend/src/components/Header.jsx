// Header.jsx
import React from 'react';
import { Link, NavLink } from 'react-router-dom';
import { Activity } from 'react-bootstrap-icons';
import '../styles/components/navbar.css';
import { useTheme } from '../context/ThemeContext';

const SunIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="4.5" />
    <line x1="12" y1="2" x2="12" y2="4.5" />
    <line x1="12" y1="19.5" x2="12" y2="22" />
    <line x1="4.22" y1="4.22" x2="5.93" y2="5.93" />
    <line x1="18.07" y1="18.07" x2="19.78" y2="19.78" />
    <line x1="2" y1="12" x2="4.5" y2="12" />
    <line x1="19.5" y1="12" x2="22" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.93" y2="18.07" />
    <line x1="18.07" y1="5.93" x2="19.78" y2="4.22" />
  </svg>
);

const MoonIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const label = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
  return (
    <button
      type="button"
      className="theme-toggle-btn"
      onClick={toggleTheme}
      aria-label={label}
      title={label}
    >
      <span className="theme-toggle-track">
        <span className="theme-toggle-thumb">
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </span>
      </span>
    </button>
  );
}

function Header() {
  return (
    <header className="topbar">
      <div className="topbar-brand-group">
        <Link className="brand" to="/">
          <Activity size={22} aria-hidden="true" />
          <span>OpenKinetics Predictor</span>
        </Link>
        <nav className="product-links" aria-label="OpenKinetics products">
          <a href="https://data.openkinetics.org">Kinetic Data</a>
          <a href="https://openkinetics.org/">OpenKinetics Index</a>
        </nav>
      </div>
      <div className="topbar-end">
        <nav className="navlinks" aria-label="Primary navigation">
          <NavLink to="/track-job">Track Job</NavLink>
          <NavLink to="/api-docs">API</NavLink>
          <NavLink to="/contribute">Contribute</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}

export default Header;
