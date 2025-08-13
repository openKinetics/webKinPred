import React, { useRef, useEffect, useCallback } from 'react';

// Enhanced configuration with scientifically-inspired parameters
const CONFIG = {
  // Entity counts - fewer for cleaner visuals
  enzymeCount: 8,
  substratesPerEnzyme: 3,
  
  // Visual parameters
  enzymeRadius: 35,
  activeSiteDepth: 0.3, // Ratio of radius for notch depth
  captureRadius: 120,
  bindingDistance: 8,
  
  // Physics
  brownianSpeed: 0.8,
  orientationRate: 0.06,
  releaseImpulse: 2.5,
  dampingFactor: 0.9,
  maxAngularVelocity: 0.15,
  
  // Timing (in frames at 60fps)
  catalysisDuration: 180, // 3 seconds
  inducedFitTime: 30,     // 0.5 seconds
  
  // Colors
  colors: {
    background: 'rgba(26, 35, 50, 1)',
    enzyme: 'rgba(100, 120, 150, 0.7)',
    enzymeGlow: 'rgba(100, 120, 150, 0.1)',
    substrate: '#3498db',
    substrateGlow: 'rgba(52, 152, 219, 0.15)',
    product: '#e74c3c',
    productGlow: 'rgba(231, 76, 60, 0.2)',
    catalysisPulse: 'rgba(255, 255, 255, 0.5)',
    environmentParticle: 'rgba(236, 240, 241, 0.1)'
  },
  
  // Visual effects
  globalAlpha: 0.25,
  glowIntensity: 0.2,
  fadeScrollDistance: 600,
  
  // Performance
  targetFrameTime: 16.67, // 60 FPS
  maxFrameTime: 33.33     // 30 FPS fallback
};

// Utility functions
const lerp = (a, b, t) => a + (b - a) * t;
const random = (min, max) => Math.random() * (max - min) + min;
const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
const distance = (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1);
const normalizeAngle = (angle) => {
  while (angle > Math.PI) angle -= 2 * Math.PI;
  while (angle < -Math.PI) angle += 2 * Math.PI;
  return angle;
};

// Entity creation functions
const createEnzyme = (x, y, radius = CONFIG.enzymeRadius) => ({
  id: Math.random(),
  type: 'enzyme',
  x, y, radius,
  rotation: random(0, Math.PI * 2),
  rotationSpeed: random(-0.01, 0.01),
  activeSiteAngle: random(0, Math.PI * 2),
  inducedFit: 0,
  targetInducedFit: 0,
  boundSubstrate: null,
  vx: random(-0.2, 0.2),
  vy: random(-0.2, 0.2),
  recoilVx: 0,
  recoilVy: 0
});

const createSubstrate = (x, y) => ({
  id: Math.random(),
  type: 'substrate',
  x, y,
  radius: random(3, 5),
  rotation: random(0, Math.PI * 2),
  angularVelocity: 0,
  vx: random(-CONFIG.brownianSpeed, CONFIG.brownianSpeed),
  vy: random(-CONFIG.brownianSpeed, CONFIG.brownianSpeed),
  state: 'wandering', // wandering, approaching, binding, catalyzing, releasing
  targetEnzyme: null,
  catalysisTimer: 0,
  isProduct: false,
  color: CONFIG.colors.substrate
});

// Physics and state update functions
const updateOrientation = (molecule, targetAngle) => {
  let angleDiff = normalizeAngle(targetAngle - molecule.rotation);
  
  molecule.angularVelocity = (molecule.angularVelocity || 0) * CONFIG.dampingFactor;
  molecule.angularVelocity += angleDiff * CONFIG.orientationRate;
  molecule.angularVelocity = clamp(molecule.angularVelocity, -CONFIG.maxAngularVelocity, CONFIG.maxAngularVelocity);
  
  molecule.rotation += molecule.angularVelocity;
};

const updatePosition = (entity, width, height, dt) => {
  // Apply Brownian motion for wandering molecules
  if (entity.type === 'substrate' && entity.state === 'wandering') {
    entity.vx += random(-0.1, 0.1) * dt;
    entity.vy += random(-0.1, 0.1) * dt;
  }
  
  // Clamp velocities to prevent tunneling
  const maxSpeed = entity.radius * 0.5;
  entity.vx = clamp(entity.vx, -maxSpeed, maxSpeed);
  entity.vy = clamp(entity.vy, -maxSpeed, maxSpeed);
  
  // Update position
  const nextX = entity.x + entity.vx * dt;
  const nextY = entity.y + entity.vy * dt;
  
  // Boundary handling with soft reflection
  if (nextX < entity.radius || nextX > width - entity.radius) {
    entity.vx *= -0.7; // Energy loss on collision
    entity.x = clamp(entity.x, entity.radius, width - entity.radius);
  } else {
    entity.x = nextX;
  }
  
  if (nextY < entity.radius || nextY > height - entity.radius) {
    entity.vy *= -0.7;
    entity.y = clamp(entity.y, entity.radius, height - entity.radius);
  } else {
    entity.y = nextY;
  }
};

const updateInducedFit = (enzyme) => {
  const fitRate = 0.08;
  enzyme.inducedFit = lerp(enzyme.inducedFit, enzyme.targetInducedFit, fitRate);
  
  // Prevent oscillation
  if (Math.abs(enzyme.inducedFit - enzyme.targetInducedFit) < 0.01) {
    enzyme.inducedFit = enzyme.targetInducedFit;
  }
};

  // Drawing functions
const drawEnzyme = (ctx, enzyme) => {
  ctx.save();
  
  // Subpixel positioning for smooth animation
  const x = Math.round(enzyme.x * 4) / 4;
  const y = Math.round(enzyme.y * 4) / 4;
  
  ctx.translate(x, y);
  ctx.rotate(enzyme.rotation);
  
  // Soft glow effect
  const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, enzyme.radius * 1.6);
  gradient.addColorStop(0, CONFIG.colors.enzymeGlow);
  gradient.addColorStop(1, 'rgba(100, 120, 150, 0)');
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(0, 0, enzyme.radius * 1.6, 0, Math.PI * 2);
  ctx.fill();
  
  // Main enzyme body with dynamic active site
  ctx.beginPath();
  const r = enzyme.radius;
  
  if (enzyme.inducedFit > 0.95) {
    // When fully induced (reaction completing), draw perfect circle
    ctx.arc(0, 0, r, 0, Math.PI * 2);
  } else {
    // Draw with active site notch that closes based on induced fit
    const notchAngle = enzyme.activeSiteAngle;
    const notchWidth = Math.PI / 6; // 30 degrees
    const maxNotchDepth = r * CONFIG.activeSiteDepth;
    const currentNotchDepth = maxNotchDepth * (1 - enzyme.inducedFit);
    
    for (let angle = 0; angle <= Math.PI * 2; angle += 0.05) {
      const angleFromNotch = Math.abs(normalizeAngle(angle - notchAngle));
      const isInNotch = angleFromNotch < notchWidth;
      
      let currentRadius = r;
      if (isInNotch) {
        // Smooth transition within the notch using cosine interpolation
        const notchProgress = angleFromNotch / notchWidth;
        const smoothFactor = (Math.cos(notchProgress * Math.PI) + 1) / 2;
        currentRadius = r - (currentNotchDepth * smoothFactor);
      }
      
      const px = Math.cos(angle) * currentRadius;
      const py = Math.sin(angle) * currentRadius;
      
      if (angle === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    
    ctx.closePath();
  }
  
  ctx.fillStyle = CONFIG.colors.enzyme;
  ctx.fill();
  ctx.strokeStyle = 'rgba(100, 120, 150, 0.9)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  
  ctx.restore();
};

const drawMolecule = (ctx, molecule) => {
  ctx.save();
  
  const x = Math.round(molecule.x * 4) / 4;
  const y = Math.round(molecule.y * 4) / 4;
  
  ctx.translate(x, y);
  ctx.rotate(molecule.rotation);
  
  // Glow effect
  const glowColor = molecule.isProduct ? CONFIG.colors.productGlow : CONFIG.colors.substrateGlow;
  ctx.fillStyle = glowColor;
  ctx.beginPath();
  ctx.arc(0, 0, molecule.radius * 2.2, 0, Math.PI * 2);
  ctx.fill();
  
  // Main molecule shape
  ctx.fillStyle = molecule.color;
  ctx.beginPath();
  
  if (molecule.isProduct) {
    // Product: simple circle, slightly smaller
    ctx.arc(0, 0, molecule.radius * 0.7, 0, Math.PI * 2);
  } else {
    // Substrate: oriented capsule
    const rx = molecule.radius * 1.3;
    const ry = molecule.radius * 0.6;
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  }
  
  ctx.fill();
  
  // Catalysis pulse effect
  if (molecule.state === 'catalyzing' && molecule.catalysisTimer > 0) {
    const progress = 1 - (molecule.catalysisTimer / CONFIG.catalysisDuration);
    const pulseAlpha = Math.sin(progress * Math.PI * 6) * 0.3 + 0.3;
    
    if (pulseAlpha > 0.1) {
      ctx.fillStyle = `rgba(255, 255, 255, ${pulseAlpha})`;
      ctx.beginPath();
      ctx.arc(0, 0, molecule.radius * 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  
  ctx.restore();
};

// Main component
function EnhancedProteinBackground() {
  const canvasRef = useRef(null);
  const entitiesRef = useRef({ enzymes: [], molecules: [] });
  const animationRef = useRef(null);
  const lastTimeRef = useRef(0);
  const accumulatorRef = useRef(0);
  const fadeEffectRef = useRef(1);
  const qualityRef = useRef({ particles: 1.0, glow: 1.0 });
  
  // Spawn entities with collision avoidance
  const spawnEntities = useCallback((width, height) => {
    const enzymes = [];
    const maxAttempts = 50;
    
    // Spawn enzymes with proper spacing
    for (let i = 0; i < CONFIG.enzymeCount; i++) {
      let placed = false;
      let attempts = 0;
      
      while (!placed && attempts < maxAttempts) {
        const candidate = createEnzyme(
          random(CONFIG.enzymeRadius * 2, width - CONFIG.enzymeRadius * 2),
          random(CONFIG.enzymeRadius * 2, height - CONFIG.enzymeRadius * 2)
        );
        
        const hasOverlap = enzymes.some(existing => {
          const dist = distance(candidate.x, candidate.y, existing.x, existing.y);
          return dist < (candidate.radius + existing.radius) * 3; // Safety margin
        });
        
        if (!hasOverlap) {
          enzymes.push(candidate);
          placed = true;
        }
        attempts++;
      }
    }
    
    // Spawn substrates
    const molecules = [];
    const totalMolecules = Math.floor(CONFIG.enzymeCount * CONFIG.substratesPerEnzyme * qualityRef.current.particles);
    
    for (let i = 0; i < totalMolecules; i++) {
      molecules.push(createSubstrate(
        random(20, width - 20),
        random(20, height - 20)
      ));
    }
    
    entitiesRef.current = { enzymes, molecules };
  }, []);
  
  // Update physics with fixed timestep
  const updatePhysics = useCallback((dt, width, height) => {
    const { enzymes, molecules } = entitiesRef.current;
    
    // Update enzymes
    enzymes.forEach(enzyme => {
      enzyme.rotation += enzyme.rotationSpeed * dt;
      
      // Gentle drift
      updatePosition(enzyme, width, height, dt);
      
      // Apply stronger recoil damping for more realistic movement
      enzyme.recoilVx *= 0.85;
      enzyme.recoilVy *= 0.85;
      enzyme.vx += enzyme.recoilVx;
      enzyme.vy += enzyme.recoilVy;
      
      // Update induced fit animation
      updateInducedFit(enzyme);
    });
    
    // Update molecules with state machine
    molecules.forEach(molecule => {
      const enzyme = molecule.targetEnzyme;
      
      switch (molecule.state) {
        case 'wandering':
          updatePosition(molecule, width, height, dt);
          
          // Look for nearby enzymes
          for (const e of enzymes) {
            if (e.boundSubstrate) continue;
            
            const dist = distance(molecule.x, molecule.y, e.x, e.y);
            if (dist < CONFIG.captureRadius) {
              molecule.state = 'approaching';
              molecule.targetEnzyme = e;
              break;
            }
          }
          break;
          
        case 'approaching':
          if (!enzyme || enzyme.boundSubstrate) {
            molecule.state = 'wandering';
            molecule.targetEnzyme = null;
            break;
          }
          
          // Move toward active site
          const siteAngle = enzyme.rotation + enzyme.activeSiteAngle;
          const targetX = enzyme.x + Math.cos(siteAngle) * enzyme.radius * 0.9;
          const targetY = enzyme.y + Math.sin(siteAngle) * enzyme.radius * 0.9;
          
          const dx = targetX - molecule.x;
          const dy = targetY - molecule.y;
          const dist = Math.hypot(dx, dy);
          
          if (dist < CONFIG.bindingDistance) {
            molecule.state = 'binding';
          } else {
            const speed = CONFIG.brownianSpeed * 1.5;
            molecule.vx = (dx / dist) * speed;
            molecule.vy = (dy / dist) * speed;
            updatePosition(molecule, width, height, dt);
          }
          
          // Orient toward binding site
          const targetAngle = siteAngle + Math.PI;
          updateOrientation(molecule, targetAngle);
          break;
          
        case 'binding':
          if (!enzyme) {
            molecule.state = 'wandering';
            break;
          }
          
          molecule.state = 'catalyzing';
          molecule.catalysisTimer = CONFIG.catalysisDuration;
          enzyme.boundSubstrate = molecule;
          enzyme.targetInducedFit = 1;
          break;
          
        case 'catalyzing':
          if (!enzyme) break;
          
          // Position at active site
          const bindingSiteAngle = enzyme.rotation + enzyme.activeSiteAngle;
          const bindingDepth = enzyme.radius * (0.7 - enzyme.inducedFit * 0.3);
          molecule.x = enzyme.x + Math.cos(bindingSiteAngle) * bindingDepth;
          molecule.y = enzyme.y + Math.sin(bindingSiteAngle) * bindingDepth;
          molecule.rotation = bindingSiteAngle + Math.PI;
          
          // Update catalysis timer
          molecule.catalysisTimer--;
          
          // Color transition during catalysis
          const progress = 1 - (molecule.catalysisTimer / CONFIG.catalysisDuration);
          // Transition from substrate blue (#3498db) to product red (#e74c3c)
          const r = Math.floor(lerp(52, 231, progress));   // 52 is blue R, 231 is red R
          const g = Math.floor(lerp(152, 76, progress));   // 152 is blue G, 76 is red G  
          const b = Math.floor(lerp(219, 60, progress));   // 219 is blue B, 60 is red B
          molecule.color = `rgb(${r}, ${g}, ${b})`;
          
          if (molecule.catalysisTimer <= 0) {
            molecule.state = 'releasing';
            molecule.isProduct = true;
            molecule.color = CONFIG.colors.product;
            enzyme.targetInducedFit = 0;
            
            // Apply gentle recoil to enzyme
            const recoilAngle = bindingSiteAngle + Math.PI;
            enzyme.recoilVx += Math.cos(recoilAngle) * 0.15;
            enzyme.recoilVy += Math.sin(recoilAngle) * 0.15;
          }
          break;
          
        case 'releasing':
          if (!enzyme) {
            molecule.state = 'wandering';
            molecule.targetEnzyme = null;
            break;
          }
          
          // Move away from enzyme with controlled release velocity
          const releaseAngle = enzyme.rotation + enzyme.activeSiteAngle + Math.PI;
          molecule.vx = Math.cos(releaseAngle) * CONFIG.releaseImpulse * 0.6;
          molecule.vy = Math.sin(releaseAngle) * CONFIG.releaseImpulse * 0.6;
          
          updatePosition(molecule, width, height, dt);
          
          // Check if far enough to release enzyme
          const distFromEnzyme = distance(molecule.x, molecule.y, enzyme.x, enzyme.y);
          if (distFromEnzyme > CONFIG.captureRadius * 1.2) {
            molecule.state = 'wandering';
            molecule.targetEnzyme = null;
            enzyme.boundSubstrate = null;
            
            // Gradual return to substrate state
            setTimeout(() => {
              if (molecule.isProduct) {
                molecule.isProduct = false;
                molecule.color = CONFIG.colors.substrate;
              }
            }, 3000);
          }
          break;
      }
    });
  }, []);
  
  // Animation loop with fixed timestep
  const animate = useCallback((currentTime) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas.getBoundingClientRect();
    
    // Fixed timestep physics
    const frameTime = Math.min(currentTime - lastTimeRef.current, CONFIG.maxFrameTime);
    accumulatorRef.current += frameTime;
    
    while (accumulatorRef.current >= CONFIG.targetFrameTime) {
      updatePhysics(1, width, height); // Normalized dt = 1
      accumulatorRef.current -= CONFIG.targetFrameTime;
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply global fade effect
    ctx.save();
    ctx.globalAlpha = fadeEffectRef.current * CONFIG.globalAlpha;
    
    const { enzymes, molecules } = entitiesRef.current;
    
    // Draw background particles for ambiance
    ctx.fillStyle = CONFIG.colors.environmentParticle;
    for (let i = 0; i < 15; i++) {
      const x = (currentTime * 0.01 + i * 50) % width;
      const y = (currentTime * 0.005 + i * 80) % height;
      ctx.beginPath();
      ctx.arc(x, y, 1, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw enzymes first (background layer)
    enzymes.forEach(enzyme => drawEnzyme(ctx, enzyme));
    
    // Draw molecules (foreground layer)
    molecules.forEach(molecule => drawMolecule(ctx, molecule));
    
    ctx.restore();
    
    lastTimeRef.current = currentTime;
    animationRef.current = requestAnimationFrame(animate);
  }, [updatePhysics]);
  
  // Handle scroll-based fading
  const handleScroll = useCallback(() => {
    const scrollY = window.scrollY || window.pageYOffset;
    fadeEffectRef.current = Math.max(0.3, 1 - scrollY / CONFIG.fadeScrollDistance);
  }, []);
  
  // Handle canvas resize
  const handleResize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas resolution
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    // Respawn entities for new dimensions
    spawnEntities(width, height);
  }, [spawnEntities]);
  
  // Performance monitoring
  const adjustQuality = useCallback(() => {
    const now = performance.now();
    const frameDelta = now - lastTimeRef.current;
    
    if (frameDelta > 25) { // Below 40 FPS
      qualityRef.current.particles = Math.max(0.5, qualityRef.current.particles * 0.9);
      qualityRef.current.glow = Math.max(0.3, qualityRef.current.glow * 0.9);
    } else if (frameDelta < 18) { // Above 55 FPS
      qualityRef.current.particles = Math.min(1.0, qualityRef.current.particles * 1.05);
      qualityRef.current.glow = Math.min(1.0, qualityRef.current.glow * 1.05);
    }
  }, []);
  
  // Setup and cleanup
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    let resizeObserver;
    
    // Initial setup
    handleResize();
    handleScroll();
    
    // Start animation
    animationRef.current = requestAnimationFrame(animate);
    
    // Setup resize observer
    resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(canvas);
    
    // Setup scroll listener
    window.addEventListener('scroll', handleScroll, { passive: true });
    
    // Performance monitoring interval
    const qualityInterval = setInterval(adjustQuality, 2000);
    
    // Cleanup
    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (resizeObserver) resizeObserver.disconnect();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      clearInterval(qualityInterval);
    };
  }, [animate, handleResize, handleScroll, adjustQuality]);
  
  const canvasStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100vh',
    zIndex: -1,
    pointerEvents: 'none'
  };
  
  return <canvas ref={canvasRef} style={canvasStyle} aria-hidden="true" />;
}

export default EnhancedProteinBackground;
