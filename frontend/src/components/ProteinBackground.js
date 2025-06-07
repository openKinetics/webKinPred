import React, { useRef, useEffect } from 'react';

/* -------------------------- CONFIGURATION -------------------------- */
// This object holds all the tweakable parameters for the simulation.
// User's preferred values have been kept.
const CFG = {
    enzymeCount: 100,          // Number of enzyme structures
    moleculeCount: 500,       // Number of substrate/product molecules
    
    // --- Visuals & Colors ---
    enzymeColor: 'rgba(100, 120, 150, 0.7)',  // Cool, professional blue-gray for the enzyme
    substrateColor: '#3498db', // A distinct blue for the substrate
    productColor: '#e74c3c',   // A contrasting red for the product
    enzymeGlowColor: 'rgba(100, 120, 150, 0.1)',
    substrateGlowColor: 'rgba(52, 152, 219, 0.15)',
    productGlowColor: 'rgba(231, 76, 60, 0.2)',

    // --- Physics & Behavior ---
    moleculeSpeed: 0.2,        // Base speed for wandering molecules
    enzymeDriftSpeed: 0.08,      // How much the enzymes slowly drift
    enzymeRotationSpeed: 0.005, // Base rotation speed for enzymes
    
    // --- Kinetics Parameters ---
    bindingRadius: 120,        // (px) How close a substrate must be to be "attracted"
    bindingDistance: 0.8,        // (px) How close a substrate must be to "dock"
    bindingDepthFactor: 0.2,   // How deep inside the enzyme the molecule goes (0=edge, 1=center)
    releaseDistance: 200,      // (px) How far a product must travel before resetting
    releaseSpeed: 0.25,         // The gentle speed at which products leave the enzyme
    affinityRange: [0.2, 1.0], // Range for KM (binding affinity). Higher value = stronger attraction.
    kcatRange: [20, 1000],    // Range for kcat (turnover rate). Value is frames, so lower = faster turnover.

    // --- Spawning Parameters ---
    minEnzymeDistance: 200,      // (px) Minimum distance between enzymes at spawn
    maxSpawnAttempts: 50,      // Failsafe for the enzyme spawning algorithm
    minMoleculeDistance: 15,     // NEW: Minimum distance between spawned molecules
    maxMoleculeSpawnAttempts: 20, // NEW: Failsafe for molecule spawning

    // --- Global Effects ---
    globalAlphaFactor: 0.17,     // Overall transparency of the animation
    fadeScrollPx: 900,         // How many pixels of scrolling to fade out the animation
};


/**
 * A React component that renders a beautiful, lightweight, and scientifically-themed
 * animated background representing enzyme kinetics (KM and kcat).
 * Includes uniform enzyme distribution and more realistic binding/release mechanics.
 */
function ProteinBackground() {
  const canvasRef = useRef(null);
  const entities = useRef({ enzymes: [], molecules: [] });
  const fadeEffect = useRef(1); // For fade-on-scroll effect
  const animationFrameId = useRef(null);

  // Utility to generate a random number in a given range
  const R = (min, max) => Math.random() * (max - min) + min;

  /* -------------------------- ENTITY SPAWNING -------------------------- */
  // Initializes the enzymes and molecules with random properties.
  const spawnEntities = (width, height) => {
    // --- Spawn Enzymes with Uniform Distribution ---
    const enzymes = [];
    for (let i = 0; i < CFG.enzymeCount; i++) {
        let attempts = 0;
        while (attempts < CFG.maxSpawnAttempts) {
            const newEnzyme = {
                id: Math.random(),
                x: R(0, width),
                y: R(0, height),
                radius: R(20, 40),
                rotation: R(0, Math.PI * 2),
                rotationSpeed: R(-CFG.enzymeRotationSpeed, CFG.enzymeRotationSpeed),
                driftVx: R(-CFG.enzymeDriftSpeed, CFG.enzymeDriftSpeed),
                driftVy: R(-CFG.enzymeDriftSpeed, CFG.enzymeDriftSpeed),
                bindingAffinity: R(...CFG.affinityRange),
                catalyticRate: R(...CFG.kcatRange),
                activeSiteAngle: Math.floor(R(0, 6)) * (Math.PI / 3),
                boundMoleculeId: null,
            };

            let isOverlapping = false;
            for (const existingEnzyme of enzymes) {
                const dist = Math.hypot(newEnzyme.x - existingEnzyme.x, newEnzyme.y - existingEnzyme.y);
                if (dist < (newEnzyme.radius + existingEnzyme.radius + CFG.minEnzymeDistance)) {
                    isOverlapping = true;
                    break;
                }
            }

            if (!isOverlapping) {
                enzymes.push(newEnzyme);
                break;
            }
            attempts++;
        }
    }
    entities.current.enzymes = enzymes;


    // --- Spawn Molecules with Uniform Distribution ---
    const molecules = [];
    for (let i = 0; i < CFG.moleculeCount; i++) {
        let attempts = 0;
        while (attempts < CFG.maxMoleculeSpawnAttempts) {
            const newMolecule = {
                id: Math.random(),
                x: R(0, width),
                y: R(0, height),
                vx: R(-CFG.moleculeSpeed, CFG.moleculeSpeed),
                vy: R(-CFG.moleculeSpeed, CFG.moleculeSpeed),
                radius: R(3, 5.5),
                state: 'wandering',
                targetEnzymeId: null,
                boundTimer: 0,
            };

            let isOverlapping = false;
            // Check against other molecules
            for (const existingMolecule of molecules) {
                const dist = Math.hypot(newMolecule.x - existingMolecule.x, newMolecule.y - existingMolecule.y);
                if (dist < (newMolecule.radius + existingMolecule.radius + CFG.minMoleculeDistance)) {
                    isOverlapping = true;
                    break;
                }
            }
            
            if (isOverlapping) {
                attempts++;
                continue; // Try a new position
            }

            // Also check against enzymes to avoid spawning inside one
            for (const enzyme of enzymes) {
                const dist = Math.hypot(newMolecule.x - enzyme.x, newMolecule.y - enzyme.y);
                if (dist < enzyme.radius + newMolecule.radius) { // Check against outer radius
                    isOverlapping = true;
                    break;
                }
            }
            
            if (!isOverlapping) {
                molecules.push(newMolecule);
                break; // Exit the attempt loop and move to the next molecule
            }
            attempts++;
        }
    }
    entities.current.molecules = molecules;
  };

  /* -------------------------- DRAWING HELPERS -------------------------- */
  // Draws a single enzyme structure with an "active site" gap.
  const drawEnzyme = (ctx, enzyme) => {
    ctx.save();
    ctx.translate(enzyme.x, enzyme.y);
    ctx.rotate(enzyme.rotation);

    // Glow effect
    ctx.fillStyle = CFG.enzymeGlowColor;
    ctx.beginPath();
    ctx.arc(0, 0, enzyme.radius * 1.5, 0, Math.PI * 2);
    ctx.fill();
    
    // Main structure
    ctx.strokeStyle = CFG.enzymeColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    // This logic now correctly creates an opening in the hexagon.
    const activeSiteSegment = Math.floor(enzyme.activeSiteAngle / (Math.PI / 3));

    for (let i = 0; i <= 6; i++) {
        const vertexIndex = i % 6;
        
        const angle = vertexIndex * Math.PI / 3;
        const x = Math.cos(angle) * enzyme.radius;
        const y = Math.sin(angle) * enzyme.radius;
        
        const prevVertexIndex = (i - 1 + 6) % 6;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else if (prevVertexIndex === activeSiteSegment) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
    ctx.restore();
  };

  const drawMolecule = (ctx, molecule) => {
    // A molecule is a "product" if it's being released, or if it has been bound for a bit.
    const isProduct = molecule.state === 'releasing' || (molecule.state === 'bound' && molecule.boundTimer < molecule.initialBindTime / 1.5);
    const color = isProduct ? CFG.productColor : CFG.substrateColor;
    const glowColor = isProduct ? CFG.productGlowColor : CFG.substrateGlowColor;

    // Glow
    ctx.fillStyle = glowColor;
    ctx.beginPath();
    ctx.arc(molecule.x, molecule.y, molecule.radius * 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Core
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(molecule.x, molecule.y, molecule.radius, 0, Math.PI * 2);
    ctx.fill();
  };


  /* -------------------------- ANIMATION LOOP -------------------------- */
  const animate = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas.getBoundingClientRect();

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = fadeEffect.current * CFG.globalAlphaFactor;

    const { enzymes, molecules } = entities.current;

    // --- Update and Draw Enzymes ---
    enzymes.forEach(enzyme => {
      enzyme.rotation += enzyme.rotationSpeed;
      enzyme.x += enzyme.driftVx;
      enzyme.y += enzyme.driftVy;
      
      if (enzyme.x < -enzyme.radius) enzyme.x = width + enzyme.radius;
      if (enzyme.x > width + enzyme.radius) enzyme.x = -enzyme.radius;
      if (enzyme.y < -enzyme.radius) enzyme.y = height + enzyme.radius;
      if (enzyme.y > height + enzyme.radius) enzyme.y = -enzyme.radius;

      drawEnzyme(ctx, enzyme);
    });

    // --- Update and Draw Molecules (The Core Kinetic Logic) ---
    molecules.forEach(mol => {
      const enzyme = mol.targetEnzymeId ? enzymes.find(e => e.id === mol.targetEnzymeId) : null;
      
      switch (mol.state) {
        case 'wandering': {
          mol.x += mol.vx;
          mol.y += mol.vy;
          if (mol.x < 0 || mol.x > width) { mol.vx *= -1; mol.x = Math.max(0, Math.min(width, mol.x)); }
          if (mol.y < 0 || mol.y > height) { mol.vy *= -1; mol.y = Math.max(0, Math.min(height, mol.y)); }

          for (const e of enzymes) {
            if (e.boundMoleculeId) continue;
            const dist = Math.hypot(mol.x - e.x, mol.y - e.y);
            if (dist < CFG.bindingRadius) {
              mol.state = 'approaching';
              mol.targetEnzymeId = e.id;
              break;
            }
          }
          break;
        }

        case 'approaching': {
          if (!enzyme || enzyme.boundMoleculeId) {
            mol.state = 'wandering';
            mol.targetEnzymeId = null;
            break;
          }

          // The target is now INSIDE the enzyme, through the active site opening
          const angle = enzyme.rotation + enzyme.activeSiteAngle;
          const targetX = enzyme.x + Math.cos(angle) * (enzyme.radius * CFG.bindingDepthFactor);
          const targetY = enzyme.y + Math.sin(angle) * (enzyme.radius * CFG.bindingDepthFactor);
          
          const dx = targetX - mol.x;
          const dy = targetY - mol.y;
          const dist = Math.hypot(dx, dy);

          if (dist < CFG.bindingDistance) {
            mol.state = 'bound';
            mol.boundTimer = enzyme.catalyticRate;
            mol.initialBindTime = enzyme.catalyticRate; // Store for color change logic
            enzyme.boundMoleculeId = mol.id;
          } else {
            const speedFactor = CFG.moleculeSpeed * 2 * enzyme.bindingAffinity;
            mol.x += (dx / dist) * speedFactor;
            mol.y += (dy / dist) * speedFactor;
          }
          break;
        }

        case 'bound': {
          if (!enzyme) {
            mol.state = 'wandering';
            break;
          }
          // Stick to the internal binding site
          const angle = enzyme.rotation + enzyme.activeSiteAngle;
          mol.x = enzyme.x + Math.cos(angle) * (enzyme.radius * CFG.bindingDepthFactor);
          mol.y = enzyme.y + Math.sin(angle) * (enzyme.radius * CFG.bindingDepthFactor);

          mol.boundTimer--;
          if (mol.boundTimer <= 0) {
            mol.state = 'releasing';
          }
          break;
        }

        case 'releasing': {
          if (!enzyme) {
              mol.state = 'wandering';
              mol.targetEnzymeId = null;
              break;
          }
          
          // --- GENTLE RELEASE a.k.a. Diffusion ---
          const releaseAngle = enzyme.rotation + enzyme.activeSiteAngle;
          mol.vx = Math.cos(releaseAngle) * CFG.releaseSpeed;
          mol.vy = Math.sin(releaseAngle) * CFG.releaseSpeed;
          mol.x += mol.vx;
          mol.y += mol.vy;

          const distFromEnzyme = Math.hypot(mol.x - enzyme.x, mol.y - enzyme.y);
          
          if (distFromEnzyme > CFG.releaseDistance) {
             mol.state = 'wandering';
             enzyme.boundMoleculeId = null;
             mol.targetEnzymeId = null;
             mol.vx = R(-CFG.moleculeSpeed, CFG.moleculeSpeed);
             mol.vy = R(-CFG.moleculeSpeed, CFG.moleculeSpeed);
          }
          break;
        }
      }
      drawMolecule(ctx, mol);
    });

    animationFrameId.current = requestAnimationFrame(animate);
  };

  /* -------------------------- LIFECYCLE HOOK -------------------------- */
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let resizeObserver;

    const handleResize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
      spawnEntities(width, height);
    };
    
    const handleScroll = () => {
        const scrollY = window.scrollY || window.pageYOffset;
        fadeEffect.current = Math.max(0, 1 - scrollY / CFG.fadeScrollPx);
    };

    handleResize();
    handleScroll();
    animationFrameId.current = requestAnimationFrame(animate);

    resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(canvas);

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (resizeObserver) resizeObserver.disconnect();
      if(animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
    };
  }, []);

  const style = {
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100vh',
      zIndex: -1,
      display: 'block',
  };

  return <canvas ref={canvasRef} style={style} aria-hidden="true" />;
}

export default ProteinBackground;
