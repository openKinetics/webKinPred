import React, { useRef, useEffect } from 'react';

/* -------------------------- CONFIGURATION -------------------------- */
// This object holds all the tweakable parameters for the evolved simulation.
const CFG = {
    enzymeCount: 15,           // Fewer, larger enzymes for a cleaner look
    moleculeCount: 125,        // Reduced molecule count to complement fewer enzymes
    
    // --- Visuals & Colors ---
    enzymeColor: 'rgba(100, 120, 150, 0.7)',
    substrateColor: '#3498db',
    productColor: '#e74c3c',
    enzymeGlowColor: 'rgba(100, 120, 150, 0.1)',
    substrateGlowColor: 'rgba(52, 152, 219, 0.15)',
    productGlowColor: 'rgba(231, 76, 60, 0.2)',
    catalysisPulseColor: 'rgba(255, 255, 255, 0.5)',

    // --- Physics & Behavior ---
    moleculeSpeed: 0.17,
    enzymeDriftSpeed: 0.05,
    enzymeRotationSpeed: 0.002,
    
    // --- Kinetics & Interaction Parameters ---
    bindingRadius: 150,        // (px) How close a substrate must be to be "attracted"
    bindingDistance: 2.0,      // (px) How close a substrate must be to "dock"
    bindingDepthFactor: 0.25,  // How deep inside the enzyme the molecule goes
    releaseDistance: 200,       // (px) How far a product must travel before enzyme is free
    releaseSpeed: 0.17,         // The initial "push" speed for product release
    affinityRange: [0.3, 1.2], // Range for KM. Higher = stronger attraction & faster orientation.
    kcatRange: [250, 2000],    // Range for kcat. Duration in frames.
    
    inducedFitFlex: 0.3,      // How much the enzyme "pinches" (0=none, 1=fully closed)
    flexSpeed: 0.05,           // How fast the enzyme flexes and relaxes
    orientationSpeedFactor: 0.04,// How fast the substrate orients itself to bind
    orientationTolerance: 0.1, // (radians) How precisely aligned substrate must be to bind
    enzymeRecoilStrength: 0.2, // How much the enzyme "jiggles" upon product release
    
    // --- Spawning & Global Effects ---
    minEnzymeDistance: 150,     // The minimum distance between the centers of two enzymes
    globalAlphaFactor: 0.35,
    fadeScrollPx: 650,
};

// Helper for linear interpolation (lerping)
const lerp = (a, b, t) => a + (b - a) * t;
const R = (min, max) => Math.random() * (max - min) + min;

/**
 * An evolved React component that renders a beautiful, lightweight, and scientifically-themed
 * animated background representing enzyme kinetics with advanced visual features like
 * induced fit, substrate orientation, and physical recoil.
 */
function ProteinBackground() {
  const canvasRef = useRef(null);
  const entities = useRef({ enzymes: [], molecules: [] });
  const fadeEffect = useRef(1);
  const animationFrameId = useRef(null);

  /* -------------------------- ENTITY SPAWNING (CORRECTED) -------------------------- */
  const spawnEntities = (width, height) => {
    const enzymes = [];
    const maxPlacementAttempts = 50; // Safety net to prevent infinite loops

    for (let i = 0; i < CFG.enzymeCount; i++) {
        let attempts = 0;
        let placed = false;
        
        while (!placed && attempts < maxPlacementAttempts) {
            const candidate = {
                id: Math.random(),
                x: R(0, width),
                y: R(0, height),
                radius: R(20, 40),
            };

            let isTooClose = false;
            // Check distance against all previously placed enzymes
            for (const placedEnzyme of enzymes) {
                const dist = Math.hypot(candidate.x - placedEnzyme.x, candidate.y - placedEnzyme.y);
                // The distance check now uses the config value
                if (dist < CFG.minEnzymeDistance) {
                    isTooClose = true;
                    break;
                }
            }

            // If it's not too close to any other enzyme, place it
            if (!isTooClose) {
                enzymes.push({
                    ...candidate,
                    rotation: R(0, Math.PI * 2),
                    rotationSpeed: R(-CFG.enzymeRotationSpeed, CFG.enzymeRotationSpeed),
                    driftVx: R(-CFG.enzymeDriftSpeed, CFG.enzymeDriftSpeed),
                    driftVy: R(-CFG.enzymeDriftSpeed, CFG.enzymeDriftSpeed),
                    bindingAffinity: R(...CFG.affinityRange),
                    catalyticRate: R(...CFG.kcatRange),
                    activeSiteAngle: Math.floor(R(0, 6)) * (Math.PI / 3),
                    boundMoleculeId: null,
                    flex: 0,
                    targetFlex: 0,
                    recoilVx: 0,
                    recoilVy: 0,
                });
                placed = true;
            }
            attempts++;
        }
        if (attempts >= maxPlacementAttempts) {
             console.warn(`Could not place an enzyme after ${maxPlacementAttempts} attempts. Check density settings.`);
        }
    }
    entities.current.enzymes = enzymes;

    const molecules = [];
    for (let i = 0; i < CFG.moleculeCount; i++) {
        molecules.push({
            id: Math.random(),
            x: R(0, width),
            y: R(0, height),
            vx: R(-CFG.moleculeSpeed, CFG.moleculeSpeed),
            vy: R(-CFG.moleculeSpeed, CFG.moleculeSpeed),
            radius: R(2, 4),
            state: 'wandering',
            targetEnzymeId: null,
            boundTimer: 0,
            shape: 'substrate',
            rotation: R(0, Math.PI * 2),
            targetRotation: 0,
            color: CFG.substrateColor,
        });
    }
    entities.current.molecules = molecules;
  };

  /* -------------------------- DRAWING HELPERS -------------------------- */
  
  const drawEnzyme = (ctx, enzyme) => {
    ctx.save();
    ctx.translate(enzyme.x, enzyme.y);
    ctx.rotate(enzyme.rotation);

    ctx.fillStyle = CFG.enzymeGlowColor;
    ctx.beginPath();
    ctx.arc(0, 0, enzyme.radius * 1.2, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.strokeStyle = CFG.enzymeColor;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    
    const activeSiteSegment = Math.floor(enzyme.activeSiteAngle / (Math.PI / 3));
    const flexAngleOffset = (Math.PI / 3) * CFG.inducedFitFlex * enzyme.flex;

    for (let i = 0; i <= 6; i++) {
        const vertexIndex = i % 6;
        let angle = vertexIndex * Math.PI / 3;
        
        if (vertexIndex === activeSiteSegment) {
            angle += flexAngleOffset;
        } else if (vertexIndex === (activeSiteSegment + 1) % 6) {
            angle -= flexAngleOffset;
        }

        const x = Math.cos(angle) * enzyme.radius;
        const y = Math.sin(angle) * enzyme.radius;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else if (vertexIndex !== (activeSiteSegment + 1) % 6) {
            ctx.lineTo(x, y);
        } else {
            ctx.moveTo(x, y);
        }
    }
    
    ctx.stroke();
    ctx.restore();
  };
  
  const drawMolecule = (ctx, mol) => {
    ctx.save();
    ctx.translate(mol.x, mol.y);
    
    const glowColor = mol.shape === 'product' ? CFG.productGlowColor : CFG.substrateGlowColor;
    ctx.fillStyle = glowColor;
    ctx.beginPath();
    ctx.arc(0, 0, mol.radius * 1.7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = mol.color;
    ctx.beginPath();

    if (mol.shape === 'substrate') {
        ctx.rotate(mol.rotation);
        ctx.moveTo(mol.radius, 0);
        ctx.arc(0, 0, mol.radius, -Math.PI / 3, Math.PI / 3, false);
        ctx.closePath();
    } else {
        ctx.arc(0, 0, mol.radius, 0, Math.PI * 2);
    }
    
    ctx.fill();

    if (mol.state === 'bound' && mol.boundTimer > 0) {
        const pulseProgress = (mol.initialBindTime - mol.boundTimer) / mol.initialBindTime;
        const pulseAlpha = Math.sin(pulseProgress * Math.PI * 4) * 0.5 + 0.5;
        if (pulseAlpha > 0.1) {
             ctx.fillStyle = `rgba(255, 255, 255, ${pulseAlpha * 0.5})`;
             ctx.beginPath();
             ctx.arc(0, 0, mol.radius * 1.2, 0, Math.PI * 2);
             ctx.fill();
        }
    }

    ctx.restore();
  };

  /* -------------------------- ANIMATION LOOP -------------------------- */
  const animate = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas.getBoundingClientRect();

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);   // identity matrix
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.restore();

    ctx.globalAlpha = fadeEffect.current * CFG.globalAlphaFactor;

    const { enzymes, molecules } = entities.current;

    // --- Update and Draw Enzymes ---
    enzymes.forEach(enzyme => {
      enzyme.rotation += enzyme.rotationSpeed;
      enzyme.recoilVx *= 0.95;
      enzyme.recoilVy *= 0.95;
      enzyme.x += enzyme.driftVx + enzyme.recoilVx;
      enzyme.y += enzyme.driftVy + enzyme.recoilVy;
      
      if (enzyme.x < -enzyme.radius) enzyme.x = width + enzyme.radius;
      if (enzyme.x > width + enzyme.radius) enzyme.x = -enzyme.radius;
      if (enzyme.y < -enzyme.radius) enzyme.y = height + enzyme.radius;
      if (enzyme.y > height + enzyme.radius) enzyme.y = -enzyme.radius;

      enzyme.flex = lerp(enzyme.flex, enzyme.targetFlex, CFG.flexSpeed);

      drawEnzyme(ctx, enzyme);
    });

    // --- Update and Draw Molecules ---
    molecules.forEach(mol => {
      const enzyme = mol.targetEnzymeId ? enzymes.find(e => e.id === mol.targetEnzymeId) : null;
      
      switch (mol.state) {
        case 'wandering': {
          mol.x += mol.vx;
          mol.y += mol.vy;
          mol.rotation += (Math.random() - 0.5) * 0.1;
          if (mol.x < 0 || mol.x > width) { mol.vx *= -1; mol.x = Math.max(0, Math.min(width, mol.x)); }
          if (mol.y < 0 || mol.y > height) { mol.vy *= -1; mol.y = Math.max(0, Math.min(height, mol.y)); }

          for (const e of enzymes) {
            if (e.boundMoleculeId) continue;
            const dist = Math.hypot(mol.x - e.x, mol.y - e.y);
            if (dist < CFG.bindingRadius) {
              mol.state = 'approaching';
              mol.targetEnzymeId = e.id;
              mol.targetRotation = e.rotation + e.activeSiteAngle + Math.PI;
              break;
            }
          }
          break;
        }

        case 'approaching': {
          if (!enzyme || enzyme.boundMoleculeId) {
            mol.state = 'wandering'; mol.targetEnzymeId = null; break;
          }
          
          const angleToSite = enzyme.rotation + enzyme.activeSiteAngle;
          const targetX = enzyme.x + Math.cos(angleToSite) * (enzyme.radius * 0.8);
          const targetY = enzyme.y + Math.sin(angleToSite) * (enzyme.radius * 0.8);
          
          const dx = targetX - mol.x;
          const dy = targetY - mol.y;
          const dist = Math.hypot(dx, dy);

          let angleDiff = mol.targetRotation - mol.rotation;
          while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
          while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
          mol.rotation += angleDiff * CFG.orientationSpeedFactor * enzyme.bindingAffinity;

          if (dist < CFG.bindingDistance && Math.abs(angleDiff) < CFG.orientationTolerance) {
            mol.state = 'bound';
            mol.boundTimer = enzyme.catalyticRate;
            mol.initialBindTime = enzyme.catalyticRate;
            enzyme.boundMoleculeId = mol.id;
            enzyme.targetFlex = 1;
          } else {
            const speedFactor = CFG.moleculeSpeed * 2 * enzyme.bindingAffinity;
            mol.x += (dx / dist) * speedFactor;
            mol.y += (dy / dist) * speedFactor;
          }
          break;
        }

        case 'bound': {
          if (!enzyme) { mol.state = 'wandering'; break; }
          const angle = enzyme.rotation + enzyme.activeSiteAngle;
          mol.x = enzyme.x + Math.cos(angle) * (enzyme.radius * CFG.bindingDepthFactor);
          mol.y = enzyme.y + Math.sin(angle) * (enzyme.radius * CFG.bindingDepthFactor);
          mol.rotation = enzyme.rotation + enzyme.activeSiteAngle + Math.PI;
          
          mol.boundTimer--;
          
          const progress = 1 - (mol.boundTimer / mol.initialBindTime);
          const r = Math.floor(lerp(parseInt(CFG.substrateColor.slice(1,3), 16), parseInt(CFG.productColor.slice(1,3), 16), progress));
          const g = Math.floor(lerp(parseInt(CFG.substrateColor.slice(3,5), 16), parseInt(CFG.productColor.slice(3,5), 16), progress));
          const b = Math.floor(lerp(parseInt(CFG.substrateColor.slice(5,7), 16), parseInt(CFG.productColor.slice(5,7), 16), progress));
          mol.color = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

          if (mol.boundTimer <= 0) {
            mol.state = 'releasing';
            mol.shape = 'product';
            enzyme.targetFlex = 0;

            const recoilAngle = angle + Math.PI;
            enzyme.recoilVx += Math.cos(recoilAngle) * CFG.enzymeRecoilStrength;
            enzyme.recoilVy += Math.sin(recoilAngle) * CFG.enzymeRecoilStrength;
          }
          break;
        }

        case 'releasing': {
          if (!enzyme) {
              mol.state = 'wandering';
              mol.targetEnzymeId = null;
              break;
          }

          const dx = mol.x - enzyme.x;
          const dy = mol.y - enzyme.y;
          const dist = Math.hypot(dx, dy) || 1;
          
          const baseVx = (dx / dist) * CFG.moleculeSpeed * 0.8;
          const baseVy = (dy / dist) * CFG.moleculeSpeed * 0.8;

          const diffusionFactor = 0.7; 
          mol.vx = baseVx + R(-CFG.moleculeSpeed * diffusionFactor, CFG.moleculeSpeed * diffusionFactor);
          mol.vy = baseVy + R(-CFG.moleculeSpeed * diffusionFactor, CFG.moleculeSpeed * diffusionFactor);

          mol.x += mol.vx;
          mol.y += mol.vy;
          
          const distFromEnzyme = Math.hypot(mol.x - enzyme.x, mol.y - enzyme.y);
          
          if (distFromEnzyme > CFG.releaseDistance) {
              enzyme.boundMoleculeId = null;
          }
          
          if (distFromEnzyme > CFG.bindingRadius * 1.5) {
              mol.state = 'wandering';
              mol.targetEnzymeId = null;
              mol.shape = 'substrate';
              mol.color = CFG.substrateColor;
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
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let resizeObserver;

    const handleResize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0); // start from scratch
      ctx.scale(dpr, dpr);                // apply DPR exactly once
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
    if(canvas) {
        resizeObserver.observe(canvas);
    }
    
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