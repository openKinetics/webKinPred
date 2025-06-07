import React, { useRef, useEffect } from 'react';
import './ProteinBackground.css';

/* ── CONFIG ─────────────────────────────────────────────────── */
const CONFIG = {
  moleculeCount  : 300,               // # cyan dots
  enzymeCount    : 0,
  connectDist    : 55,              // px – molecule-molecule link
  fadeScrollPx   : 1250,              // px until fully faded
  bindRadius     : 35,               // px – capture zone
  bindDurMsRange : [3000, 6000],     // ms – min/max binding
  enzymeSpeed    : 0.15,             // px / frame drift
  rotSpeedRad    : 0.004,            // rad / frame rotation
};
/* ───────────────────────────────────────────────────────────── */

function ProteinBackground() {
  const canvasRef   = useRef(null);
  const opacityRef  = useRef(1);
  const resRef      = useRef([]);  // molecules
  const enzRef      = useRef([]);  // enzymes
  const RAF         = useRef(null);

  const rand = (min, max) => Math.random() * (max - min) + min;

  /* Hi-DPI canvas resize */
  const resizeCanvas = (ctx) => {
    const { width, height } = ctx.canvas.getBoundingClientRect();
    const scl = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.canvas.width  = width  * scl;
    ctx.canvas.height = height * scl;
    ctx.scale(scl, scl);
  };

  /* Init entities */
  const initEntities = (w, h) => {
    resRef.current = Array.from({ length: CONFIG.moleculeCount }, () => ({
      x: rand(0, w),
      y: rand(0, h),
      r: rand(1.6, 3.6),
      vx: rand(-0.3, 0.3),
      vy: rand(-0.3, 0.3),
      pulse: rand(0, Math.PI * 2),
      boundTo: null,
      offset : null,
      releaseT: null,
    }));

    enzRef.current = Array.from({ length: CONFIG.enzymeCount }, () => ({
      x   : rand(0, w),
      y   : rand(0, h),
      r   : rand(15, 40),
      vx  : rand(-CONFIG.enzymeSpeed, CONFIG.enzymeSpeed),
      vy  : rand(-CONFIG.enzymeSpeed, CONFIG.enzymeSpeed),
      rot : rand(0, Math.PI * 2),
      mouthAngle: rand(25, 90) * (Math.PI / 180),
    }));
  };

  /* Draw a Pac-Man enzyme */
  const drawEnzyme = (ctx, e) => {
    const mouth = e.mouthAngle;
    const start = e.rot + mouth / 2;
    const end   = e.rot + 2 * Math.PI - mouth / 2;

    const g = ctx.createRadialGradient(
        e.x, e.y, e.r * 0.3,
        e.x, e.y, e.r
      );
      g.addColorStop(0, 'rgba(90,160,255,0.4)');
      g.addColorStop(1, 'rgba(90,160,255,0.05)');
      

    ctx.fillStyle = g;
    ctx.strokeStyle = 'rgba(72,180,224,0.65)';
    ctx.lineWidth = 1.6;

    ctx.beginPath();
    ctx.moveTo(e.x, e.y);          // centre
    ctx.arc(e.x, e.y, e.r, start, end, false);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  };

  /* MAIN LOOP */
  const animate = (t) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas.getBoundingClientRect();

    ctx.clearRect(0, 0, width, height);
    ctx.globalAlpha = opacityRef.current;

    /* Update & draw enzymes */
    enzRef.current.forEach((e) => {
      e.x += e.vx;
      e.y += e.vy;
      e.rot += CONFIG.rotSpeedRad;

      if (e.x - e.r < 0 || e.x + e.r > width)  e.vx *= -1;
      if (e.y - e.r < 0 || e.y + e.r > height) e.vy *= -1;

      drawEnzyme(ctx, e);
    });

    /* Update molecules */
    resRef.current.forEach((r) => {
      if (r.boundTo !== null) {
        const e = enzRef.current[r.boundTo];
        r.x = e.x + r.offset.dx;
        r.y = e.y + r.offset.dy;
        if (t >= r.releaseT) r.boundTo = null;
      } else {
        r.x += r.vx;
        r.y += r.vy;
        if (r.x < 0 || r.x > width)  r.vx *= -1;
        if (r.y < 0 || r.y > height) r.vy *= -1;

        // potential binding
        enzRef.current.forEach((e, i) => {
          const dx = r.x - e.x, dy = r.y - e.y;
          const d  = Math.hypot(dx, dy);
          if (d < e.r + CONFIG.bindRadius && Math.random() < 0.002) {
            r.boundTo = i;
            r.offset  = { dx, dy };
            r.releaseT = t + rand(...CONFIG.bindDurMsRange);
          }
        });
      }

      r.pulse += 0.045;
      const pr = r.r + Math.sin(r.pulse) * 0.6;
      ctx.fillStyle = r.boundTo !== null ? '#f6b26b' : '#48b4e0';
      ctx.beginPath();
      ctx.arc(r.x, r.y, pr, 0, Math.PI * 2);
      ctx.fill();
    });

    /* molecule–molecule links */
    for (let i = 0; i < resRef.current.length; i++) {
      for (let j = i + 1; j < resRef.current.length; j++) {
        const a = resRef.current[i], b = resRef.current[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d  = Math.hypot(dx, dy);
        if (d < CONFIG.connectDist) {
          const alpha = 1 - d / CONFIG.connectDist;
          ctx.strokeStyle = `rgba(72,180,224,${alpha * 0.25})`;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    RAF.current = requestAnimationFrame(animate);
  };

  /* INIT & CLEANUP */
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');
    resizeCanvas(ctx);
    initEntities(canvas.clientWidth, canvas.clientHeight);
    RAF.current = requestAnimationFrame(animate);
    const ro = new ResizeObserver((entries) => {
        const { width, height } = entries[0].contentRect;
        resizeCanvas(ctx);
        initEntities(width, height);    // redistribute particles into new area
    });
    ro.observe(canvas);
    /* align under navbar */
    const positionBelowNav = () => {
      const nav = document.querySelector('.custom-navbar');
      canvas.style.top = `${nav ? nav.offsetHeight : 0}px`;
    };
    positionBelowNav();

    const onResize = () => {
      resizeCanvas(ctx);
      initEntities(canvas.clientWidth, canvas.clientHeight);
      positionBelowNav();
    };
    const onScroll = () => {
      const y = window.scrollY || window.pageYOffset;
      opacityRef.current = Math.max(0, 1 - y / CONFIG.fadeScrollPx);
    };

    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onScroll);

    return () => {
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onScroll);
      cancelAnimationFrame(RAF.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="protein-bg-canvas"
      aria-hidden="true"
    />
  );
}

export default ProteinBackground;
