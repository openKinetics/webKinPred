// ProteinBackground.js
import React, { useRef, useEffect } from 'react';
import './ProteinBackground.css';

/* ---------- CONFIG ---------- */
const CFG = {
  atomCount       : 2000,
  ringCount       : 100,
  connectDist     : 60,        // px for bonds
  fadeScrollPx    : 800,
  atomSpeed       : 0.22,
  ringSpeed       : 0.01,
  ringRadiusRange : [7, 28],  // px hexagon radius
  globalAlphaFactor : 0.1,  
};
/* Basic CPK palette + letters */
const ELEM_PALETTE = [
  {sym:'C', col:'#7f8c8d'},
  {sym:'O', col:'#e74c3c'},
  {sym:'N', col:'#3498db'},
  {sym:'P', col:'#f1c40f'},
  {sym:'S', col:'#f39c12'}
];

function ProteinBackground() {
  const canvasRef = useRef(null);
  const atoms     = useRef([]);
  const rings     = useRef([]);
  const fade      = useRef(1);
  const RAF       = useRef(null);

  const R = (a,b)=>Math.random()*(b-a)+a;

  /* ---------- spawn atoms & rings ---------- */
  const spawnEntities = (w,h)=>{
    atoms.current = Array.from({length:CFG.atomCount},()=>({
      x:R(0,w),y:R(0,h),r:R(1.5,3.5),
      vx:R(-CFG.atomSpeed,CFG.atomSpeed),
      vy:R(-CFG.atomSpeed,CFG.atomSpeed),
      ...ELEM_PALETTE[Math.floor(Math.random()*ELEM_PALETTE.length)]
    }));
    rings.current = Array.from({length:CFG.ringCount},()=>({
      cx:R(0,w),cy:R(0,h),
      r:R(...CFG.ringRadiusRange),
      rot:R(0,Math.PI*2),
      vr:R(-CFG.ringSpeed,CFG.ringSpeed)
    }));
  };

  /* ---------- drawing helpers ---------- */
  const drawAtom = (ctx,a)=>{
    // glow
    ctx.fillStyle=`${a.col}22`;
    ctx.beginPath();ctx.arc(a.x,a.y,a.r*2.2,0,2*Math.PI);ctx.fill();
    // core
    ctx.fillStyle=a.col;
    ctx.beginPath();ctx.arc(a.x,a.y,a.r,0,2*Math.PI);ctx.fill();
    // letter
    ctx.fillStyle='#ffffff';
    ctx.font=`${a.r*2.2}px Arial`;
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(a.sym,a.x,a.y);
  };

  const drawBond = (ctx,a,b,d)=>{
    const alpha=1-d/CFG.connectDist;
    const thickness=d<CFG.connectDist*0.6?1.1:d<CFG.connectDist*0.45?2.2:3.3; // single/double/triple
    ctx.strokeStyle=`rgba(72,180,224,${alpha*0.25})`;
    ctx.lineWidth=thickness;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  };

  const drawRing = (ctx,r)=>{
    const {cx,cy}=r;
    ctx.strokeStyle='rgba(255,255,255,.25)';
    ctx.lineWidth=1.4;
    ctx.beginPath();
    for(let i=0;i<6;i++){
      const ang=r.rot+i*Math.PI/3;
      const x=cx+Math.cos(ang)*r.r;
      const y=cy+Math.sin(ang)*r.r;
      ctx[i?'lineTo':'moveTo'](x,y);
    }
    ctx.closePath();ctx.stroke();
    // draw small atoms at corners
    for(let i=0;i<6;i++){
      const ang=r.rot+i*Math.PI/3;
      const x=cx+Math.cos(ang)*r.r;
      const y=cy+Math.sin(ang)*r.r;
      ctx.fillStyle='#7f8c8d';
      ctx.beginPath();ctx.arc(x,y,2.5,0,2*Math.PI);ctx.fill();
    }
  };

  /* ---------- animation ---------- */
  const animate=()=>{
    const cvs=canvasRef.current;if(!cvs)return;
    const ctx=cvs.getContext('2d');
    const {width:w,height:h}=cvs.getBoundingClientRect();
    ctx.clearRect(0,0,w,h);
    ctx.globalAlpha = fade.current * CFG.globalAlphaFactor;

    /* rings first (behind) */
    rings.current.forEach(r=>{
      r.rot+=r.vr;
      r.cx+=r.vr*20; // gentle drift
      r.cy+=r.vr*14;
      if(r.cx<-r.r) r.cx=w+r.r;
      if(r.cx>w+r.r)r.cx=-r.r;
      if(r.cy<-r.r) r.cy=h+r.r;
      if(r.cy>h+r.r)r.cy=-r.r;
      drawRing(ctx,r);
    });

    /* atoms */
    atoms.current.forEach(a=>{
      a.x+=a.vx;a.y+=a.vy;
      if(a.x<0||a.x>w)a.vx*=-1;
      if(a.y<0||a.y>h)a.vy*=-1;
      drawAtom(ctx,a);
    });

    /* bonds */
    for(let i=0;i<atoms.current.length;i++){
      const A=atoms.current[i];
      for(let j=i+1;j<atoms.current.length;j++){
        const B=atoms.current[j];
        const dx=A.x-B.x;if(Math.abs(dx)>CFG.connectDist)continue;
        const dy=A.y-B.y;if(Math.abs(dy)>CFG.connectDist)continue;
        const d=Math.hypot(dx,dy);
        if(d<CFG.connectDist) drawBond(ctx,A,B,d);
      }
    }

    RAF.current=requestAnimationFrame(animate);
  };

  /* ---------- effect ---------- */
  useEffect(()=>{
    const cvs=canvasRef.current,ctx=cvs.getContext('2d');
    const resize=()=>{const {width:w,height:h}=cvs.getBoundingClientRect();
      const s=window.devicePixelRatio||1;ctx.setTransform(1,0,0,1,0,0);
      cvs.width=w*s;cvs.height=h*s;ctx.scale(s,s);spawnEntities(w,h);};
    resize();

    RAF.current=requestAnimationFrame(animate);

    const ro=new ResizeObserver(()=>resize());ro.observe(cvs);
    const scroll=()=>{fade.current=Math.max(0,1-(window.scrollY||0)/CFG.fadeScrollPx);};
    window.addEventListener('scroll',scroll);scroll();

    /* cleanup */
    return()=>{window.removeEventListener('scroll',scroll);ro.disconnect();cancelAnimationFrame(RAF.current);};
  },[]);

  return <canvas ref={canvasRef} className="protein-bg-canvas" aria-hidden="true"/>;
}

export default ProteinBackground;
