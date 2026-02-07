import { useState, useEffect, useRef } from "react";

const TARGET_URL = "https://docs.google.com/document/d/11_515-c2cktODaiLnsGgjxEtU874aeLQDkR1gdVa2jI/edit?usp=sharing";


const REF = {
  grid: [[.188,.312,.438,.375,.562,.125,0,0],[.188,.438,.75,.812,.5,.062,0,0],[0,.188,1,1,.375,0,0,0],[0,0,.562,1,.625,.5,0,0],[0,0,.438,.812,.938,.188,0,0],[0,.375,.438,.688,.625,.5,.438,.125],[.125,.375,.625,.625,.438,.688,.625,.062],[0,0,.5,.375,.25,.062,.188,0]],
  vProfile: [.062,.211,.594,.711,.539,.266,.156,.023],
  hProfile: [.25,.344,.32,.336,.297,.398,.445,.172],
  inkDensity: .32, aspectRatio: .346,
  quadrants: [.453,.172,.336,.32], centroid: [.444,.488],
  binaryHash: "00084000025ae0004954e40065d5c80032ddd8001adb9000095ba00005ff200002ffc00001ffc00001ff800000ff8000007f8000003f9c00003fff00003fdc00003dd000003ef800002ef100006ff10001adec80022df840066d3be4039a13c2321b3fe407de19c0087e1fe00871e1300031b130007300100073000000018000",
};

// ===================== IMAGE PROCESSING =====================

function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < gray.length; i++) { const j = i * 4; gray[i] = data[j] * .299 + data[j+1] * .587 + data[j+2] * .114; }
  return gray;
}

function otsu(gray) {
  const hist = new Int32Array(256);
  for (let i = 0; i < gray.length; i++) hist[Math.min(255, Math.max(0, Math.round(gray[i])))]++;
  const total = gray.length;
  let sumAll = 0;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  let sumB = 0, wB = 0, best = 0, thresh = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (wB === 0) continue;
    const wF = total - wB; if (wF === 0) break;
    sumB += t * hist[t];
    const diff = sumB / wB - (sumAll - sumB) / wF;
    const v = wB * wF * diff * diff;
    if (v > best) { best = v; thresh = t; }
  }
  return thresh;
}

function connectedComponents(binary, w, h) {
  const labels = new Int32Array(w * h);
  const parent = [0];
  let nextLabel = 1;
  function find(x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
  function union(a, b) { a = find(a); b = find(b); if (a !== b) parent[Math.max(a, b)] = Math.min(a, b); }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!binary[idx]) continue;
      const up = y > 0 && binary[(y-1)*w+x] ? labels[(y-1)*w+x] : 0;
      const left = x > 0 && binary[y*w+x-1] ? labels[y*w+x-1] : 0;
      if (up && left) { labels[idx] = Math.min(up, left); union(up, left); }
      else if (up) labels[idx] = up;
      else if (left) labels[idx] = left;
      else { labels[idx] = nextLabel; parent.push(nextLabel); nextLabel++; }
    }
  }
  const sizes = new Map();
  for (let i = 0; i < labels.length; i++) { if (labels[i]) { labels[i] = find(labels[i]); sizes.set(labels[i], (sizes.get(labels[i]) || 0) + 1); } }
  let largestLabel = 0, largestSize = 0;
  for (const [label, size] of sizes) { if (size > largestSize) { largestSize = size; largestLabel = label; } }
  return { labels, largestLabel };
}

function extractLargestCluster(sourceCanvas) {
  const maxDim = 512;
  let sw = sourceCanvas.width, sh = sourceCanvas.height;
  const scale = Math.min(1, maxDim / Math.max(sw, sh));
  const w = Math.round(sw * scale), h = Math.round(sh * scale);
  const wc = document.createElement("canvas"); wc.width = w; wc.height = h;
  const ctx = wc.getContext("2d"); ctx.drawImage(sourceCanvas, 0, 0, w, h);
  const imgData = ctx.getImageData(0, 0, w, h);
  const gray = toGrayscale(imgData);
  const thresh = otsu(gray);
  const binary = new Uint8Array(w * h);
  for (let i = 0; i < gray.length; i++) binary[i] = gray[i] < thresh ? 1 : 0;
  let inkCount = 0;
  for (let i = 0; i < binary.length; i++) inkCount += binary[i];
  if (inkCount / binary.length < 0.005 || inkCount / binary.length > 0.85) return null;
  const cc = connectedComponents(binary, w, h);
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < cc.labels.length; i++) mask[i] = cc.labels[i] === cc.largestLabel ? 1 : 0;
  let rmin = h, rmax = 0, cmin = w, cmax = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { if (mask[y*w+x]) { if (y<rmin) rmin=y; if (y>rmax) rmax=y; if (x<cmin) cmin=x; if (x>cmax) cmax=x; } }
  const pad = Math.max(2, Math.round(Math.min(w,h)*.015));
  rmin = Math.max(0, rmin-pad); rmax = Math.min(h-1, rmax+pad);
  cmin = Math.max(0, cmin-pad); cmax = Math.min(w-1, cmax+pad);
  const cw = cmax-cmin+1, ch = rmax-rmin+1;
  const S = 32;
  const norm = new Float32Array(S*S);
  const sx = cw/S, sy = ch/S;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const x0 = Math.floor(x*sx)+cmin, x1 = Math.min(cmax, Math.floor((x+1)*sx)+cmin);
    const y0 = Math.floor(y*sy)+rmin, y1 = Math.min(rmax, Math.floor((y+1)*sy)+rmin);
    let s = 0, c = 0;
    for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) { s += mask[yy*w+xx]; c++; }
    norm[y*S+x] = c > 0 ? s/c : 0;
  }
  const grid = [];
  for (let gy = 0; gy < 8; gy++) { const row = []; for (let gx = 0; gx < 8; gx++) { let s=0,c=0; for (let y=gy*4;y<(gy+1)*4;y++) for (let x=gx*4;x<(gx+1)*4;x++){s+=norm[y*S+x];c++;} row.push(s/c); } grid.push(row); }
  const vProfile = [], hProfile = [];
  for (let gx=0;gx<8;gx++){let s=0,c=0;for(let y=0;y<S;y++)for(let x=gx*4;x<(gx+1)*4;x++){s+=norm[y*S+x];c++;}vProfile.push(s/c);}
  for (let gy=0;gy<8;gy++){let s=0,c=0;for(let y=gy*4;y<(gy+1)*4;y++)for(let x=0;x<S;x++){s+=norm[y*S+x];c++;}hProfile.push(s/c);}
  let inkDensity=0; for(let i=0;i<norm.length;i++) inkDensity+=norm[i]; inkDensity/=norm.length;
  const aspectRatio = cw/ch;
  const q=[0,0,0,0],qc=[0,0,0,0];
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){const qi=(y<16?0:2)+(x<16?0:1);q[qi]+=norm[y*S+x];qc[qi]++;}
  const quadrants=q.map((v,i)=>v/qc[i]);
  let cxS=0,cyS=0,cT=0;
  for(let y=0;y<S;y++)for(let x=0;x<S;x++){const v=norm[y*S+x];cxS+=x*v;cyS+=y*v;cT+=v;}
  const centroid=cT>0?[cxS/cT/S,cyS/cT/S]:[.5,.5];
  let bits="";for(let i=0;i<S*S;i++)bits+=norm[i]>.5?"1":"0";
  let binaryHash="";for(let i=0;i<bits.length;i+=4)binaryHash+=parseInt(bits.substring(i,i+4),2).toString(16);
  return {grid,vProfile,hProfile,inkDensity,aspectRatio,quadrants,centroid,binaryHash};
}

function ncc(a,b){const n=a.length;let mA=0,mB=0;for(let i=0;i<n;i++){mA+=a[i];mB+=b[i];}mA/=n;mB/=n;let num=0,dA=0,dB=0;for(let i=0;i<n;i++){const da=a[i]-mA,db=b[i]-mB;num+=da*db;dA+=da*da;dB+=db*db;}return(dA===0||dB===0)?0:num/Math.sqrt(dA*dB);}
function cosine(a,b){let dot=0,mA=0,mB=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];mA+=a[i]*a[i];mB+=b[i]*b[i];}return(mA===0||mB===0)?0:dot/Math.sqrt(mA*mB);}
function hammingDistance(h1,h2){if(h1.length!==h2.length)return 1;let diff=0,total=0;for(let i=0;i<h1.length;i++){let xor=parseInt(h1[i],16)^parseInt(h2[i],16);while(xor){diff+=xor&1;xor>>=1;}total+=4;}return diff/total;}

function matchScore(features) {
  if (!features) return 0;
  const rg=REF.grid.flat(),cg=features.grid.flat();
  const gridScore=Math.max(0,ncc(rg,cg))*.6+cosine(rg,cg)*.4;
  const vScore=Math.max(0,ncc(REF.vProfile,features.vProfile))*.6+cosine(REF.vProfile,features.vProfile)*.4;
  const hScore=Math.max(0,ncc(REF.hProfile,features.hProfile))*.6+cosine(REF.hProfile,features.hProfile)*.4;
  const inkScore=Math.max(0,1-Math.abs(REF.inkDensity-features.inkDensity)*3.5);
  const arScore=Math.max(0,1-Math.abs(REF.aspectRatio-features.aspectRatio)*2.5);
  const qScore=Math.max(0,ncc(REF.quadrants,features.quadrants));
  const centDist=Math.sqrt(Math.pow(REF.centroid[0]-features.centroid[0],2)+Math.pow(REF.centroid[1]-features.centroid[1],2));
  const centScore=Math.max(0,1-centDist*5);
  const hashDist=hammingDistance(REF.binaryHash,features.binaryHash);
  const hashScore=Math.max(0,1-hashDist*2.5);
  return gridScore*.25+vScore*.10+hScore*.10+hashScore*.20+inkScore*.10+arScore*.08+qScore*.10+centScore*.07;
}

// Lavender bouquet SVG as a faint background
const LavenderBG = () => (
  <svg style={{ position: "fixed", bottom: -40, right: -30, width: 520, height: 700, opacity: 0.055, pointerEvents: "none", zIndex: 0 }} viewBox="0 0 520 700" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g stroke="#8b6f9e" strokeWidth="1.8" fill="none" opacity="0.9">
      {/* Stems */}
      <path d="M260 690 C258 580 250 460 245 350 Q242 280 248 200 Q250 160 255 120" strokeWidth="2.2"/>
      <path d="M260 690 C265 590 275 480 285 380 Q295 300 290 230 Q287 180 280 140" strokeWidth="2"/>
      <path d="M260 690 C252 600 235 500 220 400 Q210 330 215 260 Q218 210 225 170" strokeWidth="2"/>
      <path d="M260 690 C270 610 290 510 305 410 Q315 340 308 270 Q303 220 295 175" strokeWidth="1.8"/>
      <path d="M260 690 C245 605 222 490 205 390 Q195 320 202 250 Q206 200 215 155" strokeWidth="1.6"/>
      <path d="M260 690 C275 615 300 500 320 400 Q332 330 322 260 Q316 210 305 165" strokeWidth="1.6"/>
      <path d="M260 690 C240 610 210 480 195 370 Q185 300 195 235 Q200 185 210 145" strokeWidth="1.4"/>
    </g>
    <g fill="#b396c7" opacity="0.5">
      {/* Flower buds - stem 1 */}
      {[120,135,148,160,172,183,193].map((y,i) => <ellipse key={`a${i}`} cx={252+Math.sin(i*1.2)*4} cy={y} rx={5.5} ry={4} transform={`rotate(${-5+i*2} ${252} ${y})`}/>)}
      {/* Stem 2 */}
      {[140,154,166,178,190,200,210].map((y,i) => <ellipse key={`b${i}`} cx={283+Math.sin(i*1.1)*3} cy={y} rx={5} ry={3.8} transform={`rotate(${8+i*1.5} ${283} ${y})`}/>)}
      {/* Stem 3 */}
      {[170,183,195,206,216,226,235].map((y,i) => <ellipse key={`c${i}`} cx={222+Math.sin(i*.9)*4} cy={y} rx={5.2} ry={3.5} transform={`rotate(${-10+i*2} ${222} ${y})`}/>)}
      {/* Stem 4 */}
      {[175,188,200,212,223,233].map((y,i) => <ellipse key={`d${i}`} cx={298+Math.sin(i*1.3)*3} cy={y} rx={4.8} ry={3.5} transform={`rotate(${12+i*1.8} ${298} ${y})`}/>)}
      {/* Stem 5 */}
      {[155,168,180,192,203,213,222].map((y,i) => <ellipse key={`e${i}`} cx={212+Math.sin(i*1.1)*3} cy={y} rx={5} ry={3.8} transform={`rotate(${-8+i*1.5} ${212} ${y})`}/>)}
      {/* Stem 6 */}
      {[165,178,190,202,213,223].map((y,i) => <ellipse key={`f${i}`} cx={308+Math.sin(i*1.2)*3} cy={y} rx={4.5} ry={3.5} transform={`rotate(${10+i*2} ${308} ${y})`}/>)}
      {/* Stem 7 */}
      {[145,158,170,182,193,203,213].map((y,i) => <ellipse key={`g${i}`} cx={207+Math.sin(i*.8)*4} cy={y} rx={4.8} ry={3.5} transform={`rotate(${-12+i*1.8} ${207} ${y})`}/>)}
    </g>
    <g fill="#c4aad4" opacity="0.35">
      {/* Tiny accent buds */}
      {[115,125,133].map((y,i) => <ellipse key={`h${i}`} cx={250+Math.sin(i)*3} cy={y} rx={3.5} ry={2.8}/>)}
      {[132,142,150].map((y,i) => <ellipse key={`i${i}`} cx={286+Math.sin(i)*2} cy={y} rx={3.2} ry={2.5}/>)}
    </g>
    {/* Leaves */}
    <g stroke="#7d9e70" strokeWidth="1" fill="#7d9e70" opacity="0.25">
      <path d="M248 360 Q230 340 215 350 Q230 355 248 360Z"/>
      <path d="M268 370 Q290 350 300 358 Q288 362 268 370Z"/>
      <path d="M242 420 Q220 400 208 412 Q222 415 242 420Z"/>
      <path d="M278 410 Q300 395 310 405 Q298 408 278 410Z"/>
    </g>
    {/* Ribbon tie */}
    <path d="M235 520 Q260 510 285 520" stroke="#c9a0b8" strokeWidth="2.5" fill="none" opacity="0.3"/>
    <path d="M285 520 Q295 530 290 540 Q285 535 280 528" stroke="#c9a0b8" strokeWidth="1.8" fill="none" opacity="0.25"/>
    <path d="M235 520 Q225 530 230 540 Q235 535 240 528" stroke="#c9a0b8" strokeWidth="1.8" fill="none" opacity="0.25"/>
  </svg>
);

// ===================== COMPONENT =====================

export default function App() {
  const [status, setStatus] = useState("idle");
  const [camErr, setCamErr] = useState(null);
  const [pct, setPct] = useState(0);
  const [countdown, setCountdown] = useState(3);
  const [uploaded, setUploaded] = useState(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const fileRef = useRef(null);

  function analyze(sourceCanvas, mode) {
    const features = extractLargestCluster(sourceCanvas);
    const score = matchScore(features);
    const p = Math.round(score * 100);
    setPct(p);
    const threshold = mode === "camera" ? 68 : 60;
    if (p >= threshold) { setStatus("matched"); stopCam(); return true; }
    return false;
  }

  function scanFrame() {
    const v = videoRef.current;
    if (!v || v.readyState < 2) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
    c.getContext("2d").drawImage(v, 0, 0);
    analyze(c, "camera");
  }

  async function startCam() {
    try {
      setCamErr(null);
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } } });
      streamRef.current = s;
      if (videoRef.current) { videoRef.current.srcObject = s; await videoRef.current.play(); }
      setStatus("scanning");
      timerRef.current = setInterval(scanFrame, 400);
    } catch { setCamErr("Camera unavailable. Upload an image instead."); setStatus("error"); }
  }

  function stopCam() {
    if (timerRef.current) clearInterval(timerRef.current); timerRef.current = null;
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
  }

  function onFile(e) {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const url = ev.target.result;
      setUploaded(url); setStatus("analyzing");
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
        c.getContext("2d").drawImage(img, 0, 0);
        const ok = analyze(c, "file");
        if (!ok) setStatus("no-match");
      };
      img.src = url;
    };
    reader.readAsDataURL(file);
  }

  useEffect(() => {
    if (status !== "matched") return;
    setCountdown(3);
    const iv = setInterval(() => setCountdown(p => { if (p <= 1) { clearInterval(iv); window.open(TARGET_URL, "_blank"); return 0; } return p - 1; }), 1000);
    return () => clearInterval(iv);
  }, [status]);

  useEffect(() => () => stopCam(), []);

  function reset() {
    stopCam(); setStatus("idle"); setPct(0); setUploaded(null); setCountdown(3);
    if (fileRef.current) fileRef.current.value = "";
  }

  const rose = "#c4727f";
  const roseLight = "#e8a0ab";
  const roseDark = "#9e4f5c";

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(168deg, #fdf2f4 0%, #fce8ec 35%, #faf0f2 65%, #fdf4f5 100%)", color: "#5a3a40", fontFamily: "'Cormorant Garamond', 'Georgia', serif", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24, position: "relative", overflow: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Outfit:wght@300;400;500&display=swap');
        @keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.04)}}
        @keyframes scanline{0%{top:0}100%{top:100%}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes check{0%{stroke-dashoffset:50}100%{stroke-dashoffset:0}}
        @keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
        @keyframes gentle{0%,100%{opacity:.055}50%{opacity:.07}}
        *{box-sizing:border-box}
        button:hover{filter:brightness(1.06);transform:translateY(-1px)}
        button{transition:all .2s ease}
      `}</style>

      <LavenderBG />

      {/* Soft noise texture */}
      <div style={{ position: "fixed", inset: 0, opacity: .03, backgroundImage: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")", pointerEvents: "none", zIndex: 0 }} />

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 32, animation: "fadeIn .6s ease-out", position: "relative", zIndex: 1 }}>
        <h1 style={{ fontSize: "clamp(34px, 7vw, 52px)", fontWeight: 600, margin: 0, letterSpacing: 1, color: "#6b3a45", lineHeight: 1.1, fontStyle: "italic" }}>
          Hi Danny!
        </h1>
        <p style={{ fontFamily: "'Outfit', sans-serif", fontSize: 14, fontWeight: 400, color: "#b0808a", marginTop: 10, maxWidth: 280, lineHeight: 1.5, letterSpacing: .3 }}>
          Scan our flowers for today's message.
        </p>
      </div>

      <div style={{ width: "100%", maxWidth: 360, position: "relative", zIndex: 1, animation: "fadeIn .7s ease-out .15s both" }}>

        {status === "idle" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <button onClick={startCam} style={{
              width: "100%", padding: "18px 24px", background: `linear-gradient(135deg, ${rose}, ${roseDark})`,
              border: "none", borderRadius: 16, color: "#fff", fontSize: 15, fontFamily: "'Outfit', sans-serif",
              fontWeight: 500, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
              boxShadow: "0 4px 20px rgba(196, 114, 127, .25), inset 0 1px 0 rgba(255,255,255,.15)", letterSpacing: .3,
            }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
              Open Camera
            </button>
            <button onClick={() => fileRef.current?.click()} style={{
              width: "100%", padding: "18px 24px", background: "rgba(255,255,255,.55)",
              border: "1px solid rgba(196,114,127,.2)", borderRadius: 16, color: "#a06a74", fontSize: 15,
              fontFamily: "'Outfit', sans-serif", fontWeight: 400, cursor: "pointer", display: "flex",
              alignItems: "center", justifyContent: "center", gap: 10, backdropFilter: "blur(8px)",
              letterSpacing: .3,
            }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              Upload Photo
            </button>
            <input ref={fileRef} type="file" accept="image/*" onChange={onFile} style={{ display: "none" }} />
          </div>
        )}

        {status === "scanning" && (
          <div>
            <div style={{ borderRadius: 18, overflow: "hidden", border: `2.5px solid ${roseLight}`, position: "relative", background: "#000", aspectRatio: "4/3", boxShadow: "0 8px 32px rgba(196,114,127,.15)" }}>
              <video ref={videoRef} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} playsInline muted />
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                <div style={{ position: "absolute", left: 0, right: 0, height: 2, background: `linear-gradient(90deg,transparent,${roseLight},transparent)`, animation: "scanline 2s ease-in-out infinite", boxShadow: `0 0 16px rgba(196,114,127,.4)` }} />
                {[{top:10,left:10},{top:10,right:10},{bottom:10,left:10},{bottom:10,right:10}].map((p,i) => (
                  <div key={i} style={{ position: "absolute", ...p, width: 22, height: 22, borderColor: roseLight, borderStyle: "solid", borderWidth: 0, ...(i<2?{borderTopWidth:2.5}:{borderBottomWidth:2.5}), ...(i%2===0?{borderLeftWidth:2.5}:{borderRightWidth:2.5}) }} />
                ))}
              </div>
            </div>
            <div style={{ marginTop: 16, padding: "14px 18px", background: "rgba(255,255,255,.5)", borderRadius: 12, border: "1px solid rgba(196,114,127,.12)", backdropFilter: "blur(6px)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontFamily: "'Outfit', sans-serif", fontSize: 11, fontWeight: 400, letterSpacing: 1.5, color: "#b0808a", textTransform: "uppercase" }}>Scanning</span>
                <span style={{ fontFamily: "'Outfit', sans-serif", fontSize: 18, fontWeight: 500, color: pct > 45 ? rose : "#c0a0a6" }}>{pct}%</span>
              </div>
              <div style={{ height: 3, background: "rgba(196,114,127,.1)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${pct}%`, background: pct > 45 ? `linear-gradient(90deg,${roseDark},${rose})` : "rgba(196,114,127,.2)", borderRadius: 2, transition: "width .3s" }} />
              </div>
            </div>
            <button onClick={reset} style={{
              width: "100%", marginTop: 12, padding: 13, background: "rgba(255,255,255,.45)",
              border: "1px solid rgba(196,114,127,.15)", borderRadius: 12, color: "#b0808a",
              fontSize: 13, fontFamily: "'Outfit', sans-serif", cursor: "pointer", backdropFilter: "blur(4px)",
            }}>Cancel</button>
          </div>
        )}

        {status === "analyzing" && (
          <div style={{ textAlign: "center", padding: 36 }}>
            <div style={{ width: 40, height: 40, border: `3px solid rgba(196,114,127,.15)`, borderTopColor: rose, borderRadius: "50%", animation: "spin .7s linear infinite", margin: "0 auto 16px" }} />
            <p style={{ fontFamily: "'Outfit', sans-serif", fontSize: 13, color: "#b0808a" }}>Recognizing flowers...</p>
          </div>
        )}

        {status === "error" && (
          <div style={{ textAlign: "center", padding: 28, background: "rgba(255,255,255,.5)", borderRadius: 18, border: "1px solid rgba(196,114,127,.12)", backdropFilter: "blur(6px)" }}>
            <p style={{ fontFamily: "'Outfit', sans-serif", fontSize: 13, color: "#a06a74", marginBottom: 16 }}>{camErr}</p>
            <button onClick={() => fileRef.current?.click()} style={{
              padding: "14px 28px", background: `linear-gradient(135deg,${rose},${roseDark})`,
              border: "none", borderRadius: 12, color: "#fff", fontSize: 14, fontFamily: "'Outfit', sans-serif",
              fontWeight: 500, cursor: "pointer", boxShadow: "0 4px 16px rgba(196,114,127,.25)",
            }}>Upload Image</button>
            <input ref={fileRef} type="file" accept="image/*" onChange={onFile} style={{ display: "none" }} />
            <br/>
            <button onClick={reset} style={{
              marginTop: 10, padding: "9px 22px", background: "transparent",
              border: "1px solid rgba(196,114,127,.2)", borderRadius: 10, color: "#b0808a",
              fontSize: 12, fontFamily: "'Outfit', sans-serif", cursor: "pointer",
            }}>Back</button>
          </div>
        )}

        {status === "no-match" && uploaded && (
          <div style={{ textAlign: "center", animation: "fadeIn .4s ease-out" }}>
            <div style={{ borderRadius: 18, overflow: "hidden", border: "2px solid rgba(196,114,127,.15)", marginBottom: 14, maxHeight: 240, boxShadow: "0 4px 20px rgba(0,0,0,.06)" }}>
              <img src={uploaded} alt="Uploaded" style={{ width: "100%", display: "block", objectFit: "cover" }} />
            </div>
            <div style={{ padding: "14px 18px", background: "rgba(255,255,255,.5)", borderRadius: 12, border: "1px solid rgba(196,114,127,.15)", marginBottom: 14, backdropFilter: "blur(4px)" }}>
              <div style={{ fontFamily: "'Outfit', sans-serif", fontSize: 14, color: "#a06a74" }}>Hmm, that doesn't look right</div>
              <div style={{ fontFamily: "'Outfit', sans-serif", fontSize: 12, color: "#c0a0a6", marginTop: 3 }}>Try scanning the flowers again</div>
            </div>
            <button onClick={reset} style={{
              width: "100%", padding: 14, background: "rgba(255,255,255,.45)",
              border: "1px solid rgba(196,114,127,.18)", borderRadius: 12, color: "#a06a74",
              fontSize: 14, fontFamily: "'Outfit', sans-serif", cursor: "pointer", backdropFilter: "blur(4px)",
            }}>Try Again</button>
          </div>
        )}

        {status === "matched" && (
          <div style={{ textAlign: "center", animation: "fadeIn .5s ease-out" }}>
            <div style={{
              width: 72, height: 72, borderRadius: "50%",
              background: `linear-gradient(135deg, ${rose}, #d4889a)`,
              display: "flex", alignItems: "center", justifyContent: "center",
              margin: "0 auto 20px",
              boxShadow: `0 0 50px rgba(196,114,127,.3), 0 0 100px rgba(196,114,127,.1)`,
              animation: "pulse 2s ease-in-out infinite",
            }}>
              <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" style={{ strokeDasharray: 50, animation: "check .5s ease-out forwards" }}/>
              </svg>
            </div>
            <h2 style={{ fontSize: 26, fontWeight: 600, margin: "0 0 6px", color: "#6b3a45", fontStyle: "italic" }}>Found it!</h2>
            <p style={{ fontFamily: "'Outfit', sans-serif", fontSize: 14, color: rose, margin: "0 0 20px" }}>
              Opening your message in {countdown}…
            </p>
            <a href={TARGET_URL} target="_blank" rel="noopener noreferrer" style={{
              display: "inline-block", padding: "14px 30px",
              background: `linear-gradient(135deg, ${rose}, ${roseDark})`,
              border: "none", borderRadius: 14, color: "#fff", fontSize: 15,
              fontWeight: 500, textDecoration: "none", fontFamily: "'Outfit', sans-serif",
              boxShadow: "0 4px 20px rgba(196,114,127,.3)", letterSpacing: .3,
            }}>
              Open Now
            </a>
            <br/>
            <button onClick={reset} style={{
              marginTop: 14, padding: "8px 20px", background: "transparent",
              border: "1px solid rgba(196,114,127,.2)", borderRadius: 8, color: "#b0808a",
              fontSize: 12, fontFamily: "'Outfit', sans-serif", cursor: "pointer",
            }}>Scan Again</button>
          </div>
        )}
      </div>
    </div>
  );
}
