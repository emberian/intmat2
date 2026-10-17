// Interior 2 volume viewer: WebGL2 ray marcher over UInt8 3D textures staged by
// Intmat2.viewer_volume. "as lit" uses the same optical model as src/render.jl:
// extinction density*(alpha_s*s + alpha_p*pig*pigAbsorb), in-scatter beta*L.
'use strict';
(async function () {
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const base = (params.get('v') || 'volume') .replace(/\/?$/, '/');
  const fail = (m) => { $('err').textContent = m; throw new Error(m); };

  const canvas = $('gl');
  const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
  if (!gl) fail('WebGL2 is not available in this browser.');

  let index;
  try { index = await (await fetch(base + 'index.json')).json(); }
  catch (e) { fail('could not load ' + base + 'index.json\n' + e); }
  $('title').textContent = params.get('name') || 'Interior 2';
  const [NX, NY, NZ] = index.dims;
  const ds = index.ds || 1;
  const phys = index.physics;
  const frames = index.frames;
  const optics = Object.assign({ solid: [0.90, 0.89, 0.86], pigment: [0.95, 0.42, 0.16],
    pigment_absorb: [0.35, 1.0, 1.4], pig_floor: 0.0, stain: 0.0 }, index.optics || {});

  // ---------------------------------------------------------------- textures
  const cache = new Map();
  async function fieldBytes(frame, name) {
    const file = frames[frame].fields[name];
    if (!file) return null;
    if (!cache.has(file)) cache.set(file, fetch(base + file).then(r => {
      if (!r.ok) throw new Error(file + ': ' + r.status);
      return r.arrayBuffer();
    }).then(b => new Uint8Array(b)));
    return cache.get(file);
  }
  function makeTex(unit) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_3D, t);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R])
      gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, NX, NY, NZ, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(NX * NY * NZ));
    return t;
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const texA = makeTex(0), texB = makeTex(1), texL = makeTex(2);
  function upload(unit, tex, bytes) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, NX, NY, NZ, gl.RED, gl.UNSIGNED_BYTE, bytes);
  }
  const lastWith = (name, f) => { for (let i = f; i >= 0; i--) if (frames[i].fields[name]) return i;
    for (let i = f; i < frames.length; i++) if (frames[i].fields[name]) return i; return -1; };

  // ---------------------------------------------------------------- shaders
  const vs = `#version 300 es
  in vec2 p; out vec2 uv; void main(){ uv = p; gl_Position = vec4(p, 0., 1.); }`;
  const fs = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 uv; out vec4 o;
  uniform sampler3D A, B, Lt;
  uniform vec3 eye, fwd, right, up, box, solidC, pigC, pigAbs;
  uniform float tanh_, aspect, density, exposure, alphaS, alphaP, pigFloor, stain, Lhi, Ahi;
  uniform int mode; uniform int clipAxis; uniform float clipPos;
  vec3 ramp(float x){ // dark-to-bright diagnostic colour ramp
    return clamp(vec3(1.6*x-0.1, 1.4*x*x+0.1*x, 0.5*sqrt(x)+0.6*x*x*x), 0., 1.); }
  float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)))*43758.5453); }
  void main(){
    vec3 d = normalize(fwd + uv.x*aspect*tanh_*right + uv.y*tanh_*up);
    vec3 lo = vec3(0.), hi = box;
    if (clipAxis >= 0) hi[clipAxis] = clipPos*box[clipAxis];
    vec3 inv = 1./d, ta = (lo-eye)*inv, tb = (hi-eye)*inv;
    vec3 tmin = min(ta,tb), tmax = max(ta,tb);
    float t0 = max(max(tmin.x,tmin.y),max(tmin.z,0.)), t1 = min(min(tmax.x,tmax.y),tmax.z);
    vec3 bg = vec3(0.012,0.012,0.014);
    if (t1 <= t0) { o = vec4(pow(bg, vec3(1./2.2)), 1.); return; }
    float boxMax = max(box.x, max(box.y, box.z));
    float ds = boxMax/400.;                       // ~400 samples across the block
    vec3 acc = vec3(0.), tr = vec3(1.);
    float t = t0 + ds*hash(gl_FragCoord.xy);
    for (int i = 0; i < 1200; i++) {
      if (t >= t1) break;
      vec3 x = eye + t*d, q = x/box;
      vec3 sig, beta; float Lv;
      if (mode == 0) {
        float s = texture(A,q).r, pg = texture(B,q).r; Lv = texture(Lt,q).r*Lhi;
        float pe = max(pg-pigFloor,0.)/(1.-pigFloor);
        float as_ = density*alphaS*s, ap = density*(alphaP+stain)*pe;
        sig = as_ + ap*pigAbs; beta = as_*solidC + ap*pigC;
      } else {
        float v = texture(A,q).r; Lv = 1.;
        sig = vec3(density*v*0.5); beta = sig*ramp(v);
      }
      vec3 att = exp(-sig*ds);
      vec3 frac = mix(vec3(ds), (1.-att)/max(sig,vec3(1e-6)), step(vec3(1e-6), sig));
      acc += tr*beta*frac*Lv; tr *= att;
      if (max(tr.x,max(tr.y,tr.z)) < 1e-3) break;
      t += ds;
    }
    vec3 v = exposure*acc; v = v/(1.+0.25*v); v += tr*bg;
    o = vec4(pow(clamp(v,0.,1.), vec3(1./2.2)), 1.);
  }`;
  function compile(type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) fail(gl.getShaderInfoLog(s));
    return s;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) fail(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p');
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const U = {}; for (const n of ['A','B','Lt','eye','fwd','right','up','box','solidC','pigC','pigAbs','tanh_','aspect',
    'density','exposure','alphaS','alphaP','pigFloor','stain','Lhi','Ahi','mode','clipAxis','clipPos'])
    U[n] = gl.getUniformLocation(prog, n);
  gl.uniform1i(U.A, 0); gl.uniform1i(U.B, 1); gl.uniform1i(U.Lt, 2);

  // box in full-resolution cells so absorption per unit length matches the simulation
  const box = [NX*ds, NY*ds, NZ*ds];
  const centre = box.map(b => b/2);
  let yaw = -0.6, pitch = 0.42, dist = 3.4*Math.max(...box);

  // ---------------------------------------------------------------- state
  const modes = { lit: 0, s: 1, pig: 1, n: 1, w: 1, L: 1 };
  let frame = frames.length - 1, loading = 0;
  async function load() {
    const mode = $('mode').value, f = frame, ticket = ++loading;
    try {
      if (mode === 'lit') {
        const [a, b, l] = await Promise.all(['s', 'pig', 'L'].map(n => fieldBytes(lastWith(n, f), n)));
        if (ticket !== loading) return;
        upload(0, texA, a); upload(1, texB, b); upload(2, texL, l);
      } else {
        const g = lastWith(mode, f);
        const a = await fieldBytes(g, mode);
        if (ticket !== loading) return;
        upload(0, texA, a);
        if (g !== f) $('err').textContent = mode + ' is only staged at step ' + frames[g].step;
        else $('err').textContent = '';
      }
    } catch (e) { $('err').textContent = String(e); }
    const s = frames[f].summary || {};
    $('tlabel').textContent = 'step ' + frames[f].step;
    $('stats').textContent = Object.entries(s).filter(([k]) => k !== 'step' && k !== 't')
      .map(([k, v]) => k.padEnd(14) + (typeof v === 'number' ? v.toPrecision(4) : v)).join('\n') +
      '\n\nvolume ' + NX + '×' + NY + '×' + NZ + (ds > 1 ? ' (÷' + ds + ')' : '') +
      '\nsource ' + index.source_dims.join('×');
    draw();
  }

  function draw() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    const eye = [centre[0] + dist*Math.cos(pitch)*Math.sin(yaw), centre[1] - dist*Math.cos(pitch)*Math.cos(yaw),
                 centre[2] + dist*Math.sin(pitch)];
    const f = norm(sub(centre, eye)), r = norm(cross(f, [0, 0, 1])), u = cross(r, f);
    gl.uniform3fv(U.eye, eye); gl.uniform3fv(U.fwd, f); gl.uniform3fv(U.right, r); gl.uniform3fv(U.up, u);
    gl.uniform3fv(U.box, box);
    gl.uniform3fv(U.solidC, optics.solid); gl.uniform3fv(U.pigC, optics.pigment); gl.uniform3fv(U.pigAbs, optics.pigment_absorb);
    gl.uniform1f(U.tanh_, Math.tan(17 * Math.PI / 180)); gl.uniform1f(U.aspect, w / h);
    gl.uniform1f(U.density, Math.pow(10, +$('density').value));
    // a specimen's lit exposure (often high for dim interiors) is not applied to diagnostics
    const diagScale = $('mode').value === 'lit' ? 1 : 1 / (optics.exposure || 1);
    gl.uniform1f(U.exposure, Math.pow(2, +$('exposure').value) * diagScale);
    gl.uniform1f(U.alphaS, phys.alpha_s); gl.uniform1f(U.alphaP, phys.alpha_p);
    gl.uniform1f(U.pigFloor, optics.pig_floor); gl.uniform1f(U.stain, optics.stain); gl.uniform1f(U.Lhi, index.hi.L || phys.L0);
    gl.uniform1i(U.mode, modes[$('mode').value]);
    gl.uniform1i(U.clipAxis, +$('clipaxis').value); gl.uniform1f(U.clipPos, +$('clip').value);
    $('clabel').textContent = (+$('clip').value).toFixed(2);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const norm = (a) => { const l = Math.hypot(...a); return a.map(x => x / l); };
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];

  // ---------------------------------------------------------------- input
  $('time').max = frames.length - 1; $('time').value = frame;
  $('time').addEventListener('input', () => { frame = +$('time').value; load(); });
  $('mode').addEventListener('change', load);
  for (const id of ['clip', 'clipaxis', 'density', 'exposure']) $(id).addEventListener('input', draw);
  let drag = null;
  canvas.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return;
    yaw += (e.clientX - drag[0]) * 0.008;
    pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - drag[1]) * 0.008));
    drag = [e.clientX, e.clientY]; draw();
  });
  canvas.addEventListener('wheel', e => { e.preventDefault(); dist *= Math.exp(e.deltaY * 0.001); draw(); }, { passive: false });
  window.addEventListener('resize', draw);
  if (params.has('mode')) $('mode').value = params.get('mode');
  if (params.has('clip')) { const [ax, pos] = params.get('clip').split(','); $('clipaxis').value = ax; $('clip').value = pos; }
  if (optics.density !== undefined) $('density').value = Math.log10(optics.density);
  if (optics.exposure !== undefined) $('exposure').value = Math.log2(optics.exposure);
  load();
})();
