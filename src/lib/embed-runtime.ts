export interface EmbedConfig {
  imageDataUrl: string;
  algorithm: "floyd-steinberg" | "bayer" | "blue-noise";
  scale: number;
  dotScale: number;
  invert: boolean;
  image: {
    threshold: number;
    contrast: number;
    gamma: number;
    blur: number;
    highlightsCompression: number;
  };
  dither: { errorStrength: number; serpentine: boolean };
  shape: { cornerRadius: number };
  gridSize: number;
}

function runDitherEmbed(CONFIG: EmbedConfig): void {
  const GRID_SIZE = CONFIG.gridSize;
  const BAYER_8X8 = [
    0, 32, 8, 40, 2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44, 4, 36, 14, 46, 6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
    3, 35, 11, 43, 1, 33, 9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47, 7, 39, 13, 45, 5, 37,
    63, 31, 55, 23, 61, 29, 53, 21,
  ];

  function floydSteinberg(grayscale: Uint8Array, width: number, height: number, opts: { threshold: number; serpentine: boolean; errorStrength: number }, alpha?: Uint8Array): Float32Array {
    const errors = new Float32Array(width * height);
    for (let i = 0; i < grayscale.length; i++) errors[i] = grayscale[i];
    const positions: number[] = [];
    const strength = opts.errorStrength;
    const hasAlpha = !!(alpha && alpha.length === grayscale.length);
    for (let y = 0; y < height; y++) {
      const ltr = !opts.serpentine || y % 2 === 0;
      const sx = ltr ? 0 : width - 1;
      const ex = ltr ? width : -1;
      const step = ltr ? 1 : -1;
      for (let x = sx; x !== ex; x += step) {
        const idx = y * width + x;
        if (hasAlpha && alpha![idx] < 128) continue;
        const oldVal = errors[idx];
        const newVal = oldVal > opts.threshold ? 255 : 0;
        const err = (oldVal - newVal) * strength;
        if (newVal > 0) positions.push(x, y);
        const diffuse = (nx: number, ny: number, w: number) => {
          if (nx < 0 || nx >= width || ny >= height) return;
          const ni = ny * width + nx;
          if (hasAlpha && alpha![ni] < 128) return;
          errors[ni] += err * w;
        };
        diffuse(x + step, y, 7 / 16);
        diffuse(x - step, y + 1, 3 / 16);
        diffuse(x, y + 1, 5 / 16);
        diffuse(x + step, y + 1, 1 / 16);
      }
    }
    return new Float32Array(positions);
  }

  function bayerDither(grayscale: Uint8Array, width: number, height: number, opts: { threshold: number }, alpha?: Uint8Array): Float32Array {
    const positions: number[] = [];
    const bias = (opts.threshold - 128) / 255;
    const hasAlpha = !!(alpha && alpha.length === grayscale.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (hasAlpha && alpha![idx] < 128) continue;
        const luma = grayscale[idx] / 255;
        const bv = (BAYER_8X8[(y & 7) * 8 + (x & 7)] + 1) / 65;
        if (luma + bias > bv) positions.push(x, y);
      }
    }
    return new Float32Array(positions);
  }

  function blueNoiseDither(grayscale: Uint8Array, width: number, height: number, noiseData: Uint8Array, noiseSize: number, opts: { threshold: number }, alpha?: Uint8Array): Float32Array {
    const positions: number[] = [];
    const bias = (opts.threshold - 128) / 255;
    const hasAlpha = !!(alpha && alpha.length === grayscale.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (hasAlpha && alpha![idx] < 128) continue;
        const luma = grayscale[idx] / 255;
        const nx = x % noiseSize;
        const ny = y % noiseSize;
        const nv = noiseData[ny * noiseSize + nx] / 255;
        if (luma + bias > nv) positions.push(x, y);
      }
    }
    return new Float32Array(positions);
  }

  function generateBlueNoise(size: number): Uint8Array {
    const data = new Uint8Array(size * size);
    for (let i = 0; i < data.length; i++) data[i] = Math.floor(Math.random() * 256);
    for (let p = 0; p < 3; p++) {
      const blurred = new Float32Array(data.length);
      const r = 2;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          let s = 0, c = 0;
          for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
            const nx = (x + dx + size) % size;
            const ny = (y + dy + size) % size;
            s += data[ny * size + nx]; c++;
          }
          blurred[y * size + x] = s / c;
        }
      }
      for (let i = 0; i < data.length; i++) {
        const d = data[i] - blurred[i];
        data[i] = Math.max(0, Math.min(255, Math.round(data[i] + d * 0.5)));
      }
    }
    return data;
  }

  function roundedSquareMask(w: number, h: number, radiusPct: number): Set<number> {
    const r = Math.round(radiusPct * Math.min(w, h));
    const mask = new Set<number>();
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let inside = false;
        if (x >= r && x < w - r) inside = y >= 0 && y < h;
        else if (y >= r && y < h - r) inside = x >= 0 && x < w;
        else {
          const cx = x < r ? r : w - r - 1;
          const cy = y < r ? r : h - r - 1;
          const dx = x - cx, dy = y - cy;
          inside = dx * dx + dy * dy <= r * r;
        }
        if (inside) mask.add(y * w + x);
      }
    }
    return mask;
  }

  function invertWithMask(positions: Float32Array, gw: number, gh: number, radiusPct: number, alpha?: Uint8Array): Float32Array {
    const mask = roundedSquareMask(gw, gh, radiusPct);
    const logoSet = new Set<number>();
    for (let i = 0; i < positions.length; i += 2) logoSet.add(Math.round(positions[i + 1]) * gw + Math.round(positions[i]));
    const inv: number[] = [];
    for (const idx of mask) {
      if (!logoSet.has(idx)) {
        if (alpha && alpha[idx] < 128) continue;
        inv.push(idx % gw, Math.floor(idx / gw));
      }
    }
    return new Float32Array(inv);
  }

  function loadImage(src: string): Promise<HTMLImageElement> {
    return new Promise((res, rej) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = src;
    });
  }

  function processImage(img: HTMLImageElement, maxDim: number, scale: number, contrast: number, gamma: number, blur: number, hc: number) {
    const aspect = img.naturalWidth / img.naturalHeight;
    let outW: number, outH: number;
    if (aspect >= 1) { outW = maxDim; outH = Math.round(maxDim / aspect); }
    else { outH = maxDim; outW = Math.round(maxDim * aspect); }
    const srcW = img.naturalWidth, srcH = img.naturalHeight;
    const ac = document.createElement("canvas");
    ac.width = outW; ac.height = outH;
    const actx = ac.getContext("2d")!;
    actx.imageSmoothingEnabled = true;
    actx.imageSmoothingQuality = "high";
    actx.drawImage(img, 0, 0, outW, outH);
    const alphaData = actx.getImageData(0, 0, outW, outH).data;
    const pad = Math.ceil(blur * 3);
    const sc = document.createElement("canvas");
    sc.width = srcW + pad * 2; sc.height = srcH + pad * 2;
    const sctx = sc.getContext("2d")!;
    if (blur > 0) sctx.filter = "blur(" + blur + "px)";
    sctx.drawImage(img, pad, pad, srcW, srcH);
    sctx.filter = "none";
    const c = document.createElement("canvas");
    c.width = outW; c.height = outH;
    const ctx = c.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(sc, pad, pad, srcW, srcH, 0, 0, outW, outH);
    const px = ctx.getImageData(0, 0, outW, outH).data;
    const sw = Math.ceil(outW / scale), sh = Math.ceil(outH / scale);
    const gray = new Uint8Array(sw * sh);
    const alpha = new Uint8Array(sw * sh);
    const cf = (259 * (contrast + 255)) / (255 * (259 - contrast));
    for (let sy = 0; sy < sh; sy++) {
      for (let sx = 0; sx < sw; sx++) {
        const xx = Math.min(Math.round(sx * scale), outW - 1);
        const yy = Math.min(Math.round(sy * scale), outH - 1);
        const i = (yy * outW + xx) * 4;
        const r = px[i], g = px[i + 1], b = px[i + 2];
        const ba = px[i + 3] / 255;
        alpha[sy * sw + sx] = alphaData[i + 3];
        let luma = ba > 0.01 ? (0.299 * r + 0.587 * g + 0.114 * b) / ba : 0;
        if (contrast !== 0) luma = cf * (luma - 128) + 128;
        if (gamma !== 1) luma = 255 * Math.pow(Math.max(0, luma / 255), 1 / gamma);
        if (hc > 0) {
          const n = luma / 255;
          luma = (n < 0.5 ? n : 0.5 + (n - 0.5) * (1 - hc)) * 255;
        }
        gray[sy * sw + sx] = Math.max(0, Math.min(255, Math.round(luma)));
      }
    }
    return { grayscale: gray, alpha, width: sw, height: sh };
  }

  interface DotSys {
    count: number; baseX: Float32Array; baseY: Float32Array;
    dx: Float32Array; dy: Float32Array; brightness: Float32Array; tint: Float32Array; size: number;
  }
  function createDotSystem(points: Float32Array, sf: number, ds: number, ox: number, oy: number): DotSys {
    const count = points.length / 2;
    const baseX = new Float32Array(count), baseY = new Float32Array(count);
    const dx = new Float32Array(count), dy = new Float32Array(count);
    const brightness = new Float32Array(count), tint = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      baseX[i] = ox + points[i * 2] * sf;
      baseY[i] = oy + points[i * 2 + 1] * sf;
      brightness[i] = 1; tint[i] = 1;
    }
    return { count, baseX, baseY, dx, dy, brightness, tint, size: sf * ds };
  }

  const SHOCKWAVE_SPEED = 225, SHOCKWAVE_WIDTH = 37, SHOCKWAVE_STRENGTH = 20, SHOCKWAVE_DURATION = 675;
  const MOUSE_RADIUS = 100, MOUSE_RADIUS_SQ = MOUSE_RADIUS * MOUSE_RADIUS;
  const MOUSE_FORCE_PEAK = 40, EASING = 0.12, SNAP = 0.01;

  interface Shockwave { x: number; y: number; start: number }
  function updateDots(sys: DotSys, mx: number, my: number, mActive: boolean, sw: Shockwave[], now: number): boolean {
    const { count, baseX, baseY, dx, dy } = sys;
    let active = sw.length;
    for (let k = sw.length - 1; k >= 0; k--) {
      if (now - sw[k].start >= SHOCKWAVE_DURATION) { sw.splice(k, 1); active--; }
    }
    const sm = active > 0 ? 1 + 0.5 * (active - 1) : 0;
    let motion = false;
    for (let i = 0; i < count; i++) {
      let fx = 0, fy = 0;
      if (mActive) {
        const vx = (baseX[i] + dx[i]) - mx;
        const vy = (baseY[i] + dy[i]) - my;
        const d2 = vx * vx + vy * vy;
        if (d2 > 0.1 && d2 < MOUSE_RADIUS_SQ) {
          const d = Math.sqrt(d2);
          const fo = 1 - d / MOUSE_RADIUS;
          const f = fo * fo * fo * MOUSE_FORCE_PEAK;
          fx += (vx / d) * f; fy += (vy / d) * f;
        }
      }
      for (let k = 0; k < sw.length; k++) {
        const w = sw[k];
        const el = now - w.start;
        const rad = (el / 1000) * SHOCKWAVE_SPEED;
        const life = 1 - el / SHOCKWAVE_DURATION;
        const sxv = baseX[i] - w.x, syv = baseY[i] - w.y;
        const d = Math.sqrt(sxv * sxv + syv * syv);
        if (d >= 0.1) {
          const band = Math.abs(d - rad);
          if (band < SHOCKWAVE_WIDTH) {
            const wf = (1 - band / SHOCKWAVE_WIDTH) * life * SHOCKWAVE_STRENGTH * sm;
            fx += (sxv / d) * wf; fy += (syv / d) * wf;
          }
        }
      }
      dx[i] += (fx - dx[i]) * EASING;
      dy[i] += (fy - dy[i]) * EASING;
      if (Math.abs(dx[i]) < SNAP) dx[i] = 0;
      if (Math.abs(dy[i]) < SNAP) dy[i] = 0;
      if (dx[i] !== 0 || dy[i] !== 0) motion = true;
    }
    return motion || sw.length > 0 || mActive;
  }

  function renderDots(ctx: CanvasRenderingContext2D, sys: DotSys, invert: boolean, dpr: number, w: number, h: number) {
    ctx.clearRect(0, 0, w * dpr, h * dpr);
    const r = invert ? 0 : 138, g = invert ? 0 : 143, b = invert ? 0 : 152;
    const buckets: number[][] = new Array(126);
    for (let z = 0; z < 126; z++) buckets[z] = [];
    for (let i = 0; i < sys.count; i++) {
      const bk = 6 * Math.round(20 * sys.brightness[i]) + Math.round(5 * sys.tint[i]);
      buckets[Math.max(0, Math.min(125, bk))].push(i);
    }
    const size = sys.size * dpr, pad = 0.25 * dpr, padSize = 0.5 * dpr;
    for (let z = 0; z < 126; z++) {
      const ids = buckets[z];
      if (!ids.length) continue;
      ctx.fillStyle = "rgba(" + r + "," + g + "," + b + "," + (Math.floor(z / 6) / 20) + ")";
      for (let j = 0; j < ids.length; j++) {
        const i = ids[j];
        const rx = (sys.baseX[i] + sys.dx[i]) * dpr;
        const ry = (sys.baseY[i] + sys.dy[i]) * dpr;
        ctx.fillRect(rx - pad, ry - pad, size + padSize, size + padSize);
      }
    }
  }

  const canvas = document.getElementById("dither-embed-canvas") as HTMLCanvasElement;
  if (!canvas) return;
  const ctx = canvas.getContext("2d")!;
  const dpr = window.devicePixelRatio || 1;
  const mouse = { x: 0, y: 0, active: false };
  const shockwaves: Shockwave[] = [];
  let sys: DotSys | null = null;
  let blueNoise: Uint8Array | null = null;
  let frame = 0;
  let running = false;

  function tick() {
    if (!sys) { running = false; return; }
    const rect = canvas.getBoundingClientRect();
    const more = updateDots(sys, mouse.x, mouse.y, mouse.active, shockwaves, performance.now());
    renderDots(ctx, sys, CONFIG.invert, dpr, rect.width, rect.height);
    if (more) frame = requestAnimationFrame(tick);
    else running = false;
  }
  function startLoop() {
    if (running) return;
    running = true;
    frame = requestAnimationFrame(tick);
  }

  async function rebuild() {
    const rect = canvas.getBoundingClientRect();
    const img = await loadImage(CONFIG.imageDataUrl);
    const p = processImage(img, GRID_SIZE, 1, CONFIG.image.contrast, CONFIG.image.gamma, CONFIG.image.blur, CONFIG.image.highlightsCompression);
    const opts = { threshold: CONFIG.image.threshold, serpentine: CONFIG.dither.serpentine, errorStrength: CONFIG.dither.errorStrength };
    let pos: Float32Array;
    if (CONFIG.algorithm === "floyd-steinberg") pos = floydSteinberg(p.grayscale, p.width, p.height, opts, p.alpha);
    else if (CONFIG.algorithm === "bayer") pos = bayerDither(p.grayscale, p.width, p.height, opts, p.alpha);
    else { if (!blueNoise) blueNoise = generateBlueNoise(256); pos = blueNoiseDither(p.grayscale, p.width, p.height, blueNoise, 256, opts, p.alpha); }
    if (CONFIG.invert) pos = invertWithMask(pos, p.width, p.height, CONFIG.shape.cornerRadius, p.alpha);
    const s = Math.max(0.5, Math.min(rect.width, rect.height) * CONFIG.scale / Math.max(p.width, p.height));
    const ox = Math.round((rect.width - p.width * s) / 2);
    const oy = Math.round((rect.height - p.height * s) / 2);
    sys = createDotSystem(pos, s, CONFIG.dotScale, ox, oy);
    startLoop();
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    if (sys) renderDots(ctx, sys, CONFIG.invert, dpr, rect.width, rect.height);
  }

  let resizeTimer: ReturnType<typeof setTimeout> | null = null;
  let lastW = 0, lastH = 0;
  const ro = new ResizeObserver(() => {
    resize();
    const rect = canvas.getBoundingClientRect();
    const w = Math.round(rect.width), h = Math.round(rect.height);
    if (lastW !== 0 && (w !== lastW || h !== lastH)) {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => rebuild(), 200);
    }
    lastW = w; lastH = h;
  });
  ro.observe(canvas);
  resize();

  canvas.addEventListener("pointermove", (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.active = true;
    startLoop();
  });
  canvas.addEventListener("pointerleave", (e) => {
    if (e.pointerType !== "mouse") return;
    mouse.active = false; startLoop();
  });
  canvas.addEventListener("pointercancel", () => { mouse.active = false; startLoop(); });
  canvas.addEventListener("pointerup", (e) => {
    const rect = canvas.getBoundingClientRect();
    shockwaves.push({ x: e.clientX - rect.left, y: e.clientY - rect.top, start: performance.now() });
    if (e.pointerType !== "mouse") mouse.active = false;
    startLoop();
  });

  rebuild();
}

const RUNTIME_SOURCE = runDitherEmbed.toString();

export function buildEmbedHTML(config: EmbedConfig): string {
  const bg = config.invert ? "#ffffff" : "#0a0a0a";
  const configJSON = JSON.stringify(config);
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dither Embed</title>
<style>
  html,body{margin:0;height:100%;background:${bg};overflow:hidden;}
  #dither-embed-canvas{display:block;width:100vw;height:100vh;touch-action:none;cursor:crosshair;}
</style>
</head>
<body>
<canvas id="dither-embed-canvas"></canvas>
<script>
(${RUNTIME_SOURCE})(${configJSON});
</script>
</body>
</html>
`;
}

export async function imageSrcToDataUrl(src: string): Promise<string> {
  const res = await fetch(src);
  const blob = await res.blob();
  return await new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}
