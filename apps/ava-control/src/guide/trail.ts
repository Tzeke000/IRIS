// A faint thrust trail from the BACK of my body while I fly (10-08, Zeke: "your propulsion is coming from the back").
// Particles live in whatever coordinates the caller uses; draw() takes an offset to map them to the canvas.
export type Puff = { x: number; y: number; vx: number; vy: number; t0: number; life: number; r: number };

export class Trail {
  puffs: Puff[] = [];
  emit(p: { x: number; y: number }, back: { x: number; y: number }, speed: number, bodyR: number) {
    const now = performance.now();
    const n = speed > 6 ? 2 : 1;
    for (let i = 0; i < n; i++) {
      const j = (Math.random() - 0.5) * 0.6;
      const bx = back.x * Math.cos(j) - back.y * Math.sin(j), by = back.x * Math.sin(j) + back.y * Math.cos(j);
      this.puffs.push({ x: p.x + back.x * bodyR * 0.9, y: p.y + back.y * bodyR * 0.9,
        vx: bx * (0.04 + Math.random() * 0.05), vy: by * (0.04 + Math.random() * 0.05),
        t0: now, life: 380 + Math.random() * 260, r: bodyR * (0.08 + Math.random() * 0.07) });
    }
    if (this.puffs.length > 160) this.puffs.splice(0, this.puffs.length - 160);
  }
  draw(ctx: CanvasRenderingContext2D, color: string, off = { x: 0, y: 0 }, scale = 1) {
    const now = performance.now();
    this.puffs = this.puffs.filter((q) => now - q.t0 < q.life);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const q of this.puffs) {
      const a = (now - q.t0) / q.life;
      const x = (q.x + q.vx * (now - q.t0) - off.x) / scale, y = (q.y + q.vy * (now - q.t0) - off.y) / scale;
      const r = (q.r * (1 + a * 1.6)) / scale;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, color);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.globalAlpha = 0.55 * (1 - a) * (1 - a);
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }
}
