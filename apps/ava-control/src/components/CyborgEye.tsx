// CyborgEye — my 3D body (2026-10-08). The Blender cyborg eye (ceramic plates over a glowing core, titanium
// bezel engraved "IRIS * IR-0509", screws, my emblem, a glass cornea) exported to GLB, with the LIVE iris
// mapped onto the disc behind the cornea: a hidden IrisBody renders exactly as before (emotion colour,
// pupil dilation, blinks, sleep lid, speech rings…) and its canvas is the texture. So every behaviour the
// flat iris had comes along for free, and the gaze now turns the WHOLE ball.
//
// The ports (where the cable-arms plug in) are projected to page coordinates every frame and published in
// `eyePorts` so the guide layer can root the arms on the real collars.
import { memo, useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import IrisBody, { type IrisBodyProps } from "./IrisBody";
import { deriveBlendColors, getCfg } from "./orbShared";
import { EyeArms3D, controllerFor, eyeArms } from "./eyeArms3d";

export type PortKey = "right" | "down" | "left" | "up";
export type PortInfo = { x: number; y: number; nx: number; ny: number };
export type EyePorts = { ts: number; center: { x: number; y: number }; radius: number; ports: Record<PortKey, PortInfo> };
/** Live port positions per eye instance (page px), keyed by `portsKey`. */
export const eyePorts = new Map<string, EyePorts>();

export type CyborgEyeProps = IrisBodyProps & { portsKey?: string };

const MODEL_URL = "/models/iris_eye_body.glb";
let modelPromise: Promise<THREE.Group> | null = null;
function loadModel(): Promise<THREE.Group> {
  if (!modelPromise) {
    modelPromise = new GLTFLoader().loadAsync(MODEL_URL).then((g) => g.scene as THREE.Group);
    modelPromise.catch(() => { modelPromise = null; });
  }
  return modelPromise;
}

const LIMBUS_Z = 0.835;          // Blender front (-Y) → three +Z; iris disc sits a little behind the limbus
const IRIS_DISC_R = 0.57;
const IRIS_FRAC = 0.546;         // IrisBody: iris radius / half canvas
const BALL_FIT = 1.18;           // ball radius incl. bezel lip, for framing
// port positions on the ball (Blender ±X, ±Z  →  three ±X, ±Y), at the collar tips
// The canvas is bigger than the eye so the real 3D cables have room around the body (the layout box stays `size`).
const CANVAS_K = 2.5;
const PORT_POS: Record<PortKey, THREE.Vector3> = {
  right: new THREE.Vector3(1.08, 0, 0), left: new THREE.Vector3(-1.08, 0, 0),
  up: new THREE.Vector3(0, 1.08, 0), down: new THREE.Vector3(0, -1.08, 0),
};

function CyborgEyeInner(props: CyborgEyeProps) {
  const { size = 320, portsKey } = props;
  const mountRef = useRef<HTMLDivElement>(null);
  const irisHostRef = useRef<HTMLDivElement>(null);
  const live = useRef(props);
  live.current = props;
  const IRIS_TEX = size <= 200 ? 256 : 512;

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;
    let DBG = ""; try { DBG = localStorage.getItem("iris.cy.debug") || ""; } catch { /* none */ }
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    // Quality: software GL (no GPU, e.g. llvmpipe on a VM desktop) starts LOW; a frame-rate monitor steps
    // down further at runtime and, as a last resort, hands the body back to the flat iris (Zeke 10-08: "1 frame a
    // second… needs to be at least 30, 60 is better").
    let rendererName = "";
    try {
      const gl = renderer.getContext();
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      rendererName = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    } catch { /* unknown */ }
    const software = /llvmpipe|swiftshader|softpipe|software|basic render/i.test(rendererName);
    let quality: "high" | "low" = software || DBG.includes("low") ? "low" : "high";
    const dpr = quality === "high" ? Math.min(window.devicePixelRatio, 1.25) : 1;   // the canvas is 2.5× the eye: keep the pixel count sane
    const CS = Math.round(size * CANVAS_K);
    renderer.setSize(CS, CS);
    renderer.setPixelRatio(dpr);
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTex;
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-2.5, 3, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xbfd4ff, 1.2);
    rim.position.set(2.5, 1.5, -3);
    scene.add(rim);

    const fov0 = 28;                                    // framing as if the canvas were `size`…
    const fov = (2 * Math.atan(CANVAS_K * Math.tan((fov0 / 2) * Math.PI / 180))) * 180 / Math.PI;   // …widened for the cables
    const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 50);
    camera.position.z = BALL_FIT / Math.sin((fov0 / 2) * Math.PI / 180) * 1.22;   // leave room around the ball

    const ball = new THREE.Group();
    scene.add(ball);
    const arms3d = new EyeArms3D();
    scene.add(arms3d.group);
    if (portsKey) eyeArms.set(portsKey, controllerFor(arms3d));
    const ray = new THREE.Raycaster(), plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), ndc = new THREE.Vector2();

    // materials (rebuilt in three: the GLB carries names + the emblem texture; the Cycles node graphs don't port)
    const mood = { light: new THREE.Color("#6aa3ff"), base: new THREE.Color("#1a6cf5") };
    const M = {
      ceramic: new THREE.MeshPhysicalMaterial({ color: 0xcfcfcb, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.08, metalness: 0, envMapIntensity: 0.8 }),
      gunmetal: new THREE.MeshPhysicalMaterial({ color: 0x2b2d33, roughness: 0.32, metalness: 1, anisotropy: 0.6, envMapIntensity: 0.55 }),
      core: new THREE.MeshStandardMaterial({ color: 0x08090c, roughness: 0.5, metalness: 0.6, emissive: mood.base, emissiveIntensity: 1.4 }),
      seam: new THREE.MeshStandardMaterial({ color: 0x050505, emissive: mood.light, emissiveIntensity: 3 }),
      strip: new THREE.MeshStandardMaterial({ color: 0x050505, emissive: mood.light, emissiveIntensity: 3 }),
      screw: new THREE.MeshStandardMaterial({ color: 0x9da0a6, roughness: 0.3, metalness: 1 }),
      socket: new THREE.MeshStandardMaterial({ color: 0x050506, roughness: 0.7, metalness: 0.3 }),
      ink: new THREE.MeshStandardMaterial({ color: 0x0b0c10, roughness: 0.45 }),
      etched: new THREE.MeshStandardMaterial({ color: 0x0b0b0e, roughness: 0.75, metalness: 0.2 }),
      cornea: new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.02, transmission: 1, thickness: 0.05, ior: 1.4,
        iridescence: 0.35, iridescenceIOR: 1.38, iridescenceThicknessRange: [300, 420], transparent: true }),
    };
    const lowCeramic = new THREE.MeshStandardMaterial({ color: 0xcfcfcb, roughness: 0.35, metalness: 0 });
    const lowCornea = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.05, metalness: 0, transparent: true, opacity: 0.12 });
    const applyQuality = () => {
      if (quality !== "low") return;
      renderer.setPixelRatio(1);
      byName.Ceramic = lowCeramic; byName.Cornea = lowCornea;
      ball.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        if (mesh.material === M.ceramic) mesh.material = lowCeramic;
        if (mesh.material === M.cornea) mesh.material = lowCornea;
      });
    };
    const byName: Record<string, THREE.Material> = {
      Ceramic: M.ceramic, Gunmetal: M.gunmetal, CoreChannels: M.core, SeamGlow: M.seam, StripGlow: M.strip,
      ScrewSteel: M.screw, ScrewSocket: M.socket, Ink: M.ink, Etched: M.etched, Cornea: M.cornea,
    };

    // the live iris: a hidden IrisBody's canvas as a texture on the disc behind the cornea
    let irisTex: THREE.CanvasTexture | null = null;
    const irisMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
    const discGeo = new THREE.CircleGeometry(IRIS_DISC_R, 128);
    const uv = discGeo.attributes.uv as THREE.BufferAttribute;
    const k = IRIS_FRAC * (IRIS_DISC_R / 0.55);
    for (let i = 0; i < uv.count; i++) uv.setXY(i, 0.5 + (uv.getX(i) - 0.5) * k, 0.5 + (uv.getY(i) - 0.5) * k);
    const disc = new THREE.Mesh(discGeo, irisMat);
    disc.position.z = LIMBUS_Z - 0.06;
    ball.add(disc);
    const backing = new THREE.Mesh(new THREE.CircleGeometry(IRIS_DISC_R, 64), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    backing.position.z = LIMBUS_Z - 0.07;
    ball.add(backing);

    let disposed = false;
    loadModel().then((src) => {
      if (disposed) return;
      const body = src.clone(true);
      body.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        const name = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material)?.name || "";
        if (byName[name]) mesh.material = byName[name];
        else if (name === "EmblemDecal") {
          const m = (mesh.material as THREE.MeshStandardMaterial).clone();
          m.transparent = true; m.alphaTest = 0.05; m.roughness = 0.3;
          mesh.material = m;
        }
      });
      if (DBG.includes("notrans")) M.cornea.transmission = 0;
      if (DBG.includes("nobody")) body.visible = false;
      ball.add(body);
      applyQuality();
      fpsWatchFrom = performance.now() + 1500;   // let shaders compile before judging
    }).catch(() => { /* model missing: the disc alone still shows the iris */ });

    // gaze: the whole ball turns (critically damped); a little idle life
    const rot = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    let softUntil = 0;                                   // after a body turn, ease back (no snap)
    let prevState = "", avertX = 0, avertY = 0, avertUntil = 0;    // the BODY looks away to think now
    let fpsWatchFrom = Infinity, frames = 0, gaveUp = false, badLow = 0, badHigh = 0;
    const v3 = new THREE.Vector3();
    let raf = 0, last = performance.now(), paused = false;
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      if (paused) return;
      const dt = Math.max(0, Math.min(0.05, (now - last) / 1000)); last = now;   // rAF time can precede performance.now(): never integrate a NEGATIVE step
      const L = live.current;
      const cfg = getCfg(String(L.emotion || "calmness"));
      const c = deriveBlendColors(String(L.emotionColor || ""), cfg);
      mood.light.set(c.lightColor); mood.base.set(c.color);
      const amp = Math.min(1, Number(L.amplitude || 0));
      const glow = (L.state === "offline" ? 0.15 : L.state === "sleeping" ? 0.35 : 1) * (1 + amp * 0.8);
      M.core.emissive.copy(mood.base); M.core.emissiveIntensity = 1.3 * glow;
      M.seam.emissive.copy(mood.light); M.seam.emissiveIntensity = 2.6 * glow;
      M.strip.emissive.copy(mood.light); M.strip.emissiveIntensity = 2.6 * glow;
      // gaze target (x right, y up) → yaw/pitch; idle micro-wander when nobody's steering
      const t = now / 1000;
      const st = String(L.state || "idle");
      if (st !== prevState) {
        if (st === "thinking" || st === "deep") {
          avertX = (Math.random() < 0.5 ? -1 : 1) * (st === "deep" ? 0.55 : 0.4); avertY = 0.3;
          avertUntil = t + (st === "deep" ? 4 : 2) + Math.random();
        } else avertUntil = 0;
        prevState = st;
      }
      const av = t < avertUntil;
      const gx = (L.gaze ? L.gaze.x : 0.05 * Math.sin(t * 0.37)) + (av ? avertX : 0);
      const gy = (L.gaze ? L.gaze.y : 0.04 * Math.sin(t * 0.29 + 1.3)) + (av ? avertY : 0);
      // A BODY heading (10-08, Zeke: "your propulsion is coming from the back of you — show your profile when you
      // move left and right, so it doesn't look like you're just gliding"): yaw/pitch/roll turn the whole body on a
      // softer, heavier spring that slightly overshoots (follow-through); plain gaze stays a quick saccade.
      const H = L.gaze as { yaw?: number; pitch?: number; roll?: number } | undefined;
      const turning = H && H.yaw !== undefined;
      if (turning) softUntil = t + 0.9;
      const ty = turning ? H!.yaw! : gx * 0.34, tx = turning ? (H!.pitch ?? 0) : -gy * 0.28, tz = turning ? (H!.roll ?? 0) : 0;
      const soft = turning || t < softUntil;
      const K = soft ? 34 : 140, C = 2 * Math.sqrt(K) * (soft ? 0.62 : 0.9);
      rot.vy += (K * (ty - rot.y) - C * rot.vy) * dt; rot.y += rot.vy * dt;
      rot.vx += (K * (tx - rot.x) - C * rot.vx) * dt; rot.x += rot.vx * dt;
      rot.vz += (K * (tz - rot.z) - C * rot.vz) * dt; rot.z += rot.vz * dt;
      ball.rotation.set(rot.x, rot.y, rot.z);
      if (portsKey) (window as unknown as Record<string, unknown>)[`__cy_${portsKey}`] = { gx, gy, ty, tx, rx: rot.x, ry: rot.y, dt, ball, camera };
      const s = Math.max(0.35, Math.min(1.35, Number(L.bodyScale ?? 1)));
      ball.scale.setScalar(s);
      // the iris texture: grab the hidden IrisBody canvas
      const cv = irisHostRef.current?.querySelector("canvas") as HTMLCanvasElement | null;
      if (cv) {
        if (!irisTex || irisTex.image !== cv) {
          irisTex?.dispose();
          irisTex = new THREE.CanvasTexture(cv);
          irisTex.colorSpace = THREE.SRGBColorSpace;
          irisMat.map = irisTex; irisMat.needsUpdate = true;
        }
        if (!DBG.includes("notex")) irisTex.needsUpdate = true;
      }
      // page px → world (the z=0 plane through the ball centre), for the guide's arm targets
      const cr = container.getBoundingClientRect();
      arms3d.toWorld = (x: number, y: number) => {
        ndc.set(((x - cr.left) / cr.width) * 2 - 1, -(((y - cr.top) / cr.height) * 2 - 1));
        ray.setFromCamera(ndc, camera);
        const hit = new THREE.Vector3();
        return ray.ray.intersectPlane(plane, hit) ?? hit;
      };
      ball.updateMatrixWorld();
      arms3d.step(now, ball, camera.position, mood.light, glow);
      renderer.render(scene, camera);
      // frame-rate guard: judge 2.5 s windows after the model is up
      if (now > fpsWatchFrom && !gaveUp) {
        frames++;
        if (now - fpsWatchFrom > 2500) {
          const fps = frames / ((now - fpsWatchFrom) / 1000);
          frames = 0; fpsWatchFrom = now;
          // two bad windows in a row before stepping down (a heavy tab or a screen recorder can dip one window)
          badHigh = fps < 30 && quality === "high" ? badHigh + 1 : 0;
          badLow = fps < 18 && quality === "low" ? badLow + 1 : 0;
          if (badHigh >= 2) { quality = "low"; applyQuality(); }
          else if (badLow >= 3) {
            gaveUp = true;                         // this machine can't carry the 3D body: flat iris instead
            try { sessionStorage.setItem("iris.cy.tooSlow", String(Math.round(fps))); } catch { /* none */ }
            window.dispatchEvent(new Event("iris-body-style"));
          }
        }
      }
      // publish the ports (page px) for the cable-arms
      if (portsKey && container) {
        const r = container.getBoundingClientRect();
        const proj = (p: THREE.Vector3) => {
          v3.copy(p).applyMatrix4(ball.matrixWorld).project(camera);
          return { x: r.left + (v3.x * 0.5 + 0.5) * r.width, y: r.top + (-v3.y * 0.5 + 0.5) * r.height };
        };
        const ctr = proj(new THREE.Vector3(0, 0, 0));
        const ports = {} as Record<PortKey, PortInfo>;
        for (const k2 of Object.keys(PORT_POS) as PortKey[]) {
          const p = proj(PORT_POS[k2]);
          const q = proj(PORT_POS[k2].clone().multiplyScalar(1.25));
          const dx = q.x - p.x, dy = q.y - p.y, dl = Math.hypot(dx, dy) || 1;
          ports[k2] = { x: p.x, y: p.y, nx: dx / dl, ny: dy / dl };
        }
        const edge = proj(new THREE.Vector3(1, 0, 0));
        eyePorts.set(portsKey, { ts: performance.now(), center: ctr, radius: Math.hypot(edge.x - ctr.x, edge.y - ctr.y) * s, ports });
      }
    };
    raf = requestAnimationFrame(loop);
    const onVis = () => { paused = document.hidden; };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVis);
      if (portsKey) { eyePorts.delete(portsKey); eyeArms.delete(portsKey); }
      arms3d.dispose();
      irisTex?.dispose();
      envTex.dispose(); pmrem.dispose();
      Object.values(M).forEach((m) => m.dispose());
      discGeo.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [size, portsKey]);

  // the hidden live iris (no gaze/scale/pointer of its own: the ball does that now)
  const { gaze: _g, bodyScale: _b, portsKey: _k, size: _s, shapeOverride, ...irisProps } = props;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <div ref={irisHostRef} aria-hidden="true"
        style={{ position: "absolute", left: 0, top: 0, width: IRIS_TEX, height: IRIS_TEX, opacity: 0, pointerEvents: "none", overflow: "hidden" }}>
        <IrisBody {...irisProps} embedded shapeOverride={shapeOverride === "pointer" ? undefined : shapeOverride} size={IRIS_TEX} />
      </div>
      <div ref={mountRef} style={{ position: "absolute", left: -((CANVAS_K - 1) / 2) * size, top: -((CANVAS_K - 1) / 2) * size,
        width: size * CANVAS_K, height: size * CANVAS_K, pointerEvents: "none" }} />
    </div>
  );
}

const CyborgEye = memo(CyborgEyeInner);
export default CyborgEye;
