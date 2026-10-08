// What's on screen, in a form I can point at (2026-10-08).
// Explicit hooks: elements tagged data-iris="<id>" (header buttons, the eye, composer, every nav tab as
// "tab:<id>"). Everything else interactive gets a derived id: "<active tab>:<slug of aria-label | text |
// placeholder>" (e.g. "server:open-my-console"), plus section headings as "<tab>:section:<slug>".
// So a script can name controls without anyone hand-tagging the whole app.

export type UiItem = { id: string; label: string; kind: string; tab: string; visible: boolean; rect: { x: number; y: number; w: number; h: number } };

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);

function labelOf(el: Element): string {
  const a = el.getAttribute("aria-label");
  if (a) return a.trim();
  const ph = (el as HTMLInputElement).placeholder;
  if (ph) return ph.trim();
  const t = (el.textContent || "").replace(/\s+/g, " ").trim();
  if (t) return t.slice(0, 60);
  return el.getAttribute("title") || "";
}

function isVisible(el: Element): boolean {
  const r = (el as HTMLElement).getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return false;
  if (r.bottom < 0 || r.right < 0 || r.top > window.innerHeight || r.left > window.innerWidth) return false;
  const cs = getComputedStyle(el as HTMLElement);
  if (cs.visibility === "hidden" || cs.display === "none" || Number(cs.opacity) === 0) return false;
  // inside a closed drawer (translated off-screen) counts as hidden: the rect check above catches it
  return true;
}

export function scanUi(activeTab: string): UiItem[] {
  const out: UiItem[] = [];
  const seen = new Map<string, number>();
  const push = (el: Element, id: string, kind: string) => {
    const n = (seen.get(id) || 0) + 1;
    seen.set(id, n);
    const fid = n > 1 ? `${id}#${n}` : id;
    const r = (el as HTMLElement).getBoundingClientRect();
    (el as HTMLElement).dataset.irisAuto = fid;
    out.push({ id: fid, label: labelOf(el), kind, tab: activeTab, visible: isVisible(el),
      rect: { x: r.left, y: r.top, w: r.width, h: r.height } });
  };
  document.querySelectorAll("[data-iris]").forEach((el) => push(el, (el as HTMLElement).dataset.iris || "", "tagged"));
  const pane = document.querySelector(".operator-drawer.open .op-main") || null;
  const scopes: [Element, string][] = [[document.querySelector(".presence-root") || document.body, "main"]];
  if (pane) scopes.push([pane, activeTab]);
  for (const [scope, prefix] of scopes) {
    scope.querySelectorAll("button, input, select, textarea, a[href], [role=button]").forEach((el) => {
      if ((el as HTMLElement).dataset.iris) return;
      if (prefix === "main" && el.closest(".operator-drawer")) return;
      const l = labelOf(el);
      if (!l) return;
      push(el, `${prefix}:${slug(l)}`, el.tagName.toLowerCase());
    });
    if (prefix !== "main") {
      scope.querySelectorAll(".section > h3, h1, h2").forEach((el) => {
        const l = labelOf(el);
        if (l) push(el, `${prefix}:section:${slug(l)}`, "heading");
      });
    }
  }
  return out;
}

export function findEl(id: string): HTMLElement | null {
  const esc = (s: string) => s.replace(/"/g, '\\"');
  return (document.querySelector(`[data-iris="${esc(id)}"]`) as HTMLElement)
    || (document.querySelector(`[data-iris-auto="${esc(id)}"]`) as HTMLElement)
    || null;
}
