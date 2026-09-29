// Admin panel behaviour (kept out of the HTML so the Content-Security-Policy can forbid inline scripts).
document.addEventListener("DOMContentLoaded", () => {
  // Mobile menu
  const toggle = document.querySelector(".menu-toggle");
  const nav = document.getElementById("nav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  // Confirm dangerous actions: <form data-confirm="Really delete?">
  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (e) => {
      if (!window.confirm(form.dataset.confirm)) e.preventDefault();
    });
  });

  // Disable submit buttons after the first click (no double uploads / double crawls)
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", () => {
      setTimeout(() => form.querySelectorAll("button[type=submit], button:not([type])").forEach((b) => (b.disabled = true)), 0);
    });
  });

  // Live crawl progress on pages that have #crawl-status
  const box = document.getElementById("crawl-status");
  if (box) pollCrawl(box);

  // Pages that wait for background processing: <meta name="refresh-while" content="processing">
  const waiting = document.querySelector("[data-refresh-seconds]");
  if (waiting) setTimeout(() => window.location.reload(), Number(waiting.dataset.refreshSeconds) * 1000);
});

function pollCrawl(box) {
  const set = (name, value) => box.querySelectorAll(`[data-field="${name}"]`).forEach((el) => (el.textContent = value));
  const bar = box.querySelector(".progress > div");
  let wasRunning = box.dataset.running === "true";

  async function tick() {
    try {
      const r = await fetch("/admin/crawl/status.json", { credentials: "same-origin" });
      if (r.redirected || !r.ok) return;
      const s = await r.json();
      box.hidden = !s.kind;
      if (s.kind) {
        set("phase", s.running ? s.phase : s.error ? "failed" : s.phase);
        set("processed", s.processed);
        set("queued", s.queued);
        set("discovered", s.discovered);
        set("changes", `${s.new} new, ${s.updated} updated, ${s.unchanged} unchanged`);
        set("errors", s.errors);
        set("index", `${s.indexed} / ${s.to_index} sources, ${s.chunks} chunks`);
        set("current", s.current || "");
        const total = s.phase === "indexing" ? s.to_index : s.processed + s.queued;
        const done = s.phase === "indexing" ? s.indexed : s.processed;
        if (bar) bar.style.width = `${total ? Math.min(100, (100 * done) / total) : s.running ? 5 : 100}%`;
        document.querySelectorAll("[data-when-running]").forEach((el) => (el.hidden = !s.running));
        document.querySelectorAll("[data-when-idle]").forEach((el) => (el.hidden = s.running));
      }
      if (wasRunning && !s.running) { window.location.reload(); return; }
      wasRunning = s.running;
    } catch (e) { /* network hiccup: try again */ }
    setTimeout(tick, 2000);
  }
  tick();
}
