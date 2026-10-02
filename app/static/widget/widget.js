/*
 * Xavier's Assistant: embeddable chat widget for sxca.edu.in.
 *
 * Add to any page with ONE tag:
 *   <script src="https://CHATBOT-SERVER/widget.js" defer></script>
 *
 * Optional attributes on the tag: data-server="https://…" (API origin, defaults to the script's origin),
 * data-fonts="off" (don't load Poppins/Open Sans from Google Fonts).
 *
 * Colours can be changed from the host page without touching this file, e.g.
 *   #xaviers-assistant { --xaviers-navy: #1f3a7a; --xaviers-crimson: #b8354e; }
 *
 * Styles live in a Shadow DOM so the college website's CSS and the widget never affect each other.
 * The conversation is kept only in this browser tab (sessionStorage); the server stores no chats.
 * Answers are rendered from text into DOM nodes (never innerHTML), so model output cannot inject HTML.
 */
(function () {
  "use strict";
  if (window.__xaviersAssistantLoaded) return;
  window.__xaviersAssistantLoaded = true;

  const script = document.currentScript;
  const BASE = ((script && script.dataset.server) || (script && script.src ? new URL(script.src, location.href).origin : location.origin)).replace(/\/$/, "");
  const STORE_KEY = "xaviers-assistant-chat-v1";
  const LANG_KEY = "xaviers-assistant-lang";
  const HISTORY_TURNS = 6;

  const LANGUAGE_NAMES = {
    en: "English", hi: "हिन्दी", gu: "ગુજરાતી", mr: "मराठी", ta: "தமிழ்", te: "తెలుగు", kn: "ಕನ್ನಡ",
    ml: "മലയാളം", bn: "বাংলা", pa: "ਪੰਜਾਬੀ", or: "ଓଡ଼ିଆ", ur: "اردو", as: "অসমীয়া", sa: "संस्कृतम्",
  };
  // Short labels for the language pill on phones.
  const LANGUAGE_SHORT = { en: "EN", hi: "HI", gu: "GU", ml: "ML", ta: "TA", te: "TE", kn: "KN", mr: "MR", bn: "BN", pa: "PA", or: "OR", ur: "UR" };
  // "Type your question…" in each language, cycled in the text box so students see they can write in any of them.
  const TYPE_HINT = {
    en: "Type your question…", hi: "अपना सवाल लिखें…", gu: "તમારો પ્રશ્ન લખો…", ml: "നിങ്ങളുടെ ചോദ്യം ടൈപ്പ് ചെയ്യൂ…",
    ta: "உங்கள் கேள்வியை எழுதுங்கள்…", te: "మీ ప్రశ్నను టైప్ చేయండి…", kn: "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ…",
    mr: "तुमचा प्रश्न लिहा…", bn: "আপনার প্রশ্ন লিখুন…", pa: "ਆਪਣਾ ਸਵਾਲ ਲਿਖੋ…", or: "ଆପଣଙ୍କ ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ…",
    ur: "اپنا سوال لکھیں…",
  };
  const SPEECH_LOCALE = { en: "en-IN", hi: "hi-IN", gu: "gu-IN", mr: "mr-IN", ta: "ta-IN", te: "te-IN", kn: "kn-IN", ml: "ml-IN", bn: "bn-IN", pa: "pa-IN", or: "or-IN", ur: "ur-IN" };

  // Used until /api/widget/config answers (or if it can't be reached).
  let cfg = {
    bot_name: "Xavier's Assistant",
    college_name: "St. Xavier's College (Autonomous), Ahmedabad",
    welcome: "Hello! I'm Xavier's Assistant. I can help you with admissions, courses, fees, faculty, exams, campus facilities and events at St. Xavier's College, Ahmedabad. What would you like to know?",
    logo_url: BASE + "/static/widget/crest.png",
    languages: ["en"],
    chips: [],
    office: { url: "https://sxca.edu.in/contact-us/", email: "info@sxca.edu.in", phone: "079-29708056/7" },
    booking_message: "",
    max_chars: 1000,
  };

  // ------------------------------------------------------------------ storage (per tab, never the server)
  function load(key, fallback) {
    try { const v = sessionStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch (e) { return fallback; }
  }
  function save(key, value) {
    try { sessionStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private mode: keep in memory only */ }
  }

  let messages = load(STORE_KEY, []);   // [{role, text, sources, answered, question, feedback, local}]
  let lang = load(LANG_KEY, "en");
  let busy = false;
  let lastFocus = null;

  // ------------------------------------------------------------------ icons (static markup)
  const I = {
    chat: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H10l-4.2 3.6c-.5.4-1.3.1-1.3-.6V16A2.5 2.5 0 0 1 4 13.5z"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    plus: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
    send: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12l16-8-6 16-2.5-6.5z"/></svg>',
    mic: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/></svg>',
    speak: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 5.5v-.5a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h.5"/></svg>',
    up: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 11v9H4.5a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1zM7 11l4-7.5c1.4 0 2.5 1.1 2.5 2.5v3.5h5a2 2 0 0 1 2 2.3l-1.2 6.5a2 2 0 0 1-2 1.7H7"/></svg>',
    down: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 13V4H4.5a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1zM7 13l4 7.5c1.4 0 2.5-1.1 2.5-2.5v-3.5h5a2 2 0 0 0 2-2.3l-1.2-6.5a2 2 0 0 0-2-1.7H7"/></svg>',
    link: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10"/></svg>',
    globe: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/></svg>',
  };

  // ------------------------------------------------------------------ styles
  const CSS = `
:host {
  /* Theme: override any of these on #xaviers-assistant from the college site's CSS. */
  --navy: var(--xaviers-navy, #243a7b);
  --crimson: var(--xaviers-crimson, #b8354e);
  --slate: var(--xaviers-slate, #37424e);
  --page: var(--xaviers-page, #f4f5f9);
  --surface: var(--xaviers-surface, #ffffff);
  --bubble-bot: var(--xaviers-bubble, #eef0f7);
  --bubble-user: var(--xaviers-navy, #243a7b);
  --fg: #1d2433;
  --muted: #5b6477;
  --line: #dde1ec;
  --chip-bg: #fbeef1;
  --chip-fg: #9c2a41;
  --link: #243a7b;
  --focus: #b8354e;
  --on-accent: #ffffff;
  --display: var(--xaviers-display-font, "Poppins", "Segoe UI", system-ui, sans-serif);
  --body: var(--xaviers-body-font, "Open Sans", "Segoe UI", system-ui, sans-serif);
  all: initial;
}
@media (prefers-color-scheme: dark) {
  :host {
    --navy: var(--xaviers-navy-dark, #1c2d63);
    --crimson: var(--xaviers-crimson-dark, #c8445d);
    --page: #0e1426; --surface: #151c33; --bubble-bot: #1f2842; --bubble-user: #2e4592;
    --fg: #e8ebf4; --muted: #a2aac0; --line: #2a3452;
    --chip-bg: #2b1c2a; --chip-fg: #f1a3b3; --link: #a9b8f0; --focus: #f1a3b3;
  }
}
*, *::before, *::after { box-sizing: border-box; }
[hidden] { display: none !important; }
button, select, textarea { font: inherit; color: inherit; }
button { cursor: pointer; }
svg { width: 20px; height: 20px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }

/* ---------- launcher (always college colours, it sits on the college site) */
.launcher {
  position: fixed; right: max(20px, env(safe-area-inset-right)); bottom: max(20px, env(safe-area-inset-bottom));
  z-index: 2147483000; display: flex; align-items: center; gap: 10px;
  padding: 8px 20px 8px 8px; border: 0; border-radius: 999px;
  background: #b8354e; color: #fff; font: 600 15px/1 var(--display);
  box-shadow: 0 12px 28px -10px rgba(120, 25, 45, .65), 0 2px 6px rgba(0,0,0,.15);
  transition: transform .18s ease, box-shadow .18s ease;
}
.launcher:hover { transform: translateY(-2px); box-shadow: 0 16px 32px -10px rgba(120, 25, 45, .7), 0 2px 6px rgba(0,0,0,.15); }
.launcher .badge { width: 40px; height: 40px; border-radius: 50%; background: #243a7b; display: grid; place-items: center; flex: none; }
.launcher .badge img { width: 26px; height: 30px; object-fit: contain; }
.launcher:focus-visible { outline-color: #243a7b; }
@media (max-width: 520px) {
  .launcher { padding: 8px; }
  .launcher .label { display: none; }
  .launcher .badge { width: 46px; height: 46px; }
}

/* ---------- full-screen chat */
.overlay {
  position: fixed; inset: 0; z-index: 2147483001;
  height: var(--xa-vh, 100vh); height: var(--xa-vh, 100dvh);
  display: flex; flex-direction: column;
  background: var(--page); color: var(--fg); font: 400 15.5px/1.55 var(--body);
  opacity: 0; transform: translateY(16px) scale(.985);
  transition: opacity .22s ease, transform .22s ease;
  -webkit-text-size-adjust: 100%;
}
.overlay.open { opacity: 1; transform: none; }
@media (prefers-reduced-motion: reduce) { .overlay, .launcher { transition: none; } .dots i { animation: none !important; } }

.bar {
  flex: none; display: flex; align-items: center; gap: 12px;
  padding: calc(10px + env(safe-area-inset-top)) max(16px, env(safe-area-inset-right)) 10px max(16px, env(safe-area-inset-left));
  background: var(--navy); color: #fff; border-bottom: 4px solid var(--crimson);
}
.bar .logo { width: 40px; height: 46px; object-fit: contain; flex: none; }
.bar .titles { min-width: 0; display: grid; }
.bar h2 { margin: 0; font: 600 17px/1.2 var(--display); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar p { margin: 2px 0 0; font-size: 12.5px; opacity: .82; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar .actions { margin-left: auto; display: flex; align-items: center; gap: 8px; flex: none; }
.bar .icon-btn { width: 40px; height: 40px; display: grid; place-items: center; border-radius: 8px; border: 1px solid rgba(255,255,255,.35); background: transparent; color: #fff; }
.bar .icon-btn:hover { background: rgba(255,255,255,.12); }
.bar .close { background: var(--crimson); border-color: transparent; }
.bar .close:hover { background: var(--crimson); filter: brightness(1.1); }
.bar .new-chat .txt { display: none; }
@media (min-width: 720px) {
  .bar .new-chat { width: auto; padding: 0 14px 0 10px; gap: 6px; display: flex; font: 600 13px var(--display); }
  .bar .new-chat .txt { display: inline; }
}
@media (max-width: 420px) { .bar p { display: none; } .bar .logo { width: 34px; height: 40px; } }

.scroll { flex: 1; overflow-y: auto; overscroll-behavior: contain; -webkit-overflow-scrolling: touch; }
.log { max-width: 820px; margin: 0 auto; padding: 24px max(16px, env(safe-area-inset-left)) 12px; display: flex; flex-direction: column; gap: 16px; }

.msg { display: flex; flex-direction: column; gap: 6px; max-width: min(88%, 680px); min-width: 0; }
.msg.user { align-self: flex-end; align-items: flex-end; }
.msg.bot { align-self: flex-start; }
.bubble { padding: 11px 15px; border-radius: 16px; overflow-wrap: anywhere; min-width: 0; }
.user .bubble { background: var(--bubble-user); color: var(--on-accent); border-bottom-right-radius: 5px; white-space: pre-wrap; }
.bot .bubble { background: var(--bubble-bot); border-bottom-left-radius: 5px; }
.bubble p { margin: 0 0 8px; } .bubble p:last-child { margin-bottom: 0; }
.bubble ul, .bubble ol { margin: 4px 0 8px; padding-left: 22px; } .bubble ul:last-child, .bubble ol:last-child { margin-bottom: 0; }
.bubble li { margin: 2px 0; }
.bubble .h { font: 600 15px/1.4 var(--display); margin: 6px 0 4px; }
.bubble a { color: var(--link); text-decoration: underline; text-underline-offset: 2px; }
.bubble code { font-family: ui-monospace, Consolas, monospace; font-size: .92em; background: var(--surface); padding: 1px 5px; border-radius: 4px; }
.status { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; }
.dots { display: inline-flex; gap: 4px; }
.dots i { width: 7px; height: 7px; border-radius: 50%; background: var(--crimson); opacity: .35; animation: xa-blink 1.2s infinite ease-in-out; }
.dots i:nth-child(2) { animation-delay: .15s; } .dots i:nth-child(3) { animation-delay: .3s; }
@keyframes xa-blink { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }

.sources { display: flex; flex-wrap: wrap; gap: 6px; }
.sources a {
  display: inline-flex; align-items: center; gap: 5px; max-width: 100%;
  font-size: 12.5px; color: var(--fg); text-decoration: none;
  background: var(--surface); border: 1px solid var(--line); border-radius: 7px; padding: 4px 9px;
}
.sources a svg { width: 13px; height: 13px; color: var(--crimson); flex: none; }
.sources a span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sources a:hover { border-color: var(--crimson); }
.tools { display: flex; gap: 4px; flex-wrap: wrap; }
.tool {
  height: 32px; min-width: 32px; padding: 0 8px; display: inline-flex; align-items: center; justify-content: center; gap: 5px;
  border: 1px solid transparent; border-radius: 7px; background: transparent; color: var(--muted); font-size: 12.5px;
}
.tool svg { width: 17px; height: 17px; }
.tool:hover { background: var(--surface); border-color: var(--line); color: var(--fg); }
.tool[aria-pressed="true"] { color: var(--crimson); background: var(--surface); border-color: var(--line); }
.tool:disabled { cursor: default; }
.handoff { display: flex; flex-wrap: wrap; gap: 8px; }
.handoff a {
  display: inline-flex; align-items: center; gap: 6px; padding: 8px 14px; border-radius: 999px; text-decoration: none;
  font: 600 13px var(--display); background: var(--crimson); color: #fff;
}
.handoff a.alt { background: transparent; color: var(--fg); border: 1px solid var(--line); }

.dock { flex: none; background: var(--surface); border-top: 1px solid var(--line); padding-bottom: env(safe-area-inset-bottom); }
.dock-inner { max-width: 820px; margin: 0 auto; padding: 10px max(16px, env(safe-area-inset-left)) 6px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; }
.chips { display: flex; gap: 8px; overflow-x: auto; scrollbar-width: none; padding: 2px 0; }
.chips::-webkit-scrollbar { display: none; }
.chip {
  flex: none; border: 1px solid transparent; border-radius: 999px; padding: 7px 14px;
  background: var(--chip-bg); color: var(--chip-fg); font: 600 13px var(--display);
}
.chip:hover { border-color: var(--chip-fg); }
.chip:disabled { opacity: .55; cursor: default; }
.composer { display: flex; align-items: flex-end; gap: 8px; min-width: 0; }

/* Language pill: a native <select> stretched invisibly over a styled label, so it stays fully accessible. */
.lang-pick {
  position: relative; flex: none; height: 46px; display: flex; align-items: center; gap: 6px;
  padding: 0 28px 0 11px; border: 1px solid var(--line); border-radius: 12px;
  background: var(--surface); color: var(--fg); font: 600 13.5px var(--display); cursor: pointer;
}
.lang-pick:hover { border-color: var(--crimson); }
.lang-pick:focus-within { outline: 3px solid var(--focus); outline-offset: 2px; }
.lang-pick .globe svg { width: 18px; height: 18px; color: var(--crimson); }
.lang-pick .code { display: none; }
.lang-pick::after {
  content: ""; position: absolute; right: 12px; top: 50%; width: 6px; height: 6px; margin-top: -5px;
  border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted); transform: rotate(45deg);
}
.lang-pick select {
  position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; font-size: 16px;
  border: 0; appearance: none;
}
.lang-pick select option { color: #1d2433; background: #fff; }
@media (max-width: 560px) {
  .lang-pick { padding: 0 22px 0 9px; gap: 4px; }
  .lang-pick .name { display: none; }
  .lang-pick .code { display: inline; }
  .lang-pick::after { right: 9px; }
}

/* Text box with a hint that cycles through the languages ("Type your question…", "अपना सवाल लिखें…", …). */
.field { position: relative; flex: 1; min-width: 0; display: flex; }
.field textarea {
  flex: 1; min-width: 0; resize: none; max-height: 140px; min-height: 46px;
  padding: 11px 14px; border: 1px solid var(--line); border-radius: 12px; background: var(--page); color: var(--fg);
  font-size: 16px; line-height: 1.45;
}
.field textarea:focus { outline: none; border-color: var(--navy); box-shadow: 0 0 0 3px rgba(36, 58, 123, .18); }
.hint {
  position: absolute; left: 15px; right: 12px; top: 11.5px; pointer-events: none;
  color: var(--muted); font-size: 16px; line-height: 1.45;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  transition: opacity .35s ease, transform .35s ease;
}
.hint.leaving { opacity: 0; transform: translateY(-8px); }
.hint.arriving { opacity: 0; transform: translateY(8px); transition: none; }
.field.filled .hint { display: none; }
.composer .mic, .composer .send { width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center; flex: none; }
.composer .mic { border: 1px solid var(--line); background: var(--surface); color: var(--muted); }
.composer .mic[aria-pressed="true"] { color: #fff; background: var(--crimson); border-color: var(--crimson); }
.composer .send { border: 0; background: var(--crimson); color: #fff; }
.composer .send:disabled { opacity: .5; cursor: default; }
.fineprint { margin: 0; text-align: center; font-size: 11.5px; color: var(--muted); line-height: 1.4; }
`;

  // ------------------------------------------------------------------ markup
  const host = document.createElement("div");
  host.id = "xaviers-assistant";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `<style>${CSS}</style>
<button class="launcher" type="button" aria-haspopup="dialog" aria-expanded="false">
  <span class="badge"><img alt="" class="launcher-logo"></span><span class="label"></span>
</button>
<div class="overlay" role="dialog" aria-modal="true" aria-labelledby="xa-title" hidden>
  <header class="bar">
    <img class="logo" alt="">
    <div class="titles"><h2 id="xa-title"></h2><p class="college"></p></div>
    <div class="actions">
      <button class="icon-btn new-chat" type="button" aria-label="Start a new chat" title="New chat">${I.plus}<span class="txt">New chat</span></button>
      <button class="icon-btn close" type="button" aria-label="Close chat" title="Close (Esc)">${I.close}</button>
    </div>
  </header>
  <div class="scroll"><div class="log" role="log" aria-live="polite"></div></div>
  <div class="dock"><div class="dock-inner">
    <div class="chips" role="group" aria-label="Quick questions"></div>
    <form class="composer" novalidate>
      <label class="lang-pick" title="Answer language" hidden>
        <span class="globe" aria-hidden="true">${I.globe}</span><span class="name" aria-hidden="true"></span><span class="code" aria-hidden="true"></span>
        <select class="lang" aria-label="Answer language"></select>
      </label>
      <div class="field">
        <label class="sr" for="xa-input">Your question, in any language</label>
        <textarea id="xa-input" rows="1" autocomplete="off" enterkeyhint="send"></textarea>
        <span class="hint" aria-hidden="true"></span>
      </div>
      <button type="button" class="mic" aria-label="Speak your question" aria-pressed="false" hidden>${I.mic}</button>
      <button type="submit" class="send" aria-label="Send">${I.send}</button>
    </form>
    <p class="fineprint">AI answers from the college website can contain mistakes; confirm important details with the college office. Chats are not saved.</p>
  </div></div>
</div>`;

  const $ = (sel) => root.querySelector(sel);
  const launcher = $(".launcher"), overlay = $(".overlay"), log = $(".log"), scroller = $(".scroll");
  const input = $("#xa-input"), form = $(".composer"), sendBtn = $(".send"), micBtn = $(".mic");
  const chipsBox = $(".chips"), langSel = $(".lang"), field = $(".field"), hint = $(".hint");

  // ------------------------------------------------------------------ safe markdown → DOM
  const INLINE = /\*\*([^*\n]+)\*\*|\[([^\]\n]+)\]\(((?:https?:\/\/|mailto:)[^\s)]+)\)|(https?:\/\/[^\s<>()"']*[^\s<>()"'.,;:!?])|`([^`\n]+)`|\*([^*\s][^*\n]*)\*|([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})/g;

  function link(href, text) {
    const a = document.createElement("a");
    a.href = href; a.textContent = text;
    if (!href.startsWith("mailto:")) { a.target = "_blank"; a.rel = "noopener noreferrer"; }
    return a;
  }

  function inline(parent, text) {
    let last = 0;
    for (const m of text.matchAll(INLINE)) {
      if (m.index > last) parent.append(text.slice(last, m.index));
      if (m[1] !== undefined) { const b = document.createElement("strong"); b.textContent = m[1]; parent.append(b); }
      else if (m[2] !== undefined) parent.append(link(m[3], m[2]));
      else if (m[4] !== undefined) parent.append(link(m[4], m[4]));
      else if (m[5] !== undefined) { const c = document.createElement("code"); c.textContent = m[5]; parent.append(c); }
      else if (m[6] !== undefined) { const i = document.createElement("em"); i.textContent = m[6]; parent.append(i); }
      else if (m[7] !== undefined) parent.append(link("mailto:" + m[7], m[7]));
      last = m.index + m[0].length;
    }
    if (last < text.length) parent.append(text.slice(last));
  }

  function renderMarkdown(el, text) {
    el.replaceChildren();
    let list = null, para = null;
    const closeAll = () => { list = null; para = null; };
    for (const raw of text.replace(/\r/g, "").split("\n")) {
      const line = raw.trim();
      if (!line) { closeAll(); continue; }
      let m;
      if ((m = line.match(/^#{1,6}\s+(.*)$/))) {
        closeAll(); const h = document.createElement("div"); h.className = "h"; inline(h, m[1].replace(/\*\*/g, "")); el.append(h);
      } else if ((m = line.match(/^(?:[-*•+]|(\d+)[.)])\s+(.*)$/))) {
        const tag = m[1] ? "OL" : "UL";
        if (!list || list.tagName !== tag) { list = document.createElement(tag.toLowerCase()); el.append(list); }
        para = null;
        const li = document.createElement("li"); inline(li, m[2]); list.append(li);
      } else {
        list = null;
        if (!para) { para = document.createElement("p"); el.append(para); }
        else para.append(document.createElement("br"));
        inline(para, line);
      }
    }
  }

  // ------------------------------------------------------------------ rendering
  function scrollDown() { scroller.scrollTop = scroller.scrollHeight; }

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html) e.innerHTML = html; // static icon markup only
    return e;
  }

  function toolButton(icon, label, text) {
    const b = el("button", "tool", icon);
    b.type = "button"; b.setAttribute("aria-label", label); b.title = label;
    if (text) b.append(text);
    return b;
  }

  function renderMessage(m) {
    const wrap = el("div", "msg " + (m.role === "user" ? "user" : "bot"));
    const bubble = el("div", "bubble");
    wrap.append(bubble);
    if (m.role === "user") { bubble.textContent = m.text; log.append(wrap); return { wrap, bubble }; }

    if (m.pending) {
      bubble.replaceChildren(statusNode(m.status || "Searching the college website…"));
    } else {
      renderMarkdown(bubble, m.text);
    }
    log.append(wrap);
    if (!m.pending) decorate(m, wrap);
    return { wrap, bubble };
  }

  function statusNode(text) {
    const s = el("div", "status", '<span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>');
    s.append(text);
    return s;
  }

  // Sources, tools and hand-off buttons under a finished answer.
  function decorate(m, wrap) {
    if (m.local) return;
    if (m.answered && m.sources && m.sources.length) {
      const box = el("div", "sources");
      box.setAttribute("aria-label", "Sources");
      for (const s of m.sources) {
        if (!/^https?:\/\//.test(s.url || "")) continue;
        const a = el("a", "", I.link);
        a.href = s.url; a.target = "_blank"; a.rel = "noopener noreferrer";
        const t = el("span"); t.textContent = (s.title || s.url) + (s.date ? " · " + s.date : "");
        a.append(t); a.title = s.title || s.url;
        box.append(a);
      }
      if (box.childElementCount) wrap.append(box);
    }
    if (m.answered === false) wrap.append(handoff());

    const tools = el("div", "tools");
    const copy = toolButton(I.copy, "Copy answer");
    copy.addEventListener("click", () => copyText(m.text, copy));
    tools.append(copy);
    if ("speechSynthesis" in window) {
      const speak = toolButton(I.speak, "Read aloud");
      speak.setAttribute("aria-pressed", "false");
      speak.addEventListener("click", () => toggleSpeak(m.text, speak, m.lang || "en"));
      tools.append(speak);
    }
    if (m.question) {
      const up = toolButton(I.up, "Helpful"), down = toolButton(I.down, "Not helpful");
      const mark = () => {
        up.setAttribute("aria-pressed", String(m.feedback === "up"));
        down.setAttribute("aria-pressed", String(m.feedback === "down"));
        up.disabled = down.disabled = !!m.feedback;
      };
      up.addEventListener("click", () => sendFeedback(m, "up", mark));
      down.addEventListener("click", () => sendFeedback(m, "down", mark));
      mark();
      tools.append(up, down);
    }
    wrap.append(tools);
  }

  function handoff() {
    const box = el("div", "handoff");
    const o = cfg.office || {};
    if (o.url) { const a = link(o.url, "Contact the college office"); a.prepend(el("span", "", I.link).firstChild); box.append(a); }
    if (o.email) { const a = link("mailto:" + o.email, o.email); a.className = "alt"; box.append(a); }
    if (o.phone) { const a = link("tel:" + o.phone.replace(/[^\d+]/g, "").slice(0, 11), o.phone); a.className = "alt"; box.append(a); }
    return box;
  }

  function renderAll() {
    log.replaceChildren();
    const welcome = { role: "assistant", text: welcomeText[lang] || cfg.welcome, local: true };
    renderMessage(welcome);
    for (const m of messages) renderMessage(m);
    scrollDown();
  }

  function renderChips() {
    chipsBox.replaceChildren();
    for (const c of cfg.chips || []) {
      const b = el("button", "chip");
      b.type = "button"; b.textContent = c.label;
      b.addEventListener("click", () => {
        if (c.action === "booking") return localReply(c.label, cfg.booking_message);
        ask(c.question || c.label);
      });
      chipsBox.append(b);
    }
  }

  function renderLanguages() {
    const langs = (cfg.languages || ["en"]).filter((l) => LANGUAGE_NAMES[l]);
    $(".lang-pick").hidden = langs.length < 2;
    langSel.replaceChildren();
    for (const l of langs) { const o = document.createElement("option"); o.value = l; o.textContent = LANGUAGE_NAMES[l]; langSel.append(o); }
    if (!langs.includes(lang)) lang = langs[0] || "en";
    langSel.value = lang;
    updateLanguagePill();
    restartHints();
  }

  function updateLanguagePill() {
    $(".lang-pick .name").textContent = LANGUAGE_NAMES[lang] || "English";
    $(".lang-pick .code").textContent = LANGUAGE_SHORT[lang] || lang.toUpperCase();
    $(".lang-pick").title = "Answer language: " + (LANGUAGE_NAMES[lang] || lang);
  }

  // ---- cycling hint: the chosen language first, then every other language, one at a time
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let hintTimer = null, hintIndex = 0, hintGeneration = 0;
  function hintLanguages() {
    const langs = (cfg.languages || ["en"]).filter((l) => TYPE_HINT[l]);
    return [lang, ...langs.filter((l) => l !== lang)].filter((l) => TYPE_HINT[l]);
  }
  function showHint(code) {
    hint.textContent = TYPE_HINT[code];
    hint.dir = code === "ur" ? "rtl" : "ltr";
    hint.lang = code;
  }
  function nextHint() {
    const list = hintLanguages();
    if (list.length < 2 || overlay.hidden || field.classList.contains("filled")) return;
    hintIndex = (hintIndex + 1) % list.length;
    const generation = hintGeneration;
    hint.classList.add("leaving");                 // slide up and fade out…
    setTimeout(() => {
      if (generation !== hintGeneration) return;   // the language changed meanwhile: restartHints() took over
      showHint(list[hintIndex]);
      hint.classList.remove("leaving");
      hint.classList.add("arriving");              // …jump below, invisible…
      void hint.offsetWidth;
      hint.classList.remove("arriving");           // …and slide up into place
    }, 350);
  }
  function restartHints() {
    clearInterval(hintTimer);
    hintTimer = null;
    hintIndex = 0;
    hintGeneration += 1;
    hint.classList.remove("leaving", "arriving");
    showHint(lang in TYPE_HINT ? lang : "en");
    if (!reducedMotion.matches && !overlay.hidden && hintLanguages().length > 1) hintTimer = setInterval(nextHint, 2600);
  }
  function stopHints() { clearInterval(hintTimer); hintTimer = null; }

  // Welcome message in the chosen language (translated by the server, cached per tab).
  const welcomeText = {};
  function loadWelcome() {
    if (lang === "en" || welcomeText[lang]) return Promise.resolve();
    const want = lang;
    return fetch(BASE + "/api/widget/welcome?lang=" + encodeURIComponent(want))
      .then((r) => (r.ok ? r.json() : null))
      .then((w) => { if (w && w.language === want) welcomeText[want] = w.text; })
      .catch(() => { /* English welcome stays */ });
  }

  function setLanguage(code) {
    if (!LANGUAGE_NAMES[code] || code === lang) return;
    lang = code;
    save(LANG_KEY, lang);
    if (![...langSel.options].some((o) => o.value === code)) {
      const o = document.createElement("option"); o.value = code; o.textContent = LANGUAGE_NAMES[code]; langSel.append(o);
    }
    langSel.value = lang;
    root.host.setAttribute("lang", code);
    updateLanguagePill();
    restartHints();
    loadWelcome().then(updateWelcome);
  }

  // The welcome is always the first message; re-render just that one (safe while an answer is streaming).
  function updateWelcome() {
    const first = log.querySelector(".msg.bot .bubble");
    if (first) renderMarkdown(first, welcomeText[lang] || cfg.welcome);
  }

  function applyConfig() {
    launcher.querySelector(".label").textContent = "Ask " + cfg.bot_name;
    launcher.setAttribute("aria-label", "Open " + cfg.bot_name + " chat");
    launcher.querySelector("img").src = cfg.logo_url;
    $(".bar .logo").src = cfg.logo_url;
    $(".bar .logo").alt = cfg.college_name + " logo";
    $("#xa-title").textContent = cfg.bot_name;
    $(".college").textContent = cfg.college_name;
    input.placeholder = "";  // the cycling .hint replaces the placeholder
    input.maxLength = cfg.max_chars || 1000;
    renderChips();
    renderLanguages();
    if (!overlay.hidden && !busy) renderAll();
    else if (overlay.hidden && !busy) log.replaceChildren(); // re-rendered with the new welcome text on open
  }

  // ------------------------------------------------------------------ actions
  function persist() {
    save(STORE_KEY, messages.filter((m) => !m.pending).slice(-40));
  }

  function setBusy(v) {
    busy = v;
    sendBtn.disabled = v;
    for (const c of chipsBox.children) c.disabled = v;
  }

  function localReply(question, answer) {
    if (busy) return;
    const q = { role: "user", text: question };
    const a = { role: "assistant", text: answer, local: true };
    messages.push(q, a);
    renderMessage(q); renderMessage(a); scrollDown(); persist();
  }

  function historyForApi() {
    return messages
      .filter((m) => !m.pending && !m.local && m.text)
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.text.slice(0, 4000) }));
  }

  function parseSSE(buffer, onEvent) {
    const parts = buffer.split(/\r?\n\r?\n/);
    const rest = parts.pop();
    for (const part of parts) {
      let event = "message", data = "";
      for (const line of part.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) { try { onEvent(event, JSON.parse(data)); } catch (e) { /* ignore a malformed event */ } }
    }
    return rest;
  }

  async function ask(question) {
    question = (question || "").trim();
    if (!question || busy) return;
    stopSpeaking();
    setBusy(true);
    const history = historyForApi();
    const q = { role: "user", text: question };
    const a = { role: "assistant", text: "", sources: [], answered: true, question, pending: true, lang: "en" };
    messages.push(q, a);
    renderMessage(q);
    const { wrap, bubble } = renderMessage(a);
    wrap.setAttribute("aria-busy", "true");
    scrollDown();

    const fail = (text) => { a.text = text; a.answered = undefined; a.local = true; };
    try {
      const res = await fetch(BASE + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ message: question, history, language: lang }),
      });
      if (!res.ok || !res.body) {
        if (res.status === 429) fail("Many people are asking questions right now. Please try again in a minute.");
        else if (res.status === 422) fail("That message is too long. Please shorten it to under " + (cfg.max_chars || 1000) + " characters.");
        else fail("Sorry, I couldn't reach the assistant just now. Please try again in a moment.");
      } else {
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "", started = false;
        const show = () => { a.pending = false; renderMarkdown(bubble, a.text); scrollDown(); };
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf = parseSSE(buf + dec.decode(value, { stream: true }), (ev, data) => {
            if (ev === "token") { a.text += data.text || ""; started = true; show(); }
            else if (ev === "replace") { a.text = data.text || ""; started = true; show(); }
            else if (ev === "status" && !started) { bubble.replaceChildren(statusNode(data.text)); }
            else if (ev === "sources") { a.sources = data.sources || []; }
            else if (ev === "language") {
              // Answer comes in this language (e.g. the student typed in Malayalam): follow it in the selector.
              a.lang = data.language;
              setLanguage(data.language);
            }
            else if (ev === "done") { a.answered = !!data.answered; }
            else if (ev === "error") { fail(data.message || "Something went wrong. Please try again."); }
          });
        }
        if (!a.text) fail("Sorry, I couldn't get an answer just now. Please try again in a moment.");
      }
    } catch (e) {
      fail("You seem to be offline, or the assistant is unreachable. Please check your connection and try again.");
    }
    a.pending = false;
    wrap.removeAttribute("aria-busy");
    renderMarkdown(bubble, a.text);
    decorate(a, wrap);
    persist();
    setBusy(false);
    scrollDown();
    if (!overlay.hidden) input.focus({ preventScroll: true });
  }

  async function sendFeedback(m, rating, mark) {
    if (m.feedback) return;
    m.feedback = rating; mark(); persist();
    try {
      await fetch(BASE + "/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating, question: m.question.slice(0, 1000), sources: (m.sources || []).map((s) => s.url).slice(0, 10) }),
      });
    } catch (e) { /* feedback is best-effort */ }
  }

  async function copyText(text, btn) {
    try { await navigator.clipboard.writeText(text); }
    catch (e) {
      const t = document.createElement("textarea"); t.value = text; t.style.position = "fixed"; t.style.opacity = "0";
      document.body.append(t); t.select();
      try { document.execCommand("copy"); } catch (e2) { /* nothing else to try */ }
      t.remove();
    }
    const old = btn.title; btn.title = "Copied"; btn.setAttribute("aria-label", "Copied");
    btn.append(" Copied");
    setTimeout(() => { btn.title = old; btn.setAttribute("aria-label", old); if (btn.lastChild.nodeType === 3) btn.lastChild.remove(); }, 1500);
  }

  // ------------------------------------------------------------------ voice (the browser's own speech engines)
  // Which Indian-language voices exist depends on the device: Chrome and Android phones have most of them,
  // Windows/Edge fewer. Without a matching voice we say so instead of mispronouncing with an English voice.
  let speakingBtn = null;
  function stopSpeaking() {
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    if (speakingBtn) speakingBtn.setAttribute("aria-pressed", "false");
    speakingBtn = null;
  }
  function voiceFor(code) {
    const locale = (SPEECH_LOCALE[code] || "en-IN").toLowerCase();
    const voices = speechSynthesis.getVoices();
    const base = locale.split("-")[0];
    return voices.find((v) => v.lang.toLowerCase().replace("_", "-") === locale)
      || voices.find((v) => v.lang.toLowerCase().startsWith(base + "-") || v.lang.toLowerCase() === base)
      || null;
  }
  if ("speechSynthesis" in window) speechSynthesis.getVoices(); // starts loading the voice list
  function toggleSpeak(text, btn, code) {
    if (speakingBtn === btn) return stopSpeaking();
    stopSpeaking();
    const voice = voiceFor(code);
    if (!voice && code !== "en") {
      const label = (LANGUAGE_NAMES[code] || code) + " voice not available on this device";
      btn.title = label;
      btn.append(" " + label);
      setTimeout(() => { btn.title = "Read aloud"; if (btn.lastChild.nodeType === 3) btn.lastChild.remove(); }, 2500);
      return;
    }
    const u = new SpeechSynthesisUtterance(text.replace(/[*#`_]/g, "").replace(/https?:\/\/\S+/g, ""));
    u.lang = SPEECH_LOCALE[code] || "en-IN";
    if (voice) u.voice = voice;
    u.onend = u.onerror = () => { if (speakingBtn === btn) stopSpeaking(); };
    speakingBtn = btn; btn.setAttribute("aria-pressed", "true");
    speechSynthesis.speak(u);
  }

  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognizer = null;
  if (Recognition) {
    micBtn.hidden = false;
    micBtn.addEventListener("click", () => {
      if (recognizer) { recognizer.stop(); return; }
      recognizer = new Recognition();
      recognizer.lang = SPEECH_LOCALE[lang] || "en-IN";
      recognizer.interimResults = true;
      micBtn.setAttribute("aria-pressed", "true");
      micBtn.setAttribute("aria-label", "Stop listening");
      recognizer.onresult = (e) => {
        input.value = Array.from(e.results).map((r) => r[0].transcript).join(" ");
        autosize();
      };
      recognizer.onend = recognizer.onerror = () => {
        recognizer = null;
        micBtn.setAttribute("aria-pressed", "false");
        micBtn.setAttribute("aria-label", "Speak your question");
        input.focus();
      };
      recognizer.start();
    });
  }

  // ------------------------------------------------------------------ open / close
  let savedOverflow = "";
  function fitViewport() {
    // On phones the on-screen keyboard shrinks the visual viewport; keep the input box above it.
    const vv = window.visualViewport;
    if (vv) overlay.style.setProperty("--xa-vh", vv.height + "px");
  }

  function open() {
    if (!overlay.hidden) return;
    lastFocus = document.activeElement;
    savedOverflow = document.documentElement.style.overflow;
    document.documentElement.style.overflow = "hidden";
    overlay.hidden = false;
    launcher.setAttribute("aria-expanded", "true");
    launcher.hidden = true;
    fitViewport();
    if (!log.childElementCount) renderAll(); // keeps an answer that is still streaming attached
    scrollDown();
    requestAnimationFrame(() => requestAnimationFrame(() => overlay.classList.add("open")));
    setTimeout(() => input.focus({ preventScroll: true }), 60);
    restartHints();
  }

  function close() {
    if (overlay.hidden) return;
    stopSpeaking();
    if (recognizer) recognizer.stop();
    overlay.classList.remove("open");
    stopHints();
    launcher.hidden = false;
    launcher.setAttribute("aria-expanded", "false");
    document.documentElement.style.overflow = savedOverflow;
    const done = () => { overlay.hidden = true; };
    matchMedia("(prefers-reduced-motion: reduce)").matches ? done() : setTimeout(done, 220);
    (lastFocus && lastFocus.focus ? lastFocus : launcher).focus({ preventScroll: true });
  }

  function newChat() {
    if (busy) return;
    stopSpeaking();
    messages = [];
    persist();
    renderAll();
    input.value = ""; autosize(); input.focus();
  }

  function focusables() {
    return Array.from(overlay.querySelectorAll("button, select, textarea, a[href]"))
      .filter((e) => !e.disabled && !e.hidden && e.getClientRects().length);
  }

  overlay.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key !== "Tab") return;
    const f = focusables();
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1], active = root.activeElement;
    if (e.shiftKey && (active === first || !overlay.contains(active))) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && (active === last || !overlay.contains(active))) { e.preventDefault(); first.focus(); }
  });

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
    field.classList.toggle("filled", input.value.length > 0);  // hide the cycling hint while typing
  }

  launcher.addEventListener("click", open);
  $(".close").addEventListener("click", close);
  $(".new-chat").addEventListener("click", newChat);
  langSel.addEventListener("change", () => setLanguage(langSel.value));
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value;
    if (!q.trim() || busy) return;
    input.value = ""; autosize();
    ask(q);
  });
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", () => { if (!overlay.hidden) { fitViewport(); scrollDown(); } });
  }

  // ------------------------------------------------------------------ boot
  function loadFonts() {
    if ((script && script.dataset.fonts === "off") || document.querySelector("link[data-xaviers-fonts]")) return;
    const l = document.createElement("link");
    l.rel = "stylesheet"; l.dataset.xaviersFonts = "";
    l.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Open+Sans:wght@400;600&display=swap";
    document.head.append(l);
  }

  function mount() {
    loadFonts();
    document.body.append(host);
    applyConfig();
    fetch(BASE + "/api/widget/config")
      .then((r) => (r.ok ? r.json() : null))
      .then((c) => {
        if (!c) return;
        if (c.logo_url && c.logo_url.startsWith("/")) c.logo_url = BASE + c.logo_url;
        cfg = Object.assign({}, cfg, c);
        applyConfig();
        return loadWelcome().then(() => { if (!overlay.hidden && !busy) renderAll(); });
      })
      .catch(() => { /* keep defaults */ });
    // Let the college site open the chat from its own buttons: <a href="#ask-xavier"> or window.XaviersAssistant.open()
    window.XaviersAssistant = { open, close, ask: (q) => { open(); ask(q); } };
    if (location.hash === "#ask-xavier") open();
    window.addEventListener("hashchange", () => { if (location.hash === "#ask-xavier") open(); });
  }

  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
})();
