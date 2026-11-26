/* Theme management + hash router + bootstrap + profile/settings. */

const ROUTES = {
  overview: renderOverview, loan: renderLoan, queue: renderQueue,
  insurance: renderInsurance, lab: renderLab, monitor: () => renderMonitor("loan"),
};
const THEME_ORDER = ["dark", "light", "sunset"];
const THEME_META = {
  dark:   { label: "Dark",   icon: "&#127769;" }, // crescent moon
  light:  { label: "Light",  icon: "&#9728;"   }, // sun
  sunset: { label: "Sunset", icon: "&#127749;" }, // sunset-over-buildings
};

/* ---------- entrance animation: replayed on every route render (see navigate()) so
   leaving a page and coming back always re-plays its charts/trend-lines/cards sliding in,
   instead of them just sitting there static the second time ---------- */
function playEnterAnimation() {
  const main = document.getElementById("main");
  main.classList.remove("page-anim");
  void main.offsetWidth; // force reflow so the same class re-triggers the CSS animation
  main.classList.add("page-anim");
}

/* ---------- profile (owner name shown in the top-right avatar) ---------- */
const PROFILE_KEY = "lia-profile";
const DEFAULT_PROFILE = { name: "Shivam Koli", role: "Project owner · Portfolio build" };
function getProfile() {
  try { return Object.assign({}, DEFAULT_PROFILE, JSON.parse(localStorage.getItem(PROFILE_KEY) || "{}")); }
  catch { return { ...DEFAULT_PROFILE }; }
}
function initialsOf(name) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  return (parts.map((w) => w[0]).join("").slice(0, 2) || "??").toUpperCase();
}
function applyProfile(p) {
  const btn = document.getElementById("ownerAvatarBtn");
  btn.textContent = initialsOf(p.name);
  btn.setAttribute("data-tip", `${p.name} · ${p.role}`);
}
function saveProfile(p) {
  localStorage.setItem(PROFILE_KEY, JSON.stringify(p));
  applyProfile(p);
}

/* ---------- settings (reduce motion, auto-refresh) ---------- */
const SETTINGS_KEY = "lia-settings";
const DEFAULT_SETTINGS = { reduceMotion: false, autoRefresh: false };
function getSettings() {
  try { return Object.assign({ ...DEFAULT_SETTINGS }, JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}")); }
  catch { return { ...DEFAULT_SETTINGS }; }
}
function applySettings(s) {
  document.documentElement.dataset.motion = s.reduceMotion ? "reduced" : "normal";
}
let _autoRefreshTimer = null;
function setupAutoRefresh(s) {
  if (_autoRefreshTimer) { clearInterval(_autoRefreshTimer); _autoRefreshTimer = null; }
  if (!s.autoRefresh) return;
  _autoRefreshTimer = setInterval(() => {
    const r = currentRoute();
    if (r === "lab" || r === "monitor") { clearApiCache(""); renderRoute(r, false); }
  }, 30000);
}
function saveSettings(s) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  applySettings(s);
  setupAutoRefresh(s);
}

/* ---------- wires the avatar + gear icon to the Profile / Settings modals ---------- */
function initProfileAndSettings() {
  const profile = getProfile();
  applyProfile(profile);
  document.getElementById("profileName").value = profile.name;
  document.getElementById("profileRole").value = profile.role;

  const settings = getSettings();
  applySettings(settings);
  document.getElementById("setReduceMotion").checked = !!settings.reduceMotion;
  document.getElementById("setAutoRefresh").checked = !!settings.autoRefresh;
  setupAutoRefresh(settings);

  const profileModal = document.getElementById("profileModal");
  const settingsModal = document.getElementById("settingsModal");
  document.getElementById("ownerAvatarBtn").onclick = () => profileModal.classList.add("open");
  document.getElementById("profileClose").onclick = () => profileModal.classList.remove("open");
  document.getElementById("profileSave").onclick = () => {
    saveProfile({
      name: document.getElementById("profileName").value.trim() || DEFAULT_PROFILE.name,
      role: document.getElementById("profileRole").value.trim() || DEFAULT_PROFILE.role,
    });
    profileModal.classList.remove("open");
  };
  document.getElementById("settingsBtn").onclick = () => settingsModal.classList.add("open");
  document.getElementById("settingsClose").onclick = () => settingsModal.classList.remove("open");
  document.getElementById("settingsSave").onclick = () => {
    saveSettings({
      reduceMotion: document.getElementById("setReduceMotion").checked,
      autoRefresh: document.getElementById("setAutoRefresh").checked,
    });
    settingsModal.classList.remove("open");
  };
  [profileModal, settingsModal].forEach((m) => {
    m.addEventListener("click", (e) => { if (e.target === m) m.classList.remove("open"); });
  });
}

function currentRoute() {
  const h = location.hash.replace("#", "");
  return ROUTES[h] ? h : "overview";
}

function applyTheme(name, { rerender = true } = {}) {
  document.documentElement.dataset.theme = name;
  localStorage.setItem("lia-theme", name);
  const url = new URL(location.href);
  url.searchParams.set("theme", name);
  history.replaceState(null, "", url);
  const btn = document.getElementById("themeBtn");
  if (btn) {
    const next = THEME_META[THEME_ORDER[(THEME_ORDER.indexOf(name) + 1) % THEME_ORDER.length]];
    btn.innerHTML = THEME_META[name].icon;
    btn.setAttribute("data-tip", `Theme: ${THEME_META[name].label} (click for ${next.label})`);
  }
  if (rerender) renderRoute(currentRoute());
}

/* calls a route's render function then (unless suppressed) replays the entrance animation -
   the single place every navigation/theme-switch/refresh goes through */
function renderRoute(route, animate = true) {
  ROUTES[route]();
  if (animate) playEnterAnimation();
}

function initTheme() {
  const url = new URL(location.href);
  const theme = url.searchParams.get("theme") || localStorage.getItem("lia-theme") || "dark";
  applyTheme(THEME_ORDER.includes(theme) ? theme : "dark", { rerender: false });
  document.getElementById("themeBtn").onclick = () => {
    const cur = document.documentElement.dataset.theme;
    applyTheme(THEME_ORDER[(THEME_ORDER.indexOf(cur) + 1) % THEME_ORDER.length]);
  };
}

async function checkApiHealth() {
  const dot = document.getElementById("apiHealthDot");
  const btn = document.getElementById("apiHealthBtn");
  const res = await api("GET", "/health");
  if (res && res.status === "ok") {
    btn.setAttribute("data-tip", `API: healthy · loan ${res.models.loan} · insurance ${res.models.insurance}`);
    dot.style.background = "var(--good)"; dot.style.boxShadow = "0 0 6px var(--good)";
  } else {
    btn.setAttribute("data-tip", "API: unreachable");
    dot.style.background = "var(--bad)"; dot.style.boxShadow = "0 0 6px var(--bad)";
  }
}

function setActiveNav(route) {
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
}

function navigate() {
  const route = currentRoute();
  setActiveNav(route);
  renderRoute(route);
}

window.addEventListener("hashchange", navigate);
window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initProfileAndSettings();
  checkApiHealth();
  if (!location.hash) location.hash = "#overview"; else navigate();
});
