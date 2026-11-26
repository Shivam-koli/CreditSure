/* API client + formatting helpers. The frontend is served BY the same FastAPI app (see
   src/lia/api.py's StaticFiles mount), so every call below is same-origin - no base URL, no CORS
   headaches in normal use (CORS middleware is still enabled server-side for anyone who opens this
   page from a different origin during development). */

async function api(method, path, body) {
  try {
    const res = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail ? JSON.stringify(j.detail) : detail; } catch (e) {}
      return { error: `API error ${res.status}: ${detail}` };
    }
    return await res.json();
  } catch (e) {
    return { error: `Cannot reach the API (${e.message}). Is the server running?` };
  }
}

const apiCache = new Map();
async function apiCached(path, ttlMs = 60000) {
  const hit = apiCache.get(path);
  if (hit && Date.now() - hit.t < ttlMs) return hit.v;
  const v = await api("GET", path);
  apiCache.set(path, { v, t: Date.now() });
  return v;
}
/* drop cached GETs whose path starts with `prefix` (or everything, if omitted) so the next
   apiCached() call re-fetches - used by a page's "Refresh" header action */
function clearApiCache(prefix = "") {
  for (const k of apiCache.keys()) if (k.startsWith(prefix)) apiCache.delete(k);
}

function inr(x, compact = false) {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  x = Number(x);
  if (compact) {
    if (Math.abs(x) >= 1e7) return `Rs. ${(x / 1e7).toFixed(2)} Cr`;
    if (Math.abs(x) >= 1e5) return `Rs. ${(x / 1e5).toFixed(1)} L`;
  }
  const neg = x < 0; x = Math.abs(Math.round(x));
  let s = String(x);
  if (s.length > 3) {
    let head = s.slice(0, -3), tail = s.slice(-3), parts = [];
    while (head.length > 2) { parts.unshift(head.slice(-2)); head = head.slice(0, -2); }
    if (head) parts.unshift(head);
    s = parts.join(",") + "," + tail;
  }
  return (neg ? "-" : "") + "Rs. " + s;
}

function pct(x, d = 1) { return x === null || x === undefined ? "-" : `${(x * 100).toFixed(d)}%`; }

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
