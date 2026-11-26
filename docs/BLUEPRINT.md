# Project Blueprint — Loan Eligibility Prediction and Insurance Pricing Forecast

A living document. Every time the project changes in a meaningful way, a new entry gets added to
the **Progress Log** at the bottom. Everything above the log is kept in sync with the current state
of the repo — read it top to bottom for the story, or jump to the log to see how we got here.

---

## 1. Problem Statement

Two Indian retail-finance decisions, solved as one portfolio web app:

1. **Loan eligibility** — given an applicant's demographic, income, employment, credit-history and
   asset data, decide whether to approve the loan, and explain the decision in terms a credit
   officer (and an interviewer) can audit.
2. **Insurance premium pricing** — given an individual's demographic and health profile, predict
   their annual premium with an honest error range, not just a point number.

Both are built on **synthetic data seeded from public Kaggle datasets** (never real customer or
employer data — see `docs/DATA_CARD.md`), and both are designed to be run **free, locally, in
Docker**, with a documented mapping to a real GCP deployment, presented through a futuristic,
Dark/Light/Sunset-themed dashboard — static HTML/CSS/JS, a top-nav SaaS-style shell with tooltips
and hover micro-interactions, served by the same FastAPI app, no build step — designed for a
non-technical user making a prediction.

The project exists to be a single, coherent, end-to-end artifact for a CV and GitHub profile — not
four disconnected course notebooks. It merges four separate ProjectPro course titles into one
system:

| # | ProjectPro title | Where it lives in this repo |
|---|---|---|
| 1 | Loan Eligibility Prediction using Gradient Boosting Classifier | `src/lia/loan_train.py` |
| 2 | Loan Eligibility Prediction Project using Machine Learning on GCP | MySQL feature store, SQL views, `docs/GCP_MAPPING.md` |
| 3 | Insurance Pricing Forecast Using XGBoost Regressor | `src/lia/insurance_train.py` |
| 4 | Loan Eligibility Prediction in Python using H2O.ai | built, verified, then deliberately dropped — see Roadmap & Q&A below |

## 2. Objectives

- **Correctness over polish**: every number in the README must trace back to a file that was
  actually produced by a real run — no invented metrics, no cherry-picked comparisons.
- **Industry-standard DS vocabulary, demonstrated, not name-dropped**: experiment tracking
  (MLflow), cost-based decision thresholds, calibration, drift monitoring, SHAP reason codes, a
  model registry with champion/challenger promotion.
- **Zero cost, fully reproducible**: everything runs with `docker compose up` on a free tier of
  hardware; the optional heavier piece (MLflow UI) sits behind a Compose profile so it never slows
  down the default path. Every host port (MySQL, API/dashboard, MLflow UI) is overridable via `.env`
  so the stack never collides with other local projects on the same machine.
- **Explainability as a first-class output**, not an afterthought: every loan decision and every
  premium prediction ships with a reason code, not just a number.
- **Presentation as a first-class output, not an afterthought**: the person using this app is
  making a real decision (approve this loan? price this policy?), so the UI is designed to be read
  at a glance — a futuristic, card-based dashboard with a live Dark/Light/Sunset theme switch — while
  the backend underneath it is untouched: the same FastAPI service, the same trained pipelines, the
  same SHAP reason codes, regardless of which theme is on screen.
- **Interview-ready honesty**: limitations (synthetic data, what the seed datasets really
  contributed, licence status, a feature that was built and then deliberately dropped) are
  documented, not hidden.

## 3. Work Flow

```
Raw landing CSVs (deliberately dirty)
        │  profiling, de-dup, date/currency parsing, sentinel handling
        ▼
MySQL: raw → cleansed → analytics-layer SQL views
        │  feature engineering (FOIR, LTV, BMI recompute, calendar features)
        ▼
sklearn Pipelines (ColumnTransformer + encoder + scaler)
        │
        ├─ Loan: SMOTE-in-CV-fold → 7-candidate comparison → BayesSearchCV tuning
        │        → Platt calibration → cost-based break-even threshold → champion/challenger
        │
        └─ Insurance: linear baseline → boosted-model comparison → BayesSearchCV tuning
                 → point prediction + Mondrian conformal interval
        │
        ▼
MLflow experiment tracking (every run logged; champions registered)
        │
        ▼
SHAP reason codes attached to every prediction
        │
        ▼
FastAPI serving layer — also serves the themed dashboard (Dark/Light/Sunset, static HTML/CSS/JS) at
the same URL — all in Docker
        │
        ▼
Scheduled retrain job + PSI/KS drift check against the Aug 2026 drift episode
```

## 4. Architecture

- **Feature store**: MySQL, with SQL views doing the raw → cleansed → analytics-layer
  transformation (window functions for de-dup, date/currency normalisation, category mapping).
- **Modelling**: scikit-learn Pipelines for every model family, so preprocessing never leaks
  between train/val/test; SMOTE applied inside the CV fold only. XGBoost is the single modelling
  engine behind both the loan comparison and the insurance regressor (see Roadmap for why a second,
  independent engine — H2O.ai — was built, validated, and then removed rather than kept).
- **Experiment tracking**: MLflow, SQLite-backed (`tracking.py`), logging every candidate model
  and registering both production champions. Deliberately *not* a long-running `mlflow server`
  container — see the design-decision note in the README.
- **Explainability**: SHAP TreeExplainer reason codes on every served prediction.
- **Serving**: FastAPI in Docker, which also serves the dashboard directly (a `StaticFiles` mount,
  so there is no separate frontend container, build step or port). The dashboard itself is plain
  HTML/CSS/JS with three live-switchable themes (Dark / Light / Sunset) — a CSS custom-property
  palette per theme, swapped instantly by setting one attribute on `<html>`, with charts (Chart.js)
  and gauges (hand-drawn SVG) re-reading those same colours whenever a page re-renders.
- **Operations**: a scheduled retrain job plus a PSI/KS drift monitor, tested against a built-in
  drift episode in the synthetic data (1 Aug 2026 onward).
- **Deployment story**: everything above maps to a GCP VM / Vertex AI equivalent — see
  `docs/GCP_MAPPING.md` — without ever needing to actually provision one.

## 5. Roadmap

| Status | Item |
|---|---|
| ✅ Done | Loan module: 7-candidate comparison, SMOTE, cost-based threshold, champion/challenger |
| ✅ Done | Insurance module: XGBoost regressor beating a linear baseline, conformal interval |
| ✅ Done | SHAP reason codes for both modules |
| ✅ Done | Scheduled retrain + PSI/KS drift monitoring |
| ✅ Done | Dockerised FastAPI, with configurable host ports |
| ✅ Done | README/registry consistency fix (champion label corrected to match real results) |
| ➖ Dropped | H2O.ai independent comparison (ProjectPro title #4) — built, verified, then deliberately removed in favour of a leaner, XGBoost-focused repo |
| ✅ Done | MLflow experiment tracking + model registry |
| ➖ Dropped | Streamlit as the UI framework — a Light/Dark/Mix redesign was built in it first, then replaced outright (see Roadmap entry below and Progress Log) |
| ✅ Done | Futuristic Dark/Light/Sunset dashboard: static HTML/CSS/JS, no build step, served by FastAPI itself |
| ✅ Done | SaaS-dashboard-style redesign: top nav shell, invoice-style queue list, hover/tooltip micro-interactions |
| ✅ Done | Full-bleed layout fix, app-wide hover-lift, replayed entrance animation, editable Profile + Settings (reduce motion, auto-refresh) |
| ✅ Done | This blueprint document — kept current as the project progresses (this file) |
| 🔲 Pending | GitHub repository creation and push |
| 🔲 Pending | Kaggle seed-dataset licence verification (2 of 4 seeds confirmed, see Data Card §8.6) |
| ➖ Not needed | Academic publications — confirmed not required; this is a personal portfolio project only |

## 6. Real-World Connecting Questions & Answers

**Q: Why does the loan champion change between retrains (GLM one day, XGBoost another)?**
A: Because the comparison is apples-to-apples on a cost-based business metric (break-even
approval value in Rs crore), not just ROC-AUC. On this synthetic data the signal-to-noise ratio
puts several models within a few percent of each other on that metric, so small changes in the
random seed or SMOTE's resampling can flip the ranking. In a real bank this is exactly why you
never hard-code "XGBoost is the champion" in a README — you re-evaluate every retrain and let the
registry, not a document, decide.

**Q: The project description says it merges four ProjectPro titles, but H2O.ai is gone — why?**
A: It was built first, as a genuine independent cross-check: the identical loan comparison —
same split, same Platt calibration, same cost-based money rule — re-run on H2O's own estimators
instead of sklearn's. It worked, and it agreed with the sklearn/XGBoost champion (H2O's own best
model was also a GLM, within 2% on validation value), which is a real, useful validation result in
its own right. It was then deliberately removed: H2O.ai shows up far less often than XGBoost in
Data Scientist job descriptions, and carrying a second ML engine, its own Docker image and a JVM
dependency just to re-prove a result that already agreed wasn't worth the extra surface area for a
portfolio repo whose job is to be read and run quickly by a recruiter or interviewer. The title
itself stays listed (the project only uses the four existing ProjectPro titles, never invents new
ones) but is marked "dropped" rather than silently deleted from the story — knowing when to cut a
working feature, and being able to say why, is itself worth more in an interview than keeping it.

**Q: Why SQLite for MLflow instead of a real tracking server?**
A: `mlflow server` spins up multiple worker processes and background job schedulers meant for a
team sharing one tracking service continuously. For a single-developer local Docker setup with a
prior history of running out of disk space on heavy containers, that overhead buys nothing. A
direct SQLite connection gives the same experiment history and model registry, viewable on demand
via `mlflow ui`, without an always-on container.

**Q: Why is the data synthetic instead of using the raw Kaggle CSVs directly?**
A: Two reasons. First, so the repo can legally ship example data without depending on each seed
dataset's licence terms for *redistribution* (only for the modelling techniques they illustrate).
Second, so defects (missing values, mixed formats, outliers, drift) can be injected on purpose and
tracked in a manifest — which makes the data-cleaning section of the project demonstrable and
gradable, not just "my CSV happened to be messy."

**Q: Why three themes (Dark/Light/Sunset) instead of one fixed look?**
A: Presentation quality — a prediction tool that a real user would act on deserves to look
considered, not like a default dev dashboard. Each theme is a set of CSS custom properties
(`--bg`, `--accent`, `--card`, …) declared once per `data-theme` value; switching themes is one
attribute change on `<html>`, and every component — cards, gauges, tables, charts — recolours
instantly because they all read `var(--…)` rather than a hard-coded colour. The ML backend
(FastAPI, the trained pipelines, the SHAP reason codes) is completely unaware this exists — it is
a presentation-layer concern only.

**Q: The README first said this was a Streamlit app — why is it now plain HTML/CSS/JS?**
A: The project's own spec always allowed either ("a Streamlit or React front end"). A full
Light/Dark/Mix redesign was actually built and shipped in Streamlit first — a real, working
mutable-colour-palette theme system, since Streamlit only imports each module once per process and
a naive "import a colour constant" design can't support switching themes live. It worked, but it
still looked like a Streamlit app: a framework-generated sidebar, canvas-rendered tables that
couldn't fully follow the theme, limited control over exact spacing and card layout. Looking at the
dashboard against the reference designs it was meant to match, that gap was the actual feedback —
not a request for "more CSS on top of Streamlit," but for the dashboard to stop looking like a data
app and start looking like a product. Two real frontend paths both solve that: a React single-page
app, or a dependency-free static HTML/CSS/JS dashboard calling the same API from the browser. The
static option won on fit for this project specifically: zero Node/npm toolchain to install (the
project's own "zero cost, runs anywhere" constraint, and pip alone had already caused PATH friction
on the development machine), one fewer moving part in `docker-compose.yml` (no separate frontend
container — FastAPI serves the dashboard itself via a `StaticFiles` mount), and native `<table>`
elements that follow the theme completely, which Streamlit's canvas-rendered tables never could.
The trade-off, stated plainly: it is not a React codebase, so it doesn't demonstrate component-
framework experience the way a Vite/React build would — a fair point to raise if asked, and a real
reason someone might choose differently for a different portfolio.

## 7. Interview Talking Points

- "I ran a **fair, cost-based comparison** across seven loan models, not just ROC-AUC — the
  champion is picked by a break-even approval-value rule, so the choice maps to an actual business
  outcome."
- "I built an **independent cross-check on a second ML engine** (H2O.ai) using the identical split,
  calibration and business rule, confirmed it agreed with the sklearn/XGBoost champion — and then
  **deliberately removed it**, because XGBoost is what actually shows up in Data Scientist job
  descriptions and a leaner, focused repo reads better than a wider one. Shipping discipline is a
  skill, not just shipping."
- "I added **MLflow experiment tracking** and made a deliberate trade-off not to run a 24/7
  tracking server, because the resource cost didn't match a single-developer local setup — and I
  can explain exactly why."
- "Every prediction — loan or insurance — ships with a **SHAP reason code**, so the decision isn't
  a black box."
- "I built in a **drift episode** on purpose and monitor for it with PSI/KS, so the retrain job has
  something real to catch."
- "The data is **synthetic by design**, and I can describe exactly what signal came from the real
  seed datasets versus what I generated — I don't overclaim what the AUC numbers mean."
- "I treated the **UI as a real deliverable**, not an afterthought — a themeable, card-based
  dashboard — while keeping it strictly a presentation layer: the prediction and forecasting logic
  underneath never changed, which is itself a basic separation-of-concerns point worth making."
- "I **built the dashboard in Streamlit first, shipped it, then replaced it** with a dependency-free
  static HTML/CSS/JS frontend once it was clear the framework itself was the ceiling on how it could
  look. Recognising that a working solution still isn't the right one, and rebuilding instead of
  layering more CSS on top, is the same shipping judgement as the H2O decision above — just applied
  to the presentation layer instead of the modelling layer."

## 8. Honest Limitations (carried from the Data Card)

See `docs/DATA_CARD.md` section 8 for the full list. In short: the outcome labels come from a
documented synthetic risk model, not real repayment history; the smoker premium loading follows a
US-based seed, not Indian tariffs; the architecture images in the course knowledge file are
generic ProjectPro carousel diagrams, not project-specific; and the Kaggle seed licences are not
fully verified (2 of 4 confirmed — tracked in the roadmap above). One earlier limitation is now
resolved rather than carried forward: Streamlit's canvas-rendered tables used to keep one fixed
look regardless of the active theme — the dashboard's native HTML tables no longer have that
restriction.

---

## Progress Log

*(Newest entry first. Each entry is a snapshot of what changed and why — not a diff.)*

### 2026-11-25 — Settings feature removed; cache-busting added; Profile kept
- After the previous entry's rebuild, the Profile and Settings modals rendered as plain inline
  content on the page instead of as hidden overlays, and the full-bleed layout fix appeared not to
  have taken effect either. Root cause: `index.html`, `app.css` and `app.js` were all linked with
  the same static filenames as before (`css/app.css`, `js/app.js`, ...), so the browser kept serving
  its previously cached copies of those files even after the Docker image was rebuilt with new
  content - the new HTML loaded, but the new CSS/JS mostly didn't, which explains both symptoms at
  once (`.modal-backdrop { display: none }` never applied, neither did the `overflow-x: hidden` fix).
- Fixed for good: every CSS/JS `<link>`/`<script>` tag in `index.html` now carries a `?v=3` query
  string. Bumping that number on any future CSS/JS edit forces browsers to fetch the new file instead
  of reusing a stale cached copy - no more relying on the user remembering to hard-refresh.
- Per explicit feedback, the Settings feature (gear icon, Reduce motion, Auto-refresh) was removed
  entirely rather than debugged in place - it added complexity without being asked for twice. Pulled
  the gear button, its modal markup, the `motionOK()` plumbing in `ui.js`'s four Chart.js builders,
  and all of `app.js`'s settings/auto-refresh code. The Profile modal (avatar click -> editable Name
  + Role, saved to `localStorage`) stays, now correctly hidden until clicked.

### 2026-11-25 — Full-bleed layout, global hover-lift, live profile/settings, replayed entrance animation
- Fixed a layout bug: `<html>`/`<body>` had no explicit width/overflow guard, so on some viewport
  widths the page could gain a horizontal scrollbar with the browser's default white canvas showing
  at the edge. `.shell` is now a column flexbox that fills the viewport exactly, `html`/`body` are
  pinned to 100% width with `overflow-x: hidden`, and `html` carries the themed background too (not
  just `body`) — there is no longer any path to a stray white strip or side-scroll, in any theme.
- Broadened the hover "pop" affordance from a handful of hand-picked cards to every self-contained
  block in the app — KPI cards, queue rows, flow steps, reason lines, stat rows, tiles, callouts —
  so wherever the cursor lands, that section visibly lifts toward it, consistently, app-wide.
- Added a page-entrance animation (fade + slide-up) that replays on *every* route render — switching
  tabs, coming back to a page, or changing theme all retrigger it. Combined with Chart.js already
  destroying and rebuilding each chart per render, this means charts/sparklines/trend-lines visibly
  draw themselves in again each time a page is revisited, not just on first load.
- The top-right avatar is now a real control, not a static label: clicking it opens a Profile modal
  (Name, Role/tagline) that saves to `localStorage` and immediately updates the avatar's initials and
  tooltip. This also fixes a latent bug — the avatar was still hardcoded to a previous placeholder
  owner name from an earlier pass of this redesign; it now defaults to the project's actual owner
  and is user-editable from the UI itself rather than requiring a code edit.
- Added a Settings modal (gear icon, top-right) with two toggles persisted to `localStorage` and
  applied immediately on Save: **Reduce motion** (kills every hover-lift/entrance/chart animation at
  once, via a `data-motion="reduced"` attribute on `<html>` that both CSS and the Chart.js builders
  check) and **Auto-refresh Model Lab & Monitoring every 30s** (reuses the existing `clearApiCache` +
  re-render path already wired to the manual Refresh buttons).
- Note for next local run: `web/` is `COPY`'d into the `api` image at build time (no bind mount), so
  these changes need `docker compose up --build` (or at least a rebuild of the `api` service) to show
  up in a running stack — a plain restart won't pick them up.

### 2026-11-21 — SaaS-dashboard redesign: top nav, invoice-style queue, Dark/Light/Sunset
- The static HTML/CSS/JS dashboard below was functionally complete but still read as a data-science
  demo rather than a product. The user pointed to two reference fintech SaaS dashboards (a
  dark+green invoice app and a light+purple one) and asked for the same structural language — a top
  navigation bar instead of a sidebar, KPI cards with embedded mini-charts, and a two-pane
  list/detail view for browsing records — applied to this project's own sections and data, not
  copied wholesale (no borrowed brand names, logos or fabricated numbers).
- Replaced the left sidebar with a sticky top nav: logo (a CSS-gradient mark that recolours itself
  automatically, since its fill is `--accent`/`--accent2`) + name on the left, pill-style page tabs
  in the middle, and a cluster of icon buttons on the right (a live `/health` indicator, a link to
  the FastAPI `/docs` reference, the theme switch, and an initials avatar) — the same shape as both
  reference dashboards' header bar.
- Renamed the three themes **Dark / Light / Sunset** and, per explicit instruction, changed what
  "brand accent" means in each: the reference dashboard's neon green is specifically a dark-theme
  look this project's Dark theme does **not** borrow — Dark now uses a cool indigo/cyan glow instead
  (`--accent:#6C8CFF`, `--accent2:#38BDF8`), reserving green for its existing universal meaning
  ("eligible / stable"). Light now leans into white + a true light green brand accent
  (`--accent:#16A34A`). Sunset (renamed from the earlier "Mix") keeps its warm dusk palette but
  shifted the gradient toward true sunset tones (orange → pink, `--accent:#FB923C` /
  `--accent2:#F472B6`) rather than the previous purple/magenta. One theme-switch button now cycles
  Dark → Light → Sunset → Dark, with its icon and tooltip updating to name the *next* theme — matching
  the "single button, click to change theme" ask more directly than the old three-button segmented
  control.
- Added the micro-interactions the user asked for by name ("cursor moves in, it pops up, then
  normal again"): a small reusable CSS tooltip component (`[data-tip]`) on every icon-only button,
  and a hover-lift on clickable cards, KPI tiles and queue rows (`transform: translateY(-4px)` plus
  a soft accent-tinted glow, both transitioning back out on mouseleave) — real CSS transitions, not
  a static mockup. One real bug this caught during validation: tooltips default to popping *above*
  their element, which is invisible for anything in a nav bar pinned to the very top of the
  viewport; added a `.topnav [data-tip]` override that flips them to appear below instead.
- Rebuilt the Applications Queue page as a genuine two-pane list/detail view, the closest 1:1 match
  to the reference "invoice list" pattern: the left pane is now a scrollable list of rows (an
  initials avatar coloured by decision — EL/MR/NE in green/amber/red — id, purpose, CIBIL, and a
  right-aligned amount), the selected row highighted exactly like the reference's selected invoice
  row; the right pane leads with a header, badge and a 3-tile "loan amount / EMI / rate" summary row
  (standing in for the reference's "sub total / total / balance due" tiles), with the full
  gauge+SHAP-reasons assessment collapsed behind a real "View full assessment" toggle rather than
  always shown — a working expand/collapse, not a decorative button.
- Gave the Overview page's five KPI cards the "card with an embedded chart" look from the reference:
  four of them got a compact inline scale showing where that metric's real value sits on a
  reasonable range (e.g. ROC-AUC between 0.5 and 1.0) — reusing the same gauge/scale component used
  elsewhere, not a new fabricated number — and the "new applications" card got an actual Chart.js
  sparkline of real monthly application counts (pulled from the existing drift-trend endpoint, which
  already tracks `n` per month for drift monitoring). No number on this page is invented; every
  chart renders only data the API already returns.
- Added two small, genuinely functional "Refresh" header actions (Model Lab, Monitoring) that clear
  the client-side cache for just that page's endpoints and re-fetch — useful in practice after
  running a retrain or drift job while the dashboard tab is already open.
- Validated the same way as both earlier UI rebuilds: ran the real FastAPI service against the
  already-trained models, drove a real headless Chrome through all 6 pages × all 3 themes (18
  combinations, 0 console/page errors after the tooltip-clipping fix above), and re-ran the backend
  test suite as a regression check (22/22, unchanged — this was presentation-layer only).

### 2026-11-21 — Streamlit replaced with a static HTML/CSS/JS dashboard
- After seeing the Streamlit redesign below running, the user's feedback was blunt: it still looked
  "old" next to the two reference fintech-dashboard images, and asked for it to be rebuilt in
  "frontend technologies which are giving futuristic UI's." That is a framework swap, not another
  CSS pass, so before building anything the user was asked to choose between three real options —
  a React + Vite SPA, a dependency-free static HTML/CSS/JS dashboard, or staying in Streamlit with
  more custom CSS — since guessing wrong on a rebuild this size would be expensive to redo. They
  chose the static HTML/CSS/JS option.
- Rebuilt the dashboard from scratch as `web/`: `index.html` (the page shell — sidebar, nav, theme
  switch), `css/app.css` (the three themes as CSS custom properties — see the Q&A below for why this
  is actually a cleaner mechanism than Streamlit's mutable-dict trick), and three JS files with no
  build step and no framework (`js/api.js` the fetch wrapper and formatters, `js/ui.js` the shared
  card/badge/gauge/scale/Chart.js helpers, `js/pages.js` the six page renderers, `js/app.js` the
  theme switch and hash-based router). Every number, formula and SHAP-reason rendering rule was
  carried over unchanged from the Streamlit version so nothing about what the app says changed —
  only how it looks and what renders it.
- Made the dashboard and the API the same server: added a `StaticFiles` mount at the bottom of
  `src/lia/api.py` serving `web/` at `/` (registered after every API route, so it only catches
  requests nothing else matched), plus CORS middleware for anyone who opens the page from a
  different origin during development. The browser's JS now calls `/loan/score`, `/monitor/meta/
  loan` etc. directly with relative paths — same-origin, so no `API_URL` configuration and no CORS
  issue in the normal case. This also meant one fewer moving part: the separate `app` Streamlit
  container in `docker-compose.yml` is gone entirely, and so is the `APP_PORT` variable — the
  dashboard now lives at the same URL as the API (`API_PORT`, default 8001).
- Removed the Streamlit app completely: `app/streamlit_app.py`, `app/ui.py`, `app/style.py`,
  `app/.streamlit/`, and the Streamlit-specific `scripts/test_app_render.py` AppTest harness. Dropped
  `streamlit` and `plotly` from `requirements.txt` (nothing else in the backend used them; `requests`
  was also dropped since the old server-side `ui.py` API client no longer exists — the browser calls
  the API directly now). Updated `docker/Dockerfile` (copies `web/` instead of `app/`, drops the
  now-unused port 8501), `Makefile`, `.env.example`, and this document's own Problem Statement,
  Architecture, Roadmap, Q&A and Interview Points sections, which still described the Streamlit app
  as current.
- Validated the same way as the Streamlit rebuild: ran the real FastAPI service against the
  already-trained models, confirmed `/` serves the dashboard and every API route still resolves
  correctly ahead of the static-file catch-all, and re-ran the full backend test suite as a
  regression check (still 22/22, untouched — this was a presentation-layer swap only).

### 2026-11-21 — Futuristic Light/Dark/Mix UI redesign
- The user asked for the app to look "futuristic," matching two reference fintech-dashboard images,
  with fresh colours and three selectable modes (Light, Dark, Mix) — explicitly as a presentation
  layer change only, with the ML backend (FastAPI + the trained pipelines) working exactly as
  before.
- Rebuilt `app/style.py` from scratch around a mutable, shared colour-palette object (`PAL`,
  `STATUS`) instead of frozen per-theme constants. The reason this matters technically: Streamlit
  only executes a module's top-level code once per process, so a module that does
  `from style import SOME_COLOUR` at import time would be stuck with whichever theme was active on
  the very first page load — switching themes later would do nothing for that module. Mutating a
  shared dict object in place (`PAL.clear(); PAL.update(new_theme)`) instead of reassigning it means
  every module that already imported `PAL` sees the new values immediately, because they all hold a
  reference to the same object. Defined three palettes — `LIGHT` (clean fintech-dashboard white/
  teal), `DARK` (near-black with teal/violet glow accents), `MIX` (a vibrant purple/pink/indigo
  gradient look, closer to the second reference image) — each a flat dict of every colour the app's
  CSS needs (background, card, accent, status colours, sidebar, etc.), plus a `build_css()` function
  that renders the whole stylesheet from whichever palette is active.
- Found the real, version-stable CSS selectors for Streamlit's native page-navigation sidebar by
  reading the installed Streamlit frontend's own JS bundle directly (`stSidebarNav`,
  `stSidebarNavItems`, `stSidebarNavLink`, and the `aria-current="page"` attribute for the active
  page) rather than guessing at fragile, auto-generated class names — used to restyle the sidebar
  nav to match the active theme and to switch the app from a top nav bar to a sidebar nav
  (`st.navigation(..., position="sidebar")`), closer to both reference images' left-hand nav.
- Refactored `app/ui.py` (every chart/gauge/badge/card helper) and all six pages in
  `app/streamlit_app.py` to read colours from `PAL["..."]` at the moment each widget is drawn,
  instead of importing frozen colour constants — the mechanical change the live-switching design
  above depends on.
- Added a Light/Dark/Mix segmented-control switch at the top of the sidebar, persisted in the URL's
  `?theme=` query parameter (so a themed screenshot or link stays themed on reload) and in the
  widget's own session state (so clicking it mid-session needs no reload).
- Extended the stylesheet to recolour Streamlit's own native widgets (inputs, selects, buttons,
  labels, forms) inside the sidebar and main content so they track whichever theme is active,
  rather than fighting it — with one documented exception: `st.dataframe` renders its grid to a
  `<canvas>`, which only follows Streamlit's native `.streamlit/config.toml` theme, not CSS, so the
  Applications Queue, Model Lab and Monitoring tables keep one consistent look (close to Dark, the
  app's default) across all three in-app themes. Noted as a platform limitation in the Honest
  Limitations section above, not hidden.
- Validated the whole redesign against the **real** backend rather than mocked data: started the
  actual FastAPI service against the already-trained models and processed data already in the repo,
  then used Streamlit's own `AppTest` headless-testing framework to render all 6 pages under all 3
  themes (18 combinations) against that live API, catching any exception. All 18 passed. Re-ran the
  full backend test suite afterwards as a regression check (22/22 passed, unchanged) to confirm the
  redesign touched only the presentation layer. Added the small test script itself
  (`scripts/test_app_render.py`) to the repo so this check can be re-run after any future UI change.
- Synced the four changed files (`app/style.py`, `app/ui.py`, `app/streamlit_app.py`,
  `app/.streamlit/config.toml`) plus the new test script to the real project folder.

### 2026-11-21 — H2O.ai removed; host ports made configurable
- Resolved the open scope question below: after walking through exactly where XGBoost and H2O.ai
  each sit in the project, the user decided to **remove H2O.ai** entirely rather than keep it
  alongside XGBoost — XGBoost appears far more often in Data Scientist job descriptions, and H2O
  was always a separate, optional add-on rather than something XGBoost depended on, so removing it
  put nothing else at risk.
- Deleted `src/lia/h2o_train.py`, `tests/test_h2o_train.py`, `requirements-h2o.txt` and
  `docker/Dockerfile.h2o`; removed the `h2o-compare` service from `docker-compose.yml`, the `h2o`/
  `h2o-docker` Makefile targets, the `h2o-smoke` CI job, and the H2O→Vertex AI row from
  `docs/GCP_MAPPING.md`. Rewrote every H2O-related passage in the README (title sentence,
  architecture diagram, Quick Start, results section/table, repository map, test count, limitations
  list) and updated `docs/DATA_CARD.md`'s ProjectPro-titles table with an "Implemented?" column
  marking title #4 as deliberately dropped. Full test suite re-run afterwards: 22/22 passing (down
  from 24, the two H2O-specific tests removed with the module).
- Separately, the user flagged that two other local projects were already running on this laptop
  and asked whether this project's ports could collide. Checked the actual ports this project uses
  (MySQL 3306, API 8001, Streamlit app 8501, MLflow UI 5000) against what was already in use —
  no collision — but made all four host ports overridable via `.env` (`MYSQL_PORT`, `API_PORT`,
  `APP_PORT`, `MLFLOW_UI_PORT`) in `docker-compose.yml` regardless, so a future collision on a
  different machine is a one-line `.env` edit instead of hand-editing `docker-compose.yml`.
- (This entry was written up after the fact, once the blueprint document itself was found to be out
  of sync with the already-completed removal — the code, README and Data Card changes above were
  real and already in the repo; only this file's own story had fallen behind.)

### 2026-11-21 — README fix, MLflow logbook naming confirmed
- Confirmed with the user that "MLflow logbook" is an accurate plain-language name for the
  experiment-tracking system added on 2026-11-21 (see below).
- Started the Kaggle seed-dataset licence check (see Data Card §8.6): confirmed the
  `mirichoi0218/insurance` (Medical Cost Personal Datasets) seed is described by its own author as
  public domain, and the Analytics Vidhya 614-row loan prediction seed is mirrored on OpenML tagged
  CC0: Public Domain. Could not independently confirm the licence tags for the two remaining seeds
  (the ~4,269-row CIBIL/assets set and the ~45,000-row credit-risk set) because Kaggle's dataset
  pages require a logged-in session this tool can't reach. Practical mitigation already in place
  regardless: the repo never commits the raw downloaded seed CSVs, only the synthetic generator
  output, which contains no verbatim seed rows — so redistribution risk is effectively zero either
  way; the remaining open item is purely about citing sources correctly in the Data Card.

### 2026-11-21 — Living blueprint document created; H2O/XGBoost scope question raised
- Created this document as the single, continuously-updated story of the project, covering problem
  statement, objectives, workflow, architecture, roadmap, real-world Q&A and interview points, per
  the plan to keep it current as work progresses rather than writing it once at the end.
- Open question with the user (resolved in the next entry above): whether to keep the H2O.ai
  independent-comparison module alongside XGBoost, or drop it in favour of XGBoost alone.

### 2026-11-21 — README/registry fix, H2O.ai comparison, MLflow tracking
- Fixed a real inconsistency: the README claimed XGBoost (tuned) was the loan champion; the actual
  Docker-retrained `models/registry.json` showed Logistic Regression (GLM, no SMOTE). Rewrote the
  entire loan comparison table with the correct numbers for every model, not just the champion row.
  Corrected the insurance champion's test metrics the same way.
- Built `src/lia/h2o_train.py`: an independent H2O.ai comparison (GLM / DRF / GBM / XGBoost /
  AutoML) on the loan problem, using the same time-based split, Platt calibration and cost-based
  money rule as the sklearn comparison, so the two are genuinely comparable. Ran it for real
  (fast and full modes): H2O's own best model is also a GLM, and it does not beat the sklearn/
  XGBoost champion by more than 2% — an independent cross-engine validation of the earlier finding,
  not a contradiction. Shipped as its own Docker image (`docker/Dockerfile.h2o`) behind a Compose
  `tools` profile so it never runs as part of the default `docker compose up`. *(This module, and
  the "two engines" framing below, was later deliberately removed — see the H2O removal entry
  above.)*
- Built `src/lia/tracking.py`: a fail-soft MLflow wrapper (SQLite-backed tracking URI, not a
  running `mlflow server`) used by `loan_train.py`, `insurance_train.py` and `h2o_train.py` to log
  every candidate model's params/metrics and register both production champions in the MLflow
  Model Registry. Solved an `skops` untrusted-types issue that was silently blocking model logging
  for every non-trivial model type (XGBoost, SMOTE pipelines, tree-based models).
- Added `h2o`, `h2o-docker` and `mlflow-ui` Makefile targets, a non-blocking H2O smoke-test CI job,
  and updated `.gitignore` for the new `mlflow/` directory.
- All 24 tests pass (22 original + 2 new for the H2O module).
