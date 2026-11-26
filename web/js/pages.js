/* One render function per page, mirroring the project's original Streamlit app 1:1 so every number
   and every reason code means exactly the same thing - only the presentation layer changed. */

const main = () => document.getElementById("main");

const PURPOSES = ["Home Loan", "Personal Loan", "Auto Loan", "Education Loan", "Business Loan",
                  "Home Improvement", "Medical Emergency", "Debt Consolidation"];
const EXAMPLES = {
  "Strong salaried applicant": { age: 34, emp: "Salaried", inc: 95000, co: 0, amt: 1800000, ten: 180, purpose: "Home Loan", cibil: 790, delq: 0, exist: 8000, coll: 2400000, tier: 1, yrs: 10, assets: 900000 },
  "Borderline self-employed": { age: 38, emp: "Self-Employed", inc: 42000, co: 0, amt: 900000, ten: 84, purpose: "Business Loan", cibil: 665, delq: 1, exist: 7000, coll: 0, tier: 2, yrs: 6, assets: 300000 },
  "High-risk personal loan": { age: 27, emp: "Salaried", inc: 24000, co: 0, amt: 500000, ten: 48, purpose: "Personal Loan", cibil: 585, delq: 3, exist: 6500, coll: 0, tier: 3, yrs: 3, assets: 50000 },
};
function reasonUnit(it) { return `${it.impact > 0 ? "+" : ""}${it.impact.toFixed(2)}`; }

function errorCard(msg) { return `<div class="err-box">${esc(msg)}</div>`; }

/* ======================================================================================= OVERVIEW */
async function renderOverview() {
  main().innerHTML = `<div class="loading">Loading overview…</div>`;
  const [loan, ins, queue, trend] = await Promise.all([
    apiCached("/monitor/meta/loan"), apiCached("/monitor/meta/insurance"), apiCached("/loan/queue?limit=1"),
    apiCached("/monitor/drift-trend/loan"),
  ]);
  if (loan?.error || ins?.error) { main().innerHTML = errorCard(loan?.error || ins?.error); return; }

  let html = topbar("Loan Eligibility & Insurance Pricing", "Gradient Boosting · XGBoost · MySQL · Docker",
    [["Loan model", loan.name.split(" (")[0]], ["Premium model", "XGBoost"]],
    "Synthetic data seeded from public datasets · personal portfolio project",
    { id: "ovOpenQueue", icon: "&#128228;", label: "Open queue" });

  html += `<div class="hero" style="margin-bottom:18px">
    <div style="display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap">
      <div style="max-width:720px"><div class="big" style="font-size:2rem">One platform, two decisions</div>
      <p style="color:var(--muted);margin:.5rem 0 0 0;line-height:1.55">
      <b style="color:var(--text)">Lending:</b> approve a loan only when the expected interest outweighs the expected loss,
      and say why in plain language.<br>
      <b style="color:var(--text)">Insurance:</b> price an individual health policy and show how uncertain that price is.</p></div>
      <div style="align-self:center">${badge("ELIGIBLE")} ${badge("MANUAL_REVIEW")} ${badge("NOT_ELIGIBLE")}</div>
    </div>
    <div style="height:14px"></div>
    <div class="flow">
      <div class="st"><b>Raw CSV</b><span>messy landing files</span></div><div class="ar">&#10140;</div>
      <div class="st"><b>MySQL</b><span>raw → clean tables, SQL views</span></div><div class="ar">&#10140;</div>
      <div class="st"><b>Pipelines</b><span>impute · encode · SMOTE</span></div><div class="ar">&#10140;</div>
      <div class="st"><b>Models</b><span>GLM · RF · GBM · XGBoost</span></div><div class="ar">&#10140;</div>
      <div class="st"><b>FastAPI</b><span>decision + SHAP reasons</span></div><div class="ar">&#10140;</div>
      <div class="st"><b>This app</b><span>HTML/JS</span></div><div class="ar">&#10140;</div>
      <div class="st"><b>Monitoring</b><span>PSI drift · gated retrain</span></div>
    </div></div>`;

  const sparkId = "ovSpark" + Math.random().toString(36).slice(2, 8);
  if (loan.metrics && ins.metrics) {
    const t = loan.metrics.test, it = ins.metrics.test;
    html += `<div class="grid grid-5">
      <div class="kpi"><div class="k">Loan ROC-AUC (test)</div><div class="v">${t.roc_auc.toFixed(3)}</div><div class="s">Aug–Oct 2026, unseen</div>
        ${miniScale(t.roc_auc, 0.5, 1, cssVar("--accent"))}</div>
      <div class="kpi"><div class="k">Loan MCC (test)</div><div class="v">${t.mcc_at_global_threshold.toFixed(3)}</div><div class="s">at the validation threshold</div>
        ${miniScale(t.mcc_at_global_threshold, 0, 1, cssVar("--accent"))}</div>
      <div class="kpi"><div class="k">Premium R² (test)</div><div class="v">${it.r2.toFixed(2)}</div><div class="s">linear baseline ${it.baseline_r2.toFixed(2)}</div>
        ${miniScale(it.r2, 0, 1, cssVar("--accent"))}</div>
      <div class="kpi"><div class="k">Error vs linear</div><div class="v">-${Math.round(it.rmse_improvement_vs_linear * 100)}%</div><div class="s">RMSE reduction</div>
        ${miniScale(it.rmse_improvement_vs_linear * 100, 0, 50, cssVar("--good"))}</div>
      <div class="kpi"><div class="k">New applications</div><div class="v">${queue?.n ?? "-"}</div><div class="s">waiting in the queue</div>
        <div class="spark"><canvas id="${sparkId}"></canvas></div></div>
    </div>`;
  }

  html += `<div style="height:16px"></div><div class="grid grid-3">
    <a class="card-link" href="#loan"><div class="card hoverable">${cardTitle("Loan eligibility", "Score one applicant: probability of default, break-even decision, reason codes and the largest amount that would be approved.")}<div class="small" style="margin-top:8px;color:var(--accent)">Open loan eligibility &#8594;</div></div></a>
    <a class="card-link" href="#insurance"><div class="card hoverable">${cardTitle("Insurance pricing", "Quote a health premium with an 80% range, what drives it, and how BMI and tobacco change the price.")}<div class="small" style="margin-top:8px;color:var(--accent)">Open insurance pricing &#8594;</div></div></a>
    <a class="card-link" href="#lab"><div class="card hoverable">${cardTitle("Model lab & monitoring", "Fair comparison of 7 models on a time split, drift on every feature and the retrain history.")}<div class="small" style="margin-top:8px;color:var(--accent)">Open model lab &#8594;</div></div></a>
  </div>`;
  main().innerHTML = html;
  document.getElementById("ovOpenQueue").onclick = () => { location.hash = "#queue"; };
  if (trend && !trend.error && trend.months) sparkline(sparkId, trend.months.map((m) => m.n), cssVar("--accent"));
}

/* ======================================================================================= LOAN */
function loanFormHTML(selected) {
  const ex = EXAMPLES[selected];
  const opt = (arr, val) => arr.map((v) => `<option value="${v}"${String(v) === String(val) ? " selected" : ""}>${v}</option>`).join("");
  return `<div class="card">
    ${cardTitle("Applicant", "Start from an example, then adjust any field")}
    <label class="field">Start from an example</label>
    <select id="loanExample">${Object.keys(EXAMPLES).map((k) => `<option${k === selected ? " selected" : ""}>${k}</option>`).join("")}</select>

    <div class="small" style="margin-top:14px;font-weight:600;letter-spacing:.05em">PERSONAL</div>
    <label class="field">Age</label><input type="number" id="loanAge" min="18" max="70" value="${ex.age}">
    <label class="field">Employment</label><select id="loanEmp">${opt(["Salaried", "Self-Employed"], ex.emp)}</select>
    <label class="field">Years employed</label><input type="number" id="loanYears" min="0" max="50" value="${ex.yrs}">
    <label class="field">Education</label><select id="loanEdu">${opt(["Graduate", "Not Graduate"], "Graduate")}</select>
    <label class="field">Dependents (3 = 3 or more)</label><input type="number" id="loanDep" min="0" max="3" value="0">
    <label class="field">Home ownership</label><select id="loanHome">${opt(["Rented", "Owned", "Mortgaged", "Other"], "Rented")}</select>
    <label class="field">City tier</label><select id="loanTier">${opt([1, 2, 3], ex.tier)}</select>

    <div class="small" style="margin-top:14px;font-weight:600;letter-spacing:.05em">INCOME &amp; CREDIT</div>
    <label class="field">Monthly income (Rs.)</label><input type="number" id="loanInc" min="5000" max="700000" step="1000" value="${ex.inc}">
    <label class="field">Co-applicant income (Rs.)</label><input type="number" id="loanCo" min="0" max="700000" step="1000" value="${ex.co}">
    <label class="field">Existing EMIs (Rs./month)</label><input type="number" id="loanExist" min="0" max="500000" step="500" value="${ex.exist}">
    <label class="field">Existing loans</label><input type="number" id="loanNLoans" min="0" max="10" value="1">
    <label class="field">Late payments (24 months)</label><input type="number" id="loanDelq" min="0" max="10" value="${ex.delq}">
    <label class="field" style="display:flex;align-items:center;gap:8px;text-transform:none;letter-spacing:0">
      <input type="checkbox" id="loanNewCred" style="width:auto"> New to credit (no bureau score)</label>
    <label class="field">CIBIL score</label><input type="number" id="loanCibil" min="300" max="900" value="${ex.cibil}">

    <div class="small" style="margin-top:14px;font-weight:600;letter-spacing:.05em">LOAN</div>
    <label class="field">Purpose</label><select id="loanPurpose">${opt(PURPOSES, ex.purpose)}</select>
    <label class="field">Loan amount (Rs.)</label><input type="number" id="loanAmt" min="20000" max="30000000" step="10000" value="${ex.amt}">
    <label class="field">Tenure (months)</label><input type="number" id="loanTen" min="6" max="360" step="6" value="${ex.ten}">
    <label class="field">Collateral value (Rs., 0 if none)</label><input type="number" id="loanColl" min="0" max="100000000" step="50000" value="${ex.coll}">
    <label class="field">Declared assets (Rs.)</label><input type="number" id="loanAssets" min="0" max="500000000" step="50000" value="${ex.assets}">

    <button class="btn" id="loanSubmit" style="margin-top:16px">Assess eligibility</button>
  </div>`;
}

function loanBodyFromForm() {
  const tier = Number(document.getElementById("loanTier").value);
  const newCred = document.getElementById("loanNewCred").checked;
  const body = {
    age: Number(document.getElementById("loanAge").value),
    employment_type: document.getElementById("loanEmp").value,
    years_employed: Number(document.getElementById("loanYears").value),
    dependents: Number(document.getElementById("loanDep").value),
    education: document.getElementById("loanEdu").value,
    home_ownership: document.getElementById("loanHome").value,
    city_tier: tier,
    monthly_income_inr: Number(document.getElementById("loanInc").value),
    co_applicant_income_inr: Number(document.getElementById("loanCo").value),
    existing_loans_count: Number(document.getElementById("loanNLoans").value),
    existing_emi_inr: Number(document.getElementById("loanExist").value),
    past_delinquency_24m: Number(document.getElementById("loanDelq").value),
    loan_purpose: document.getElementById("loanPurpose").value,
    loan_amount_inr: Number(document.getElementById("loanAmt").value),
    tenure_months: Number(document.getElementById("loanTen").value),
    collateral_value_inr: Number(document.getElementById("loanColl").value),
    residential_assets_inr: Number(document.getElementById("loanAssets").value),
    property_area: { 1: "Urban", 2: "Semiurban", 3: "Rural" }[tier],
  };
  if (!newCred) body.cibil_score = Number(document.getElementById("loanCibil").value);
  return body;
}

function wireLoanForm(onSubmit) {
  document.getElementById("loanExample").onchange = (e) => {
    main().querySelector(".grid-2 > div:first-child").outerHTML = loanFormHTML(e.target.value);
    wireLoanForm(onSubmit);
  };
  const newCredBox = document.getElementById("loanNewCred");
  const cibilInput = document.getElementById("loanCibil");
  newCredBox.onchange = () => { cibilInput.disabled = newCredBox.checked; cibilInput.style.opacity = newCredBox.checked ? .4 : 1; };
  document.getElementById("loanSubmit").onclick = () => onSubmit(loanBodyFromForm());
}

function decisionHTML(res, compact = false) {
  const p = res.probability_of_default, thr = res.approval_threshold, dec = res.decision;
  const color = statusColor(dec);
  const gap = thr ? (thr - p) / thr : 0;
  const note = res.policy_flags.length ? "A lending policy rule applies before the model"
    : `Risk is <b>${Math.abs(gap * 100).toFixed(0)}% ${gap > 0 ? "below" : "above"}</b> the break-even level for this loan`;
  const vmax = Math.max(0.6, thr * 2.2);
  const gaugeId = "gaugeLoan" + Math.random().toString(36).slice(2, 8);
  const radarId = "radarLoan" + Math.random().toString(36).slice(2, 8);

  let html = !compact ? topbar("Loan Eligibility", "Calibrated default risk · break-even decision · SHAP reasons",
    [["Probability of default", pct(p)], ["Break-even PD", pct(thr)]], note) : "";

  html += `<div class="hero" style="margin-bottom:18px"><div class="grid" style="grid-template-columns:${compact ? "1fr 1.2fr" : "1fr 1.45fr 1fr"};gap:20px">
    <div>${cardTitle("Applicant profile", "0-100 on each dimension (descriptive, not model inputs)")}
      <div style="position:relative"><canvas id="${radarId}"></canvas></div></div>
    <div>
      <div id="${gaugeId}-slot">${gaugeSVG({ value: p, vmax, centerText: pct(p), centerSub: "probability of default", color, marker: thr, markerLabel: `break-even ${pct(thr, 0)}`, ticks: [[0, "0%"], [vmax / 2, pct(vmax / 2, 0)], [vmax, pct(vmax, 0)]] })}</div>
      <div style="text-align:center;margin-top:-6px">${badge(dec)}
      <div class="small" style="margin-top:8px">risk band <b style="color:var(--text)">${esc(res.risk_band)}</b></div></div>
      ${res.policy_flags.map((f) => `<div class="warn-box">${esc(f)}</div>`).join("")}
    </div>
    ${compact ? "" : `<div>
      <div class="big">${res.cibil_score ? Math.round(res.cibil_score) : "New to credit"}</div>
      ${cardTitle("Credit score", "CIBIL, 300-900")}
      ${scale(res.cibil_score, 300, 900, [[0, "var(--bad)"], [.45, "var(--warn)"], [.72, "var(--good)"], [1, "var(--good)"]], ["Poor", "Fair", "Excellent"])}
      <div style="height:14px"></div>
      ${cardTitle("Affordability (FOIR)", "EMIs as a share of income · policy limit 85%")}
      <div class="big">${pct(res.foir, 0)}</div>
      ${scale(res.foir, 0, 1, [[0, "var(--good)"], [.5, "var(--good)"], [.7, "var(--warn)"], [.85, "var(--bad)"], [1, "var(--bad)"]], ["0%", "50%", "100%"])}
      <div style="height:10px"></div>
      ${row2("Loan-to-income", res.loan_to_income ? res.loan_to_income.toFixed(1) + "x annual" : "-")}
    </div>`}
  </div></div>`;

  html += `<div class="grid" style="grid-template-columns:${compact ? "1fr" : "1.25fr 1fr 1fr"};gap:18px">
    <div class="card">${cardTitle("Why this decision", "SHAP contributions in log-odds of default · longer bar = stronger effect")}
      <div class="grid grid-2">
        <div><div class="small" style="color:var(--bad)">PUSHING RISK UP</div>${reasons(res.reasons_against, cssVar("--bad"), reasonUnit)}</div>
        <div><div class="small" style="color:var(--good)">PULLING RISK DOWN</div>${reasons(res.reasons_for, cssVar("--good"), reasonUnit)}</div>
      </div></div>`;

  if (!compact) {
    const e = res.economics;
    html += `<div class="card">${cardTitle("Loan economics", "What the bank earns or loses on this loan")}
      ${row2("Loan amount", inr(res.loan_amount_inr))}${row2("EMI / month", inr(res.emi_inr))}
      ${row2("Interest rate", res.interest_rate_pct.toFixed(2) + "%")}${row2("Tenure", (res.tenure_months / 12).toFixed(1) + " years")}
      ${row2("Net interest if repaid", inr(e.net_interest_if_repaid), cssVar("--good"))}
      ${row2("Loss if it defaults", inr(e.loss_if_default), cssVar("--bad"))}
      ${row2("Expected value", inr(e.expected_net_value), e.expected_net_value > 0 ? cssVar("--good") : cssVar("--bad"))}</div>`;

    let msg;
    if (dec === "ELIGIBLE") msg = `Expected default risk of ${pct(p)} is comfortably below the ${pct(thr)} level at which this loan stops being profitable.`;
    else if (dec === "MANUAL_REVIEW") msg = `Risk of ${pct(p)} is close to the ${pct(thr)} break-even level, so a credit officer should check documents before deciding.`;
    else msg = `Risk of ${pct(p)} is well above the ${pct(thr)} break-even level: on average this loan would lose money.`;
    if (res.policy_flags.length) msg = "The application breaks a lending policy, so it is declined regardless of the model score.";
    let sugHtml = "";
    const sug = res.suggestion;
    if (sug && sug.max_eligible_amount_inr) {
      sugHtml = `<div class="callout"><b>What would get approved?</b><br>A loan of up to <b>${inr(sug.max_eligible_amount_inr)}</b>
        (EMI ${inr(sug.emi_inr)}) with the same tenure and profile, at PD ${pct(sug.probability_of_default)}.</div>`;
    } else if (sug !== undefined && sug !== null) {
      sugHtml = `<div class="callout">Reducing the amount alone would not make this application eligible; the credit profile is the main issue.</div>`;
    }
    html += `<div class="card">${cardTitle("What it means")}<p style="font-size:.9rem;line-height:1.5">${msg}</p>${sugHtml}
      <div class="small" style="margin-top:8px">${esc(res.model.name)} · v${esc(res.model.version)}</div></div>`;
  }
  html += `</div>`;
  return { html, radarId };
}

async function renderLoan() {
  main().innerHTML = `<div class="grid grid-2" style="align-items:start">${loanFormHTML(Object.keys(EXAMPLES)[0])}<div id="loanResult"><div class="loading">Scoring…</div></div></div>`;
  const resultPane = () => document.getElementById("loanResult");
  const score = async (body) => {
    resultPane().innerHTML = `<div class="loading">Scoring…</div>`;
    const res = await api("POST", "/loan/score", body);
    if (res.error) { resultPane().innerHTML = errorCard(res.error); return; }
    const { html, radarId } = decisionHTML(res, false);
    resultPane().innerHTML = html;
    radarChart(radarId, res.profile);
  };
  wireLoanForm(score);
  await score(loanBodyFromForm());
}

/* ======================================================================================= QUEUE */
const DECISION_INITIALS = { ELIGIBLE: "EL", MANUAL_REVIEW: "MR", NOT_ELIGIBLE: "NE" };

async function renderQueue() {
  main().innerHTML = `<div class="loading">Loading queue…</div>`;
  const data = await apiCached("/loan/queue");
  if (data.error) { main().innerHTML = errorCard(data.error); return; }
  const s = data.summary;
  let html = topbar("Applications Queue", "New, unlabelled applications scored by the champion model",
    [["Eligible", s.ELIGIBLE || 0], ["Review", s.MANUAL_REVIEW || 0], ["Not eligible", s.NOT_ELIGIBLE || 0]],
    `${data.n} applications received 1–5 Nov 2026`,
    { id: "queueRefresh", icon: "&#8635;", label: "Refresh" });

  html += `<div style="display:flex;gap:12px;align-items:center;margin-bottom:14px;flex-wrap:wrap">
    <div class="seg-ctl" id="queueFilter">
      <button class="active" data-v="All">All</button><button data-v="ELIGIBLE">Eligible</button>
      <button data-v="MANUAL_REVIEW">Manual review</button><button data-v="NOT_ELIGIBLE">Not eligible</button>
    </div>
    <input type="text" id="queueSearch" placeholder="Search by application ID…" style="max-width:240px">
  </div>`;

  html += `<div class="grid" style="grid-template-columns:1fr 1.25fr;gap:18px;align-items:start">
    <div class="card"><div id="queueListHead">${cardTitle("Applications", `${data.applications.length} shown · click a row to open it`)}</div>
      <div id="queueBody" style="max-height:620px;overflow:auto"></div></div>
    <div class="card" id="queueDetail"><div class="loading">Select an application…</div></div>
  </div>`;
  main().innerHTML = html;
  document.getElementById("queueRefresh").onclick = () => { clearApiCache("/loan/queue"); renderQueue(); };

  let rows = data.applications;
  const renderRows = () => {
    const filter = document.querySelector("#queueFilter button.active").dataset.v;
    const q = document.getElementById("queueSearch").value.trim().toLowerCase();
    let filtered = rows.filter((r) => (filter === "All" || r.decision === filter) && (!q || r.application_id.toLowerCase().includes(q)));
    document.querySelector("#queueListHead .csub").textContent = `${filtered.length} shown · click a row to open it`;
    document.getElementById("queueBody").innerHTML = filtered.map((r) => `
      <div class="qrow" data-id="${esc(r.application_id)}">
        ${avatarBadge(DECISION_INITIALS[r.decision] || "?", statusColor(r.decision))}
        <div style="flex:1;min-width:0">
          <div class="qr-id">${esc(r.application_id)}</div>
          <div class="qr-sub">${esc(r.loan_purpose)} · CIBIL ${r.cibil_score ? Math.round(r.cibil_score) : "NTC"}</div>
        </div>
        <div style="text-align:right">
          <div class="qr-amt">${inr(r.loan_amount_inr, true)}</div>
          <div class="qr-sub qr-pd">PD ${(r.pd * 100).toFixed(1)}%</div>
        </div>
      </div>`).join("") || `<div class="small" style="padding:14px 4px">No applications match the filter.</div>`;
    document.querySelectorAll(".qrow[data-id]").forEach((row) => {
      row.onclick = () => { document.querySelectorAll(".qrow").forEach((t) => t.classList.remove("selected")); row.classList.add("selected"); openApp(row.dataset.id); };
    });
    if (filtered.length) document.querySelector(".qrow[data-id]").click();
    else document.getElementById("queueDetail").innerHTML = `<div class="small">No applications match the filter.</div>`;
  };
  const openApp = async (id) => {
    document.getElementById("queueDetail").innerHTML = `<div class="loading">Loading…</div>`;
    const res = await api("GET", `/loan/queue/${encodeURIComponent(id)}`);
    if (res.error) { document.getElementById("queueDetail").innerHTML = errorCard(res.error); return; }
    const a = res.applicant;
    const { html: body, radarId } = decisionHTML(res, true);
    const moreId = "qMore" + Math.random().toString(36).slice(2, 8);
    document.getElementById("queueDetail").innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap">
        <div><div class="ctitle" style="font-size:1.1rem">${esc(a.application_id)}</div>
        <div class="csub" style="margin-bottom:0">${esc(a.loan_purpose)} · ${esc(a.employment_type)} · ${esc(a.city || "-")} · via ${esc(a.channel)}</div></div>
        ${badge(res.decision)}
      </div>
      <div class="small" style="margin:6px 0 2px">Age ${a.age ? Math.round(a.age) : "-"} · income ${inr(a.monthly_income_inr)}/month · asks ${inr(a.loan_amount_inr)} over ${Math.round(a.tenure_months || 0)} months</div>
      ${tileRow([["Loan amount", inr(res.loan_amount_inr, true)], ["EMI / month", inr(res.emi_inr, true)], ["Interest rate", res.interest_rate_pct.toFixed(2) + "%"]])}
      <button class="toggle-more" id="${moreId}-btn">View full assessment &#9662;</button>
      <div class="more-wrap" id="${moreId}">${body}</div>`;
    radarChart(radarId, res.profile, 230);
    const wrap = document.getElementById(moreId), btn = document.getElementById(`${moreId}-btn`);
    btn.onclick = () => {
      const open = wrap.classList.toggle("open");
      btn.innerHTML = open ? "Hide full assessment &#9652;" : "View full assessment &#9662;";
    };
  };
  document.querySelectorAll("#queueFilter button").forEach((b) => b.onclick = () => {
    document.querySelectorAll("#queueFilter button").forEach((x) => x.classList.remove("active")); b.classList.add("active"); renderRows();
  });
  document.getElementById("queueSearch").oninput = renderRows;
  renderRows();
}

/* ======================================================================================= INSURANCE */
function insFormHTML() {
  const opt = (arr, val) => arr.map((v) => `<option value="${v}"${String(v) === String(val) ? " selected" : ""}>${typeof v === "number" ? inr(v, true) : v}</option>`).join("");
  return `<div class="card">${cardTitle("Policyholder")}
    <label class="field">Age</label><input type="number" id="insAge" min="18" max="70" value="35">
    <label class="field">Sex</label><select id="insSex">${opt(["Male", "Female"], "Male")}</select>
    <div class="grid grid-2" style="gap:10px">
      <div><label class="field">Height (cm)</label><input type="number" id="insHeight" min="120" max="215" value="170"></div>
      <div><label class="field">Weight (kg)</label><input type="number" id="insWeight" min="25" max="250" value="74"></div>
    </div>
    <label class="field">Tobacco use</label><select id="insSmoker">${opt(["No", "Yes"], "No")}</select>
    <label class="field">Pre-existing condition</label><select id="insPed">${opt(["No", "Yes"], "No")}</select>
    <label class="field">Dependent children</label><input type="number" id="insChildren" min="0" max="6" value="1">
    <label class="field">Occupation</label><select id="insOcc">${opt(["IT / Office", "Business Owner", "Government", "Healthcare", "Manufacturing", "Professional (CA/Lawyer/Architect)"], "IT / Office")}</select>
    <label class="field">Cover (sum insured)</label><select id="insCover">${opt([300000, 500000, 1000000, 1500000, 2500000], 500000)}</select>
    <label class="field">Zone</label><select id="insZone">${opt(["West", "North", "South", "East", "Central"], "West")}</select>
    <label class="field">City tier</label><select id="insTier">${opt([1, 2, 3], 1)}</select>
    <button class="btn" id="insSubmit" style="margin-top:16px">Get premium</button></div>`;
}
function insBodyFromForm() {
  return {
    age: Number(document.getElementById("insAge").value), sex: document.getElementById("insSex").value,
    smoker: document.getElementById("insSmoker").value, pre_existing_condition: document.getElementById("insPed").value,
    height_cm: Number(document.getElementById("insHeight").value), weight_kg: Number(document.getElementById("insWeight").value),
    children: Number(document.getElementById("insChildren").value), occupation: document.getElementById("insOcc").value,
    sum_assured_inr: Number(document.getElementById("insCover").value), zone: document.getElementById("insZone").value,
    city_tier: Number(document.getElementById("insTier").value),
  };
}

async function renderInsurance() {
  main().innerHTML = `<div class="grid grid-2" style="align-items:start">${insFormHTML()}<div id="insResult"><div class="loading">Quoting…</div></div></div>`;
  const resultPane = () => document.getElementById("insResult");
  const quote = async (body) => {
    resultPane().innerHTML = `<div class="loading">Quoting…</div>`;
    const r = await api("POST", "/insurance/quote", body);
    if (r.error) { resultPane().innerHTML = errorCard(r.error); return; }
    const curveId = "insCurve" + Math.random().toString(36).slice(2, 8);
    resultPane().innerHTML = topbar("Insurance Pricing Forecast", "XGBoost premium prediction · 80% error range · SHAP drivers",
      [["Annual premium", inr(r.annual_premium_inr)], ["Per month", inr(r.monthly_equivalent_inr)]],
      `Higher than <b>${r.percentile.toFixed(0)}%</b> of health policyholders in the training data`) +
      `<div class="hero" style="margin-bottom:18px"><div class="grid" style="grid-template-columns:1.15fr 1.35fr 1fr;gap:20px">
        <div>${cardTitle("What if BMI changes?", "Predicted premium for this person across BMI")}
          <div style="position:relative;height:260px"><canvas id="${curveId}"></canvas></div></div>
        <div>${gaugeSVG({ value: r.percentile, vmax: 100, centerText: inr(r.annual_premium_inr), centerSub: "estimated annual premium", color: cssVar("--accent"), ticks: [[0, "0th"], [50, "50th"], [100, "100th"]] })}
          <div style="text-align:center;margin-top:-6px"><span class="badge" style="color:var(--accent)"><span class="dot"></span>
          ${esc(r.range_confidence)} range ${inr(r.range_low_inr)} – ${inr(r.range_high_inr)}</span>
          <div class="small" style="margin-top:8px">arc = percentile among policyholders</div></div></div>
        <div>${cardTitle("Body mass index", "Asian-Pacific cut-offs")}<div class="big">${r.bmi_used} <span class="chip">${esc(r.bmi_band)}</span></div>
          ${scale(r.bmi_used, 15, 40, [[0, "var(--warn)"], [.14, "var(--good)"], [.32, "var(--good)"], [.5, "var(--warn)"], [.75, "var(--bad)"], [1, "var(--bad)"]], ["15", "23 · 27.5", "40"])}
          <div style="height:12px"></div>${cardTitle("Price range", "80% of similar policies fall in this band")}
          ${scale(r.annual_premium_inr, r.range_low_inr, r.range_high_inr, [[0, "var(--moss)"], [.5, "var(--accent)"], [1, "var(--moss)"]], [inr(r.range_low_inr), "estimate", inr(r.range_high_inr)])}
          ${row2("Tobacco use", r.smoker, r.smoker === "Yes" ? cssVar("--bad") : cssVar("--good"))}</div>
      </div></div>
      <div class="grid grid-3">
        <div class="card">${cardTitle("What raises the premium", "Effect of each factor on the price, from SHAP")}${reasons(r.raises_premium, cssVar("--bad"), (it) => `${it.premium_effect_pct > 0 ? "+" : ""}${it.premium_effect_pct}%`)}</div>
        <div class="card">${cardTitle("What lowers the premium")}${reasons(r.lowers_premium, cssVar("--good"), (it) => `${it.premium_effect_pct > 0 ? "+" : ""}${it.premium_effect_pct}%`)}</div>
        <div class="card">${cardTitle("How good is the estimate?", "Measured on policies from Feb–Oct 2026 that the model never saw")}
          ${row2("Typical error (RMSE)", inr(r.model.test_rmse_inr))}${row2("R² · XGBoost", r.model.test_r2.toFixed(2), cssVar("--good"))}
          ${row2("R² · linear baseline", r.model.linear_baseline_r2.toFixed(2), cssVar("--muted"))}${row2("Segment", esc(r.model.segment))}
          <div class="small" style="margin-top:6px">Premium prediction for one person, not a time-series forecast.</div></div>
      </div>`;
    const accent = cssVar("--accent"), warn = cssVar("--warn"), text = cssVar("--text");
    lineChart(curveId, r.bmi_curve.bmi, [
      { label: "Non-smoker", data: r.bmi_curve.non_smoker, borderColor: accent, backgroundColor: accent, tension: .35, pointRadius: 0, borderWidth: 2.5 },
      { label: "Smoker", data: r.bmi_curve.smoker, borderColor: warn, backgroundColor: warn, borderDash: [4, 3], tension: .35, pointRadius: 0, borderWidth: 2.5 },
    ], { height: 260 });
  };
  document.getElementById("insSubmit").onclick = () => quote(insBodyFromForm());
  await quote(insBodyFromForm());
}

/* ======================================================================================= MODEL LAB */
async function renderLab() {
  main().innerHTML = `<div class="loading">Loading model lab…</div>`;
  const [meta, comp, imeta, icomp] = await Promise.all([
    apiCached("/monitor/meta/loan"), apiCached("/monitor/comparison/loan"),
    apiCached("/monitor/meta/insurance"), apiCached("/monitor/comparison/insurance"),
  ]);
  if (meta.error || comp.error || imeta.error || icomp.error) { main().innerHTML = errorCard("Model reports unavailable. Is the API running?"); return; }
  const champ = meta.name;
  const ch = comp.find((r) => r.model === champ);
  const short = (m) => m.replace("Logistic regression", "LogReg").replace(", BayesSearchCV", "");

  let html = topbar("Model Lab", "Same features, same split, same calibration for every model",
    [["Champion", champ.split(" (")[0]], ["Rule", meta.rule === "breakeven" ? "break-even" : "global"]],
    "Train → Jan 2026 · validate Feb–Jul 2026 · test Aug–Oct 2026 (drift period)",
    { id: "labRefresh", icon: "&#8635;", label: "Refresh" });
  html += kpiRow([
    ["Val ROC-AUC", ch.val_roc_auc.toFixed(3), "validation"], ["Test ROC-AUC", ch.test_roc_auc.toFixed(3), "unseen period"],
    ["Test MCC", ch.test_mcc_at_global_threshold.toFixed(3), "at the validation threshold"],
    ["Test value", inr(ch.test_value_breakeven_rule, true), "net value with the money rule"],
    ["Approve-all value", inr(ch.test_value_approve_all, true), "the baseline to beat"],
  ]);
  html += `<div style="height:16px"></div><div class="grid grid-2">
    <div class="card">${cardTitle("Ranking power · validation ROC-AUC", "How well each model separates good from bad loans")}<div style="height:290px"><canvas id="aucChart"></canvas></div></div>
    <div class="card">${cardTitle("Business value · validation (Rs crore)", "Interest on good loans minus losses on bad ones, best rule")}<div style="height:290px"><canvas id="valChart"></canvas></div></div>
  </div><div style="height:18px"></div>
  <div class="card">${cardTitle("Full comparison", "The champion has the best validation value (within 2%), then the best validation AUC")}
    <div style="overflow:auto"><table class="data-table"><thead><tr><th>Model</th><th>Val AUC</th><th>Val MCC</th><th>Test AUC</th><th>Test MCC</th><th>Test value (Rs cr)</th><th>Fit (s)</th></tr></thead>
    <tbody>${comp.map((r) => `<tr style="${r.model === champ ? `background:color-mix(in srgb, var(--accent) 12%, transparent)` : ""}">
      <td>${esc(r.model)}</td><td>${r.val_roc_auc.toFixed(3)}</td><td>${r.val_mcc_best.toFixed(3)}</td>
      <td>${r.test_roc_auc.toFixed(3)}</td><td>${r.test_mcc_at_global_threshold.toFixed(3)}</td>
      <td>${(r.test_value_breakeven_rule / 1e7).toFixed(2)}</td><td>${r.fit_seconds > 0 ? r.fit_seconds.toFixed(1) : "in search"}</td></tr>`).join("")}</tbody></table></div></div>
  <div style="height:18px"></div><div class="grid grid-3">
    <div class="card">${cardTitle("Insurance · test RMSE (Rs.)", "Lower is better · XGBoost vs linear baselines")}<div style="height:260px"><canvas id="insRmseChart"></canvas></div></div>
    <div class="card">${cardTitle("Top loan drivers", "Mean |SHAP|, champion model")}<div style="height:260px"><canvas id="loanFiChart"></canvas></div></div>
    <div class="card">${cardTitle("Top premium drivers", "Mean |SHAP| on log-premium")}<div style="height:260px"><canvas id="insFiChart"></canvas></div></div>
  </div>`;
  const a = meta.assumptions || {};
  html += `<div class="small" style="margin-top:14px">Money rule assumptions: cost of funds ${a.cost_of_funds_pct}%, exposure at default ${a.ead_factor} of the loan,
    recoveries estimated from historical charged-off loans. Gender and marital status are never model inputs.</div>`;
  main().innerHTML = html;
  document.getElementById("labRefresh").onclick = () => { clearApiCache("/monitor/meta"); clearApiCache("/monitor/comparison"); renderLab(); };

  const byAuc = [...comp].sort((a, b) => a.val_roc_auc - b.val_roc_auc);
  barHChart("aucChart", byAuc.map((r) => short(r.model)), byAuc.map((r) => r.val_roc_auc), short(champ), (v) => v.toFixed(3), [0.82, 0.875]);
  const byVal = [...comp].sort((a, b) => a.val_best_value - b.val_best_value);
  barHChart("valChart", byVal.map((r) => short(r.model)), byVal.map((r) => r.val_best_value / 1e7), short(champ), (v) => v.toFixed(2));
  const byIns = [...icomp].sort((a, b) => b.test_rmse - a.test_rmse);
  barHChart("insRmseChart", byIns.map((r) => r.model), byIns.map((r) => r.test_rmse), imeta.name, (v) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }));
  const loanFi = Object.entries(meta.feature_importance).slice(0, 8).reverse();
  barHChart("loanFiChart", loanFi.map(([k]) => k.replace(/_/g, " ")), loanFi.map(([, v]) => v), null, (v) => v.toFixed(2));
  const insFi = Object.entries(imeta.feature_importance).slice(0, 8).reverse();
  barHChart("insFiChart", insFi.map(([k]) => k.replace(/_/g, " ")), insFi.map(([, v]) => v), null, (v) => v.toFixed(2));
}

/* ======================================================================================= MONITORING */
async function renderMonitor(kind = "loan") {
  main().innerHTML = `<div class="seg-ctl" id="monKind" style="margin-bottom:14px">
    <button data-v="loan" class="${kind === "loan" ? "active" : ""}">Loan</button>
    <button data-v="insurance" class="${kind === "insurance" ? "active" : ""}">Insurance</button></div>
    <div id="monBody"><div class="loading">Loading monitoring…</div></div>`;
  document.querySelectorAll("#monKind button").forEach((b) => b.onclick = () => {
    document.querySelectorAll("#monKind button").forEach((x) => x.classList.remove("active")); b.classList.add("active"); renderMonBody(b.dataset.v);
  });
  await renderMonBody(kind);
}
async function renderMonBody(kind) {
  const body = document.getElementById("monBody");
  body.innerHTML = `<div class="loading">Loading…</div>`;
  const [rep, trend, reg] = await Promise.all([apiCached(`/monitor/drift/${kind}`), apiCached(`/monitor/drift-trend/${kind}`), apiCached("/monitor/models")]);
  if (rep.error || trend.error) { body.innerHTML = errorCard("No monitoring data yet. Run the drift job (python -m lia.drift)."); return; }
  const sd = rep.score_drift || {}, lr = rep.label_rate;
  let html = topbar("Monitoring", "Population Stability Index per feature and for the model score",
    [["Status", rep.overall.toUpperCase()], ["Max PSI", rep.max_psi.toFixed(2)]],
    "PSI < 0.10 stable · 0.10–0.25 watch · > 0.25 act (retrain)",
    { id: "monRefresh", icon: "&#8635;", label: "Refresh" });
  html += kpiRow([
    ["Overall", rep.overall[0].toUpperCase() + rep.overall.slice(1), rep.name],
    ["Score PSI", (sd.psi || 0).toFixed(3), sd.status || ""],
    ["Features on watch", String(rep.n_watch), `${rep.n_alert} on alert`],
    ["Default rate", lr ? `${pct(lr.reference)} → ${pct(lr.current)}` : "-", lr ? "training vs recent" : ""],
  ], 4);
  html += `<div style="height:16px"></div><div class="card">${cardTitle("Drift by month", "PSI of each month against the training reference · the dashed lines are the watch and alert levels")}
    <div style="height:340px"><canvas id="trendChart"></canvas></div></div>
  <div style="height:18px"></div><div class="grid grid-2">
    <div class="card">${cardTitle("Latest check, per feature", rep.time.slice(0, 16).replace("T", " ") + " UTC")}<div style="height:${Math.max(300, 20 * rep.features.length)}px"><canvas id="psiChart"></canvas></div></div>
    <div class="card">${cardTitle("Model registry", "Every training, promotion and skipped retrain is logged")}
      ${reg ? `<div style="max-height:420px;overflow:auto"><table class="data-table"><thead><tr><th>Time</th><th>Kind</th><th>Event</th><th>Name</th></tr></thead>
      <tbody>${[...reg.history].reverse().slice(0, 14).map((h) => `<tr><td>${esc(h.time || "")}</td><td>${esc(h.kind || "")}</td><td>${esc(h.event || "")}</td><td>${esc(h.name || "")}</td></tr>`).join("")}</tbody></table></div>` : ""}
      <div class="small" style="margin-top:6px">Scheduler: drift check every 24 h · retrain check every 30 days · a challenger replaces the champion only if it wins
      on the latest labelled months. ${statusBadge(rep.overall)}</div></div>
  </div>`;
  body.innerHTML = html;
  document.getElementById("monRefresh").onclick = () => {
    clearApiCache(`/monitor/drift/${kind}`); clearApiCache(`/monitor/drift-trend/${kind}`); clearApiCache("/monitor/models");
    renderMonBody(kind);
  };

  const months = trend.months;
  const palette = [cssVar("--warn"), cssVar("--accent2"), cssVar("--moss"), cssVar("--muted"), cssVar("--bad"), cssVar("--accent")];
  const datasets = trend.features.map((f, i) => ({ label: f.replace(/_/g, " "), data: months.map((m) => m[f]), borderColor: palette[i % palette.length], borderWidth: 1.6, pointRadius: 0, tension: .2 }));
  datasets.push({ label: "model score", data: months.map((m) => m.score), borderColor: cssVar("--text"), borderWidth: 3, pointRadius: 3, tension: .2 });
  lineChart("trendChart", months.map((m) => m.month), datasets, { height: 340 });

  const feats = [...rep.features].sort((a, b) => a.psi - b.psi);
  barHChart("psiChart", feats.map((f) => f.feature.replace(/_/g, " ")), feats.map((f) => f.psi),
    null, (v) => v.toFixed(3), [0, Math.max(0.3, Math.max(...feats.map((f) => f.psi)) * 1.15)], Math.max(300, 20 * feats.length));
  // recolor the PSI bars by status (stable/watch/alert) after Chart.js builds them
  const chart = window._charts["psiChart"];
  if (chart) { chart.data.datasets[0].backgroundColor = feats.map((f) => statusColor(f.status)); chart.update(); }
}
