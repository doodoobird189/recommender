// Single-page UI. All DOM is built with textContent (never innerHTML) so movie data can't inject markup.
"use strict";

const MIN_PICKS = 3;
const state = {
  selected: new Map(), // movie_id -> movie
  sampleRatings: null, // {movieId: rating} when "borrowing" a MovieLens user
  lastRequest: 0,
};

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 4000);
}

const label = (m) => (m.year ? `${m.title} (${m.year})` : m.title);

function poster(movie, extraClass = "") {
  const box = el("div", { class: `poster ${extraClass}` });
  if (movie.poster_url) {
    box.append(el("img", { src: movie.poster_url, alt: label(movie), loading: "lazy" }));
  } else {
    box.append(el("div", { class: "fallback", text: label(movie) }));
  }
  return box;
}

// ── selection ───────────────────────────────────────────────────────────────

function toggle(movie) {
  clearSample();
  if (state.selected.has(movie.movie_id)) state.selected.delete(movie.movie_id);
  else state.selected.set(movie.movie_id, movie);
  renderSelection();
}

function clearSample() {
  if (!state.sampleRatings) return;
  state.sampleRatings = null;
  state.selected.clear();
  $("sample-note").hidden = true;
}

function renderSelection() {
  const chips = [...state.selected.values()].map((m) =>
    el("span", { class: "chip" }, label(m),
      el("button", { type: "button", "aria-label": `Remove ${m.title}`, text: "×", onclick: () => toggle(m) }))
  );
  $("selected").replaceChildren(...chips);
  for (const btn of document.querySelectorAll("#popular .poster")) {
    btn.classList.toggle("selected", state.selected.has(Number(btn.dataset.id)));
  }
  const n = state.sampleRatings ? Object.keys(state.sampleRatings).length : state.selected.size;
  const go = $("go");
  go.disabled = n < MIN_PICKS;
  go.textContent = n < MIN_PICKS ? `Pick at least ${MIN_PICKS} movies (${n}/${MIN_PICKS})` : `Recommend from ${n} movies →`;
}

// ── popular grid ────────────────────────────────────────────────────────────

async function loadPopular(seed) {
  const grid = $("popular");
  grid.replaceChildren(...Array.from({ length: 18 }, () => el("div", { class: "poster skeleton" })));
  try {
    const { movies } = await api(`/api/popular?n=18${seed != null ? `&seed=${seed}` : ""}`);
    grid.replaceChildren(
      ...movies.map((m) => {
        const btn = el("button", { type: "button", class: "poster", "data-id": m.movie_id, title: label(m), onclick: () => toggle(m) });
        btn.replaceChildren(...poster(m).childNodes);
        return btn;
      })
    );
    renderSelection();
  } catch (e) {
    toast(e.message);
  }
}

// ── search ──────────────────────────────────────────────────────────────────

let searchTimer;
let activeIndex = -1;

function setupSearch() {
  const input = $("search");
  const list = $("search-results");

  input.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => runSearch(input.value), 180);
  });
  input.addEventListener("keydown", (e) => {
    const items = [...list.children];
    if (list.hidden || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      activeIndex = (activeIndex + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.classList.toggle("active", i === activeIndex));
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      items[activeIndex].click();
    } else if (e.key === "Escape") {
      closeSearch();
    }
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search")) closeSearch();
  });
}

let searchSeq = 0;

function closeSearch() {
  const list = $("search-results");
  list.hidden = true;
  list.replaceChildren();
}

async function runSearch(q) {
  const seq = ++searchSeq;
  if (q.trim().length < 2) return closeSearch();
  const list = $("search-results");
  try {
    const { movies } = await api(`/api/search?q=${encodeURIComponent(q)}`);
    // ignore out-of-order responses and results for text the user has since cleared
    if (seq !== searchSeq || $("search").value.trim().length < 2) return;
    activeIndex = -1;
    list.replaceChildren(
      ...(movies.length
        ? movies.map((m) =>
            el("li", { role: "option", onclick: () => { toggle(m); $("search").value = ""; ++searchSeq; closeSearch(); } },
              m.poster_url ? el("img", { src: m.poster_url, alt: "" }) : el("span", { class: "thumb" }),
              el("span", {}, el("div", { text: label(m) }),
                el("div", { class: "muted small", text: `${m.genres.slice(0, 3).join(" · ")}  ·  ${m.n_ratings.toLocaleString()} ratings` })))
          )
        : [el("li", { class: "muted", text: "No matches" })])
    );
    list.hidden = false;
  } catch (e) {
    toast(e.message);
  }
}

// ── sample MovieLens user ───────────────────────────────────────────────────

async function borrowUser() {
  try {
    const u = await api("/api/sample-user");
    state.selected.clear();
    u.top_rated.forEach((m) => state.selected.set(m.movie_id, m));
    state.sampleRatings = u.ratings;
    const note = $("sample-note");
    note.textContent = `Using all ${u.n_ratings} real ratings from MovieLens user #${u.user_idx} (their favourites shown above, including low ratings, which steer the content model away). Click any movie to start your own list instead.`;
    note.hidden = false;
    renderSelection();
    recommend();
  } catch (e) {
    toast(e.message);
  }
}

// ── recommendations ─────────────────────────────────────────────────────────

function blendText(w) {
  const pct = Math.round(w * 100);
  return `${pct}% collaborative filtering · ${100 - pct}% content similarity`;
}

async function recommend() {
  const cf_weight = Number($("blend").value);
  const body = state.sampleRatings
    ? { ratings: state.sampleRatings, cf_weight, n: 12 }
    : { liked: [...state.selected.keys()], cf_weight, n: 12 };
  const requestId = ++state.lastRequest;
  const results = $("results");
  results.hidden = false;
  $("recs").replaceChildren(...Array.from({ length: 8 }, () => el("div", { class: "card" }, el("div", { class: "poster skeleton" }))));
  try {
    const data = await api("/api/recommend", { method: "POST", body });
    if (requestId !== state.lastRequest) return; // a newer request superseded this one
    renderResults(data);
  } catch (e) {
    toast(e.message);
  }
}

function renderResults(data) {
  const taste = data.taste_profile.length
    ? [el("span", { class: "muted small", text: "Your taste:" }), ...data.taste_profile.map((t) => el("span", { class: "tag", text: t }))]
    : [];
  $("taste").replaceChildren(...taste);

  $("recs").replaceChildren(
    ...data.recommendations.map((m) => {
      const p = poster(m);
      p.append(el("span", { class: "rank", text: `#${m.rank}` }));
      const because = m.because_you_liked.length
        ? el("div", { class: "because" }, "Because you liked ",
            ...m.because_you_liked.flatMap((b, i) => [i ? " & " : "", el("strong", { text: b.title })]))
        : null;
      return el("article", { class: "card" }, p,
        el("div", { class: "card-body" },
          el("div", { class: "card-title", text: m.title }),
          el("div", { class: "card-meta", text: [m.year, m.genres.slice(0, 3).join(" · ")].filter(Boolean).join("  ·  ") }),
          because,
          m.shared_tags.length ? el("div", { class: "tags" }, ...m.shared_tags.map((t) => el("span", { class: "tag", text: t }))) : null,
          m.overview ? el("div", { class: "overview", text: m.overview, title: m.overview }) : null));
    })
  );
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── metrics ─────────────────────────────────────────────────────────────────

const METRIC_COLS = [
  ["precision@10", "P@10"], ["recall@10", "R@10"], ["ndcg@10", "NDCG@10"],
  ["cold_recall@20", "Cold R@20"], ["coverage@10", "Coverage"], ["novelty@10", "Novelty"],
];

async function loadMetrics() {
  try {
    const { metrics } = await api("/api/model");
    const split = metrics?.splits?.temporal;
    if (!split) return;
    const served = `Hybrid (cf=${metrics.selected_cf_weight})`;
    const fmt = (k, v) => (v == null ? "–" : k === "coverage@10" ? `${(v * 100).toFixed(1)}%` : v.toFixed(3));
    const rows = Object.entries(split.models).map(([name, m]) =>
      el("tr", { class: name === served ? "highlight" : null },
        el("td", { text: name }), ...METRIC_COLS.map(([k]) => el("td", { text: fmt(k, m[k]) })))
    );
    $("metrics").replaceChildren(
      el("h3", { text: "Offline evaluation" }),
      el("p", { class: "muted small", text: `Temporal split: trained on ratings before ${split.cutoff}, evaluated on ${split.n_eval_users.toLocaleString()} users' future ratings. Highlighted: the blend this app serves, tuned on an earlier validation window.` }),
      el("div", { class: "table-wrap" },
        el("table", { class: "metrics-table" },
          el("thead", {}, el("tr", {}, el("th", { text: "Model" }), ...METRIC_COLS.map(([, l]) => el("th", { text: l })))),
          el("tbody", {}, ...rows)))
    );
  } catch {
    /* metrics are optional */
  }
}

// ── init ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  setupSearch();
  loadPopular();
  loadMetrics();
  const blend = $("blend");
  $("blend-readout").textContent = blendText(Number(blend.value));
  blend.addEventListener("input", () => ($("blend-readout").textContent = blendText(Number(blend.value))));
  blend.addEventListener("change", () => { if (!$("results").hidden) recommend(); });
  $("go").addEventListener("click", recommend);
  $("shuffle").addEventListener("click", () => loadPopular(Math.floor(Math.random() * 1e6)));
  $("sample-user")?.addEventListener("click", borrowUser);
});
