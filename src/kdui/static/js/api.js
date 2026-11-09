// The only module that knows URLs: the kdui HTTP API under api/v1, the token
// exchange, JSON errors, pollers, and the small client-side store the pages
// share (current dataset, preprocessing choice). See README.md for the list.

/** Job states a job never leaves (kdui.models.TERMINAL_STATES). */
export const TERMINAL_STATES = new Set([
  "completed",
  "budget_stopped",
  "cancelled",
  "failed",
  "interrupted",
]);

/** Window event sent after a request that queued, stopped or killed a job. */
export const JOBS_CHANGED = "kd:jobs-changed";

/** Window event the shell sends each time it has read GET /system; detail: SystemInfo. */
export const SYSTEM_READ = "kd:system-read";

const API_BASE = new URL("api/v1/", document.baseURI);

/** An error answer of the API (an ErrorResponse) or a failed request. */
export class ApiError extends Error {
  constructor(status, code, message, field = null) {
    super(message);
    this.name = "ApiError";
    this.status = status; // HTTP status, 0 when the server could not be reached
    this.code = code; // ErrorCode of kdui.models, or "network"
    this.field = field; // dotted request field the message is about
  }
}

/** Absolute URL of an API path such as "/jobs/j-1/events", with query items. */
export function apiUrl(path, query = null) {
  const url = new URL(path.replace(/^\//, ""), API_BASE);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === null || value === undefined) continue;
      url.searchParams.set(key, Array.isArray(value) ? value.join(",") : String(value));
    }
  }
  return url.href;
}

/** URL of an artifact file (report, figure, record), for links, img and iframe. */
export function artifactUrl(artifactId) {
  return apiUrl(`/artifacts/${encodeURIComponent(artifactId)}`);
}

const JOB_POSTS = /^\/(jobs($|\/[^/]+\/(stop|kill)$)|runs\/[^/]+\/(resume|clone)$)/;

// The request key of POST /session (SessionOpened.request_key). Every request
// but GET carries it in this header; the cookie alone only reads. It is kept in
// this origin's storage, which a server on another port cannot read, while the
// cookie is sent to every port of the host.
const KEY_HEADER = "X-KD-Key"; // kdui.models.REQUEST_KEY_HEADER
const KEY_STORE = "kdui.request_key";

/**
 * One API request. Resolves with the decoded JSON body (or the text of a
 * non-JSON answer, or null for 204); rejects with ApiError.
 * opts: { query, body (JSON-encoded), raw (Blob/File sent as is), signal }
 */
export async function request(method, path, opts = {}) {
  const init = { method, credentials: "same-origin", signal: opts.signal, headers: {} };
  if (opts.raw !== undefined) {
    init.body = opts.raw;
    init.headers["Content-Type"] = "application/octet-stream";
  } else if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    init.headers["Content-Type"] = "application/json";
  }
  if (method !== "GET") init.headers[KEY_HEADER] = window.localStorage.getItem(KEY_STORE) || "";
  let response;
  try {
    response = await fetch(apiUrl(path, opts.query), init);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(0, "network", "The KD server could not be reached. Is kd-ui still running?");
  }
  const type = response.headers.get("Content-Type") || "";
  const body =
    response.status === 204 ? null : type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    if (body && typeof body === "object" && body.message) {
      // 504 kd_unavailable: the short kd process timed out or died. Nothing was
      // refused or changed, so the same request may simply be repeated.
      const message = body.code === "kd_unavailable" ? `KD did not answer in time; nothing was refused or changed. Try again. (${body.message})` : body.message;
      throw new ApiError(response.status, body.code, message, body.field ?? null);
    }
    throw new ApiError(response.status, "http", `${method} ${path} answered ${response.status} ${response.statusText}`);
  }
  if (method !== "GET" && JOB_POSTS.test(path)) {
    window.dispatchEvent(new CustomEvent(JOBS_CHANGED, { detail: { path } }));
  }
  return body;
}

export const get = (path, query = null, opts = {}) => request("GET", path, { ...opts, query });
export const post = (path, body = {}, opts = {}) => request("POST", path, { ...opts, body });

/** Send a file as the raw request body, e.g. upload("/uploads", file, {filename: file.name}). */
export const upload = (path, blob, query = null) => request("POST", path, { raw: blob, query });

/**
 * Open the session. A URL ending in #token=<t> (printed by kd-ui) is exchanged
 * for the session cookie and the request key, and the token is removed from
 * the address bar; any other URL checks the existing cookie and needs the key
 * kept from an earlier exchange. Resolves with SessionInfo.
 */
export async function openSession() {
  const match = /^#token=([^&]+)/.exec(window.location.hash);
  if (!match) {
    if (!window.localStorage.getItem(KEY_STORE)) throw new ApiError(401, "unauthorized", "This browser holds no key of this KD server.");
    return get("/session");
  }
  const token = decodeURIComponent(match[1]);
  const { pathname, search } = window.location;
  window.history.replaceState(null, "", `${pathname}${search}#/data`);
  const opened = await post("/session", { token });
  window.localStorage.setItem(KEY_STORE, opened.request_key);
  return opened;
}

// ---------------------------------------------------------------------------
// Polling: one outstanding request, slower while the tab is hidden.

const hidden = () => document.visibilityState === "hidden";

function sleeper() {
  // A sleep that ends early when the tab becomes visible or wake() is called.
  let wake = () => {};
  const onVisible = () => {
    if (!hidden()) wake();
  };
  document.addEventListener("visibilitychange", onVisible);
  return {
    sleep(ms) {
      return new Promise((resolve) => {
        const timer = setTimeout(resolve, ms);
        wake = () => {
          clearTimeout(timer);
          resolve();
        };
      });
    },
    wake: () => wake(),
    close: () => document.removeEventListener("visibilitychange", onVisible),
  };
}

/**
 * Call `fn` repeatedly, waiting `interval` ms between calls (`hiddenInterval`
 * while the tab is hidden). The next call starts only after the previous one
 * settled. `fn` returning true stops the loop. Errors go to `onError` (default:
 * console) and the loop goes on at the hidden pace; a 401 stops it.
 * `fn(live)` gets a function that is false once stop() was called: an answer
 * that arrives after stop() belongs to a page that is gone and is dropped.
 * Returns { stop(), now() }; now() runs the next call at once.
 */
export function pollEvery(fn, { interval = 2000, hiddenInterval = 10000, onError = null } = {}) {
  let stopped = false;
  const live = () => !stopped;
  const clock = sleeper();
  (async () => {
    while (!stopped) {
      let pause = hidden() ? hiddenInterval : interval;
      try {
        if ((await fn(live)) === true) break;
      } catch (err) {
        if (stopped) break;
        if (onError) onError(err);
        else console.error(err);
        if (err instanceof ApiError && err.status === 401) break;
        pause = hiddenInterval;
      }
      if (stopped) break;
      await clock.sleep(pause);
    }
    clock.close();
  })();
  return {
    stop() {
      stopped = true;
      clock.wake();
    },
    now: () => clock.wake(),
  };
}

/**
 * Follow a cursor journal (EventsPage or LogPage): GET path?cursor=..., call
 * onPage(page) for each page, ask again at once while page.has_more, and stop
 * when isDone(page) returns true (it may be async). The caller handles
 * page.reset (drop what it holds). Returns { stop(), now() }.
 */
export function pollCursor(path, { onPage, isDone = () => false, cursor = null, interval = 1000, hiddenInterval = 5000, onError = null }) {
  let next = cursor;
  return pollEvery(
    async (live) => {
      for (;;) {
        const page = await get(path, next === null ? null : { cursor: next });
        if (!live()) return true;
        next = page.next_cursor;
        onPage(page);
        if (!page.has_more) return Boolean(await isDone(page));
      }
    },
    { interval, hiddenInterval, onError },
  );
}

/**
 * Follow one job: its JobDetail, its progress events and (optionally) its
 * log, in one sequential loop. Each round reads the job, then drains events
 * and log; after the round in which the job is in a terminal state it stops.
 * handlers: { onJob(job), onEvents(page), onLog(page) } (each optional).
 * Returns { stop(), now() }.
 */
export function followJob(jobId, { onJob, onEvents, onLog, interval = 1000, hiddenInterval = 5000, onError = null }) {
  const id = encodeURIComponent(jobId);
  const cursors = { events: null, logs: null };
  const drain = async (kind, handler, live) => {
    for (;;) {
      const page = await get(`/jobs/${id}/${kind}`, cursors[kind] === null ? null : { cursor: cursors[kind] });
      if (!live()) return;
      cursors[kind] = page.next_cursor;
      handler(page);
      if (!page.has_more) return;
    }
  };
  return pollEvery(
    async (live) => {
      const job = await get(`/jobs/${id}`);
      if (!live()) return true;
      if (onJob) onJob(job);
      if (onEvents) await drain("events", onEvents, live);
      if (onLog) await drain("logs", onLog, live);
      return TERMINAL_STATES.has(job.state);
    },
    { interval, hiddenInterval, onError },
  );
}

// ---------------------------------------------------------------------------
// Store: small client-side state the pages share, kept in localStorage.
// Known keys (see README.md): "dataset", "preprocessing", "run".

const PREFIX = "kdui.";
const listeners = new Map();

function emit(key, value) {
  for (const fn of listeners.get(key) || []) fn(value);
}

export const store = {
  /** The stored value of `key`, or `fallback` when there is none. */
  get(key, fallback = null) {
    const raw = window.localStorage.getItem(PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  },
  /** Store a JSON value; null removes the key. Listeners of the key are called. */
  set(key, value) {
    if (value === null || value === undefined) window.localStorage.removeItem(PREFIX + key);
    else window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
    emit(key, value ?? null);
  },
  /** Call fn(value) whenever `key` changes, in this tab or another. Returns unsubscribe. */
  on(key, fn) {
    if (!listeners.has(key)) listeners.set(key, new Set());
    listeners.get(key).add(fn);
    return () => listeners.get(key).delete(fn);
  },
};

window.addEventListener("storage", (event) => {
  if (event.key && event.key.startsWith(PREFIX)) {
    const key = event.key.slice(PREFIX.length);
    emit(key, event.newValue === null ? null : JSON.parse(event.newValue));
  }
});
