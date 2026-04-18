// Simple dark-mode toggle. Persists in localStorage under "mode".
// Applies via <html data-mode="light|dark"> so CSS variables switch.
(function () {
  var KEY = "mode";
  var root = document.documentElement;

  function apply(m) { root.setAttribute("data-mode", m); }
  function current() { return root.getAttribute("data-mode") || "light"; }

  // head.html already applied the persisted mode. Just wire the button.
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("#bts-mode-toggle");
    if (!btn) return;
    var next = current() === "dark" ? "light" : "dark";
    apply(next);
    try { localStorage.setItem(KEY, next); } catch (_) {}
  });

  // Respect OS preference if the user has never chosen.
  try {
    if (!localStorage.getItem(KEY) && window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches) {
      apply("dark");
    }
  } catch (_) {}
})();
