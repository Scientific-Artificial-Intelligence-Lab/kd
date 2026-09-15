// Extend the hero background to the viewport, then fade it with scrolling.
// Text, navigation and search retain their ordinary document behavior.
(function () {
  "use strict";

  var hero = document.querySelector(".kd-hero:not(.kd-hero--compact)");
  if (!hero) return;

  var root = document.documentElement;
  var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var fadeDistance;
  var ticking = false;
  root.classList.add("kd-home");

  function frame() {
    ticking = false;
    var progress = Math.min(1, Math.max(0, window.scrollY / fadeDistance));
    root.style.setProperty("--kd-hero-opacity", String(1 - progress));
  }

  function measure() {
    root.classList.toggle("kd-home--fx", !motion.matches);
    var box = hero.getBoundingClientRect();
    // Reduced motion keeps a static background within the hero's page area.
    root.style.setProperty(
      "--kd-hero-end", box.bottom + window.scrollY + "px"
    );
    fadeDistance = box.height * 0.8;
    frame();
  }

  addEventListener("scroll", function () {
    if (!ticking && !motion.matches) {
      ticking = true;
      requestAnimationFrame(frame);
    }
  }, { passive: true });
  addEventListener("resize", measure, { passive: true });
  motion.addEventListener("change", measure);
  new ResizeObserver(measure).observe(hero);
  measure();
})();
