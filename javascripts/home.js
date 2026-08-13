// The homepage opens on the run alone: no header, no sidebar, nothing to read
// past. Scrolling dissolves that first screen in place while the page under it
// comes up, and brings the navigation in with it.
//
// Every piece of this is opt-in from here. The stylesheet's default is the
// static page -- everything visible, nothing transformed -- and the classes
// this file adds are what switch the choreography on, so a blocked or failed
// script leaves a plain page rather than a blank one.
(function () {
  "use strict";

  var hero = document.querySelector(".kd-hero");
  var flow = document.querySelector(".kd-flow");
  if (!hero || !flow) return;

  var root = document.documentElement;
  var inner = hero.querySelector(".kd-hero__inner");
  var cue = hero.querySelector(".kd-hero__cue");
  var column = document.querySelector(".md-content__inner");
  var ticking = false;
  root.classList.add("kd-home");

  // How far the content column sits from the left edge of the window -- the
  // sidebar's width at desktop sizes, nothing at mobile ones. The hero and the
  // block below it both reach back across it to span the full window, and the
  // usual `50% - 50vw` trick cannot: it is measured from the middle of the
  // column, which is off-centre exactly because of this gap.
  function measure() {
    if (!column) return;
    var box = column.getBoundingClientRect();
    root.style.setProperty("--kd-bleed", box.left + "px");
    root.style.setProperty(
      "--kd-bleed-right",
      Math.max(0, document.documentElement.clientWidth - box.right) + "px"
    );
  }

  // The layout is measured either way. A reader who asked for less motion still
  // gets the page; what they do not get is the scrubbing, so the hero scrolls
  // away like any other block and the navigation is there from the start.
  measure();
  addEventListener("resize", measure, { passive: true });
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  root.classList.add("kd-home--fx");

  function frame() {
    ticking = false;
    var height = window.innerHeight;

    // p: how far the first screen has dissolved. Half a viewport, so the
    // handover is over well before the quick start reaches reading position.
    var p = Math.min(1, Math.max(0, window.scrollY / (height * 0.5)));
    inner.style.opacity = String(1 - p);
    inner.style.transform =
      "translateY(" + -p * 54 + "px) scale(" + (1 - p * 0.035) + ")";
    if (cue) cue.style.opacity = String(Math.max(0, 1 - p * 3));
    root.style.setProperty("--kd-p", p.toFixed(3));
    root.classList.toggle("kd-home--past", p > 0.02);

    // q: how far the page below has arrived. Measured from the top of the flow
    // rather than from the scroll position, so it is the incoming block's own
    // progress across the viewport that fades it in -- it comes up translucent
    // and the hero is visible dissolving underneath it.
    var top = flow.getBoundingClientRect().top;
    var q = Math.min(1, Math.max(0, (height - top) / (height * 0.34)));
    root.style.setProperty("--kd-q", (q * q).toFixed(3));
  }

  addEventListener(
    "scroll",
    function () {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(frame);
      }
    },
    { passive: true }
  );
  addEventListener("resize", frame, { passive: true });
  frame();
})();
