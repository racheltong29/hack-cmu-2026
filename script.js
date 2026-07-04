(() => {
  const stage = document.getElementById("parallax");
  const layers = Array.from(document.querySelectorAll(".layer"));
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  let ticking = false;

  function applyParallax() {
    const scrollY = window.scrollY;
    const viewportHeight = stage.clientHeight;

    for (const layer of layers) {
      const speed = parseFloat(layer.dataset.speed);
      const startOffset = parseFloat(layer.dataset.offset || 0) * viewportHeight;
      const maxUpOffset = Math.max(0, layer.offsetHeight - viewportHeight);

      // Layers start pushed down by startOffset (revealing what's behind them),
      // then rise toward 0 as you scroll, and keep rising to reveal more of
      // their own lower content up to maxUpOffset.
      const raw = startOffset - scrollY * speed;
      const translateY = Math.max(-maxUpOffset, Math.min(startOffset, raw));
      layer.style.transform = `translate3d(0, ${translateY}px, 0)`;
    }

    ticking = false;
  }

  function onScroll() {
    if (!ticking) {
      requestAnimationFrame(applyParallax);
      ticking = true;
    }
  }

  if (!prefersReducedMotion) {
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
  }

  applyParallax();
})();
