(() => {
  const layers = Array.from(document.querySelectorAll(".layer"));
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  let ticking = false;

  function applyParallax() {
    const scrollY = window.scrollY;
    const viewportHeight = window.innerHeight;

    for (const layer of layers) {
      const speed = parseFloat(layer.dataset.speed);
      const maxOffset = (layer.offsetHeight - viewportHeight) / 2;
      const offset = Math.min(scrollY * speed, maxOffset);
      layer.style.transform = `translate3d(0, ${-offset}px, 0)`;
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
