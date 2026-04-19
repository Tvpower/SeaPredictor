export default function Topbar() {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Stage 3 Web UI</p>
        <h2>Honduras Sep 18 drift forecast</h2>
      </div>

      <div className="topbar-actions">
        <input
          aria-label="Search scene or coordinates"
          placeholder="Scene or coordinates"
        />
        <button>Export</button>
        <button>Share</button>
      </div>
    </header>
  );
}
