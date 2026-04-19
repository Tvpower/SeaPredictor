export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <p className="eyebrow">SeaPredictor</p>
        <h1>Debris Drift Console</h1>
      </div>

      <nav className="side-nav">
        <button className="side-nav-active">
          <span>Global forecast</span>
          <span>LIVE</span>
        </button>
        <button>Scene cache</button>
        <button>Validation</button>
        <button>OSCAR fields</button>
      </nav>

      <div className="sidebar-sections">
        <div className="mini-panel">
          <p className="mini-label">Detector</p>
          <p>ResNet-18 CNN</p>
          <span>11-band Sentinel-2 tiles</span>
        </div>
        <div className="mini-panel">
          <p className="mini-label">Forecast</p>
          <p>OpenDrift plus OSCAR</p>
          <span>7000 Monte Carlo particles</span>
        </div>
      </div>

      <div className="status-panel">
        <p className="mini-label">MVP mode</p>
        <span>Cached detections, live physics runs.</span>
      </div>
    </aside>
  );
}
