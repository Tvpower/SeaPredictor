export default function Card() {
  return (
    <section className="analysis-card">
      <div className="card-header">
        <div>
          <p className="mini-label">Current scene</p>
          <h3>S2 Honduras</h3>
        </div>
        <span className="scene-pill">35 seeds</span>
      </div>

      <div className="metric-grid">
        <div>
          <span>Particles</span>
          <strong>7000</strong>
        </div>
        <div>
          <span>Window</span>
          <strong>7d</strong>
        </div>
        <div>
          <span>Hit rate</span>
          <strong className="success">72%</strong>
        </div>
      </div>

      <div className="current-track">
        <div />
      </div>
      <p>
        CNN detections seed OpenDrift particles. Cyan tracks show forecast spread
        from OSCAR surface currents.
      </p>
    </section>
  );
}
