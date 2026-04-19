import type { ForecastResponse, SceneIndexEntry } from "@/lib/api";

export interface CardProps {
  scene: SceneIndexEntry | null;
  forecast: ForecastResponse | null;
}

export default function Card({ scene, forecast }: CardProps) {
  if (!scene) {
    return (
      <section className="analysis-card">
        <div className="card-header">
          <div>
            <p className="mini-label">Current scene</p>
            <h3>None selected</h3>
          </div>
        </div>
        <p>Pick a cached scene from the sidebar to inspect its CNN detections.</p>
      </section>
    );
  }

  const particles = forecast?.stats.n_particles ?? 0;
  const days = forecast?.params.days ?? 0;

  return (
    <section className="analysis-card">
      <div className="card-header">
        <div>
          <p className="mini-label">Current scene</p>
          <h3>{scene.scene_id}</h3>
        </div>
        <span className="scene-pill">{scene.n_detections} seeds</span>
      </div>

      <div className="metric-grid">
        <div>
          <span>Particles</span>
          <strong>{particles ? particles.toLocaleString() : "—"}</strong>
        </div>
        <div>
          <span>Window</span>
          <strong>{days ? `${days}d` : "—"}</strong>
        </div>
        <div>
          <span>Tiles scanned</span>
          <strong className="success">{scene.n_tiles}</strong>
        </div>
      </div>

      <div className="current-track">
        <div />
      </div>
      <p>
        {scene.has_oscar_coverage
          ? "OSCAR currents available — forecast will use real surface drift."
          : "OSCAR not available for this date — forecast falls back to diffusion only."}
      </p>
    </section>
  );
}
