import type { SceneIndexEntry } from "@/lib/api";

export interface SidebarProps {
  scenes: SceneIndexEntry[];
  selectedSceneId: string | null;
  onSelectScene: (sceneId: string) => void;
  loading?: boolean;
  error?: string | null;
}

export default function Sidebar({
  scenes,
  selectedSceneId,
  onSelectScene,
  loading,
  error,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <p className="eyebrow">SeaPredictor</p>
        <h1>Debris Drift Console</h1>
      </div>

      <div className="sidebar-sections" style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
        <p className="mini-label">Cached scenes</p>

        {loading && <p style={{ color: "var(--text-muted)", fontSize: 12 }}>Loading…</p>}
        {error && <p style={{ color: "#ff806f", fontSize: 12 }}>{error}</p>}

        <nav className="side-nav">
          {scenes.map((s) => {
            const isActive = s.scene_id === selectedSceneId;
            return (
              <button
                key={s.scene_id}
                className={isActive ? "side-nav-active" : ""}
                onClick={() => onSelectScene(s.scene_id)}
                title={s.scene_id}
                style={{ textAlign: "left", flexDirection: "column", alignItems: "flex-start", gap: 2, paddingTop: 8, paddingBottom: 8, height: "auto", minHeight: 38 }}
              >
                <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                  {s.scene_id}
                </span>
                <span style={{ color: "var(--text-muted)", fontSize: 11, fontWeight: 500 }}>
                  {s.obs_date} · {s.n_detections} det
                </span>
              </button>
            );
          })}
          {!loading && !error && scenes.length === 0 && (
            <p style={{ color: "var(--text-muted)", fontSize: 12 }}>
              No cached scenes. Run <code>python -m src.pipeline.build_scenes</code>.
            </p>
          )}
        </nav>
      </div>

      <div className="mini-panel">
        <p className="mini-label">Detector</p>
        <p>ResNet-18 CNN</p>
        <span>11-band Sentinel-2 tiles</span>
      </div>
      <div className="mini-panel">
        <p className="mini-label">Forecast</p>
        <p>OpenDrift + OSCAR</p>
        <span>Monte Carlo particle drift</span>
      </div>

      <div className="status-panel">
        <p className="mini-label">MVP mode</p>
        <span>Cached detections, live physics runs.</span>
      </div>
    </aside>
  );
}
