import type { SceneIndexEntry } from "@/lib/api";

export interface TopbarProps {
  scene: SceneIndexEntry | null;
}

export default function Topbar({ scene }: TopbarProps) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Stage 3 Web UI</p>
        <h2>
          {scene
            ? `${scene.scene_id} · ${scene.obs_date}`
            : "Select a cached scene"}
        </h2>
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
