import Globe from "@/components/Globe/Globe";
import Sidebar from "@/components/Layout/Sidebar";
import Topbar from "@/components/Layout/Topbar";
import Card from "@/components/Layout/Card";

export default function Page() {
  return (
    <main className="app-shell">
      <Sidebar />

      <section className="globe-stage">
        <Topbar />

        <div className="globe-layer">
          <Globe />
        </div>

        <div className="stage-vignette" />

        <div className="scene-card-slot">
          <Card />
        </div>

        <div className="pipeline-strip">
          {[
            ["Stage 1", "Detector", "35 debris tiles from 79 MARIDA patches"],
            ["Stage 2", "Forecast", "OpenDrift evolves 7000 particles"],
            ["Stage 3", "Explore", "FastAPI serves cached GeoJSON layers"],
          ].map(([stage, title, body]) => (
            <article className="pipeline-card" key={stage}>
              <p className="mini-label">{stage}</p>
              <h3>{title}</h3>
              <p>{body}</p>
            </article>
          ))}
        </div>

        <aside className="right-stack">
          <section className="control-panel">
            <div className="panel-title">
              <h3>Forecast controls</h3>
              <span>cached</span>
            </div>
            <div className="field-stack">
              <label>
                Scene
                <select>
                  <option>honduras_sep18</option>
                  <option>demo_caribbean_window</option>
                </select>
              </label>
              <div className="field-row">
                <label>
                  Days
                  <input
                    defaultValue="7"
                  />
                </label>
                <label>
                  Particles
                  <input
                    defaultValue="200"
                  />
                </label>
              </div>
              <button className="primary-action">Run forecast</button>
            </div>
          </section>

          <section className="legend-panel">
            <h3>Layer legend</h3>
            <div className="legend-list">
              <div>
                <span className="legend-dot seed" />
                Detector seed tiles
              </div>
              <div>
                <span className="legend-line" />
                OpenDrift particle paths
              </div>
              <div>
                <span className="legend-dot final" />
                Final particle positions
              </div>
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}
