"use client";

import { useState } from "react";

export interface ForecastParams {
  days: number;
  n_per_seed: number;
  wind_speed_ms: number;
  /** Direction the wind blows FROM, in compass degrees (0=N, 90=E). */
  wind_dir_deg: number;
  /** Fraction of wind speed transferred to particles (Stokes drift proxy). */
  wind_drift_factor: number;
}

export interface ControlPanelProps {
  disabled?: boolean;
  running?: boolean;
  initial?: Partial<ForecastParams>;
  animate: boolean;
  onAnimateChange: (next: boolean) => void;
  onRun: (params: ForecastParams) => void;
  status?: string | null;
  error?: string | null;
}

interface WindPreset {
  label: string;
  speed: number;
  dir: number;
  factor: number; // percent (0-10), divided by 100 before sending
}

const WIND_PRESETS: WindPreset[] = [
  { label: "Calm", speed: 0, dir: 0, factor: 0 },
  { label: "Trade wind", speed: 8, dir: 80, factor: 2.5 },
  { label: "Storm", speed: 18, dir: 120, factor: 3.5 },
];

const WIND_VANE_SIZE = 56;

export default function ControlPanel({
  disabled,
  running,
  initial,
  animate,
  onAnimateChange,
  onRun,
  status,
  error,
}: ControlPanelProps) {
  const [days, setDays] = useState<number>(initial?.days ?? 7);
  const [particlesPerSeed, setParticlesPerSeed] = useState<number>(
    initial?.n_per_seed ?? 100,
  );
  const [windSpeed, setWindSpeed] = useState<number>(initial?.wind_speed_ms ?? 0);
  const [windDir, setWindDir] = useState<number>(initial?.wind_dir_deg ?? 90);
  // Stored as percent in the UI (matches legacy UI); convert to fraction on submit.
  const [windFactorPct, setWindFactorPct] = useState<number>(
    initial?.wind_drift_factor != null ? initial.wind_drift_factor * 100 : 2.5,
  );

  const applyPreset = (p: WindPreset) => {
    setWindSpeed(p.speed);
    setWindDir(p.dir);
    setWindFactorPct(p.factor);
  };

  const handleRun = () => {
    if (disabled || running) return;
    onRun({
      days,
      n_per_seed: particlesPerSeed,
      wind_speed_ms: windSpeed,
      wind_dir_deg: windDir,
      wind_drift_factor: windFactorPct / 100,
    });
  };

  // Wind FROM convention: arrow points toward where the wind is going.
  const arrowDeg = (windDir + 180) % 360;
  const windLabel =
    windSpeed > 0 ? `${windSpeed.toFixed(1)} m/s from ${windDir}°` : "off";

  return (
    <section className="control-panel">
      <div className="panel-title">
        <h3>Forecast controls</h3>
        <span>{running ? "running…" : "ready"}</span>
      </div>
      <div className="field-stack">
        <div className="field-row">
          <label>
            Days
            <input
              type="number"
              min={1}
              max={30}
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
            />
          </label>
          <label>
            Particles / seed
            <input
              type="number"
              min={10}
              max={500}
              step={10}
              value={particlesPerSeed}
              onChange={(e) => setParticlesPerSeed(Number(e.target.value))}
            />
          </label>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 4,
          }}
        >
          <p className="mini-label" style={{ margin: 0 }}>
            Wind forcing
          </p>
          <span style={{ color: "var(--text-muted)", fontSize: 11 }}>{windLabel}</span>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "stretch" }}>
          <div style={{ flex: 1, display: "grid", gap: 8 }}>
            <div className="field-row">
              <label>
                Speed (m/s)
                <input
                  type="number"
                  min={0}
                  max={40}
                  step={0.5}
                  value={windSpeed}
                  onChange={(e) => setWindSpeed(Number(e.target.value))}
                />
              </label>
              <label>
                From (deg)
                <input
                  type="number"
                  min={0}
                  max={359}
                  step={5}
                  value={windDir}
                  onChange={(e) =>
                    setWindDir(((Number(e.target.value) % 360) + 360) % 360)
                  }
                />
              </label>
            </div>
            <label>
              Drift factor (% of wind)
              <input
                type="number"
                min={0}
                max={10}
                step={0.1}
                value={windFactorPct}
                onChange={(e) => setWindFactorPct(Number(e.target.value))}
              />
            </label>
          </div>

          <div
            aria-hidden
            style={{
              width: WIND_VANE_SIZE,
              height: WIND_VANE_SIZE,
              borderRadius: "50%",
              border: "1px solid var(--line-soft)",
              background: "rgba(83, 234, 253, 0.06)",
              alignSelf: "center",
              position: "relative",
              flexShrink: 0,
            }}
          >
            <svg
              viewBox="-50 -50 100 100"
              width={WIND_VANE_SIZE}
              height={WIND_VANE_SIZE}
              style={{ position: "absolute", inset: 0 }}
            >
              <circle r={48} fill="none" stroke="rgba(180,230,255,0.18)" />
              <text x={0} y={-36} textAnchor="middle" fontSize={10} fill="#94a8ba">
                N
              </text>
              <g transform={`rotate(${arrowDeg})`}>
                <line
                  x1={0}
                  y1={28}
                  x2={0}
                  y2={-28}
                  stroke={windSpeed > 0 ? "var(--cyan)" : "rgba(148,168,186,0.5)"}
                  strokeWidth={2.5}
                  strokeLinecap="round"
                />
                <polygon
                  points="-6,-22 6,-22 0,-34"
                  fill={windSpeed > 0 ? "var(--cyan)" : "rgba(148,168,186,0.5)"}
                />
              </g>
            </svg>
          </div>
        </div>

        <div style={{ display: "flex", gap: 6 }}>
          {WIND_PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => applyPreset(p)}
              style={{
                flex: 1,
                background: "rgba(83, 234, 253, 0.08)",
                color: "var(--cyan)",
                border: "1px solid rgba(83, 234, 253, 0.25)",
                borderRadius: 6,
                padding: "6px 8px",
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        <label
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            background: "rgba(255,255,255,0.04)",
            borderRadius: 6,
            padding: "8px 10px",
            cursor: "pointer",
            userSelect: "none",
          }}
        >
          <span>
            <span style={{ display: "block", fontSize: 12, fontWeight: 600 }}>
              Animate drift paths
            </span>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              Flow dashes along OpenDrift trajectories
            </span>
          </span>
          <input
            type="checkbox"
            checked={animate}
            onChange={(e) => onAnimateChange(e.target.checked)}
          />
        </label>

        <button
          className="primary-action"
          onClick={handleRun}
          disabled={disabled || running}
        >
          {running ? "Forecasting…" : "Run forecast"}
        </button>
        {status && (
          <p style={{ color: "var(--text-muted)", fontSize: 12, margin: 0 }}>
            {status}
          </p>
        )}
        {error && (
          <p style={{ color: "#ff806f", fontSize: 12, margin: 0 }}>{error}</p>
        )}
      </div>
    </section>
  );
}
