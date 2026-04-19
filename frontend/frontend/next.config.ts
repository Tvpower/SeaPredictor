import type { NextConfig } from "next";

/**
 * Two run modes:
 *
 *   1. Dev split:   `npm run dev`           → :3000, talks to FastAPI on :8000
 *                   set NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
 *
 *   2. Prod static: `npm run build:static`  → emits ./out, served by FastAPI
 *                   leave NEXT_PUBLIC_API_BASE_URL unset (same-origin)
 *
 * Setting `output: "export"` only when explicitly requested keeps `next dev`
 * fast and lets us still use server-only features in dev if we ever need them.
 */
const isStatic = process.env.NEXT_OUTPUT === "export";

const nextConfig: NextConfig = {
  ...(isStatic
    ? {
        output: "export",
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),
};

export default nextConfig;
