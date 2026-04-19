import type { Metadata } from "next";
import "./global.css";

export const metadata: Metadata = {
  title: "SeaPredictor",
  description: "Satellite debris detection and ocean drift forecasting",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
