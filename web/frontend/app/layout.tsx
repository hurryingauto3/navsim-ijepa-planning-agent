import "./globals.css";
import type { Metadata } from "next";
import NavBar from "./components/NavBar";

export const metadata: Metadata = {
  title: "NavSim Showcase",
  description: "Interactive browser showcase for NavSim agents",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 text-neutral-100">
        <NavBar />
        {children}
      </body>
    </html>
  );
}
