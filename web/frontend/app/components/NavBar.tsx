"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function NavBar() {
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  return (
    <nav className="border-b border-neutral-800 bg-neutral-900/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-8">
            <Link href="/" className="text-xl font-bold text-white hover:text-neutral-300 transition">
              NavSim Showcase
            </Link>
            <div className="hidden md:flex space-x-6">
              <Link
                href="/"
                className={`px-3 py-2 rounded-md text-sm font-medium transition ${
                  isActive("/")
                    ? "text-white bg-neutral-800"
                    : "text-neutral-400 hover:text-white hover:bg-neutral-800/50"
                }`}
              >
                Demo
              </Link>
              <Link
                href="/about"
                className={`px-3 py-2 rounded-md text-sm font-medium transition ${
                  isActive("/about")
                    ? "text-white bg-neutral-800"
                    : "text-neutral-400 hover:text-white hover:bg-neutral-800/50"
                }`}
              >
                About
              </Link>
            </div>
          </div>
          <div className="text-xs text-neutral-500">
            Master&apos;s Thesis | NYU Tandon 2025
          </div>
        </div>
      </div>
    </nav>
  );
}


