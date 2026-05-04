import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Scout Agent: Gayrimenkul Asistanı",
  description: "Mamdani bulanık mantık ile emlak ilan tarayıcı",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr">
      <body className="bg-gray-100 min-h-screen">{children}</body>
    </html>
  );
}
