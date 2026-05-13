import fs from "node:fs";
import path from "node:path";

const globalStyles = fs.readFileSync(path.join(process.cwd(), "app/globals.css"), "utf8");

export const metadata = {
  title: "DIY-Assist",
  description: "Safe appliance troubleshooting"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <style dangerouslySetInnerHTML={{ __html: globalStyles }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
