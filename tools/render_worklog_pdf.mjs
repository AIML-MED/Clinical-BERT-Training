import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const { marked } = require("marked");


const inputPath = path.resolve(process.argv[2]);
const outputPath = path.resolve(process.argv[3]);
const markdown = fs.readFileSync(inputPath, "utf8");
const content = marked.parse(markdown, { gfm: true });

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>OSCAR Continued Pretraining 工作记录</title>
<style>
  @page { size: A4; margin: 17mm 16mm 18mm; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: #18212b;
    font-family: "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", Arial, sans-serif;
    font-size: 10.2pt;
    line-height: 1.55;
  }
  h1, h2, h3 { color: #17365d; break-after: avoid; }
  h1 { font-size: 22pt; margin: 0 0 8mm; padding-bottom: 3mm; border-bottom: 2px solid #2d6a8e; }
  h2 { font-size: 15pt; margin: 7mm 0 2.5mm; padding-bottom: 1.2mm; border-bottom: 1px solid #c9d9e6; }
  h3 { font-size: 12pt; margin: 5mm 0 2mm; }
  p { margin: 0 0 2.6mm; orphans: 3; widows: 3; }
  ul, ol { margin: 1.5mm 0 3mm; padding-left: 7mm; }
  li { margin: 0.8mm 0; }
  code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; }
  p code, li code { background: #eef3f7; padding: 0.2mm 1mm; border-radius: 2px; }
  pre {
    margin: 2.5mm 0 4mm;
    padding: 3mm 3.5mm;
    background: #f4f7f9;
    border-left: 3px solid #4f86a6;
    border-radius: 3px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    break-inside: avoid;
  }
  pre code { font-size: 8.4pt; line-height: 1.4; }
  table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm; font-size: 9.2pt; }
  th { background: #dce9f2; color: #17365d; font-weight: 700; }
  th, td { border: 1px solid #aebfcb; padding: 1.7mm 2mm; text-align: left; vertical-align: top; }
  tr { break-inside: avoid; }
  blockquote { margin: 3mm 0; padding: 2mm 4mm; background: #f7f4e8; border-left: 3px solid #b69a42; }
  a { color: #1d648c; text-decoration: none; }
  hr { border: 0; border-top: 1px solid #b7c7d2; margin: 6mm 0; }
</style>
</head>
<body>${content}</body>
</html>`;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.setContent(html, { waitUntil: "load" });
await page.emulateMedia({ media: "print" });
await page.pdf({
  path: outputPath,
  format: "A4",
  printBackground: true,
  displayHeaderFooter: true,
  headerTemplate: "<span></span>",
  footerTemplate: `<div style="width:100%;font:8px Arial;color:#647482;text-align:center;">
    <span class="pageNumber"></span> / <span class="totalPages"></span>
  </div>`,
  margin: { top: "17mm", right: "16mm", bottom: "18mm", left: "16mm" },
});
await browser.close();

console.log(outputPath);
