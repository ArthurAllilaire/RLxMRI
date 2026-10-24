// Builds the Word version of submission_v2.md for co-authors to edit.
// Needs the npm package `docx` (v9), e.g. `NODE_PATH=<dir>/node_modules node make_submission_docx.js`.
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, Document, HeadingLevel, ImageRun, LevelFormat, Packer, Paragraph, TextRun,
} = require("docx");

const VANCOUVER = path.resolve(__dirname, "..");
const SOURCE = path.join(VANCOUVER, "submission_v2.md");
const OUT = path.join(VANCOUVER, "ISMRM2027_abstract_draft.docx");
const IMAGE_WIDTH_PX = 600; // A4 text width with 1-inch margins, at 96 dpi
// Leaves room for the heading and caption, so a tall figure stays on one page with its caption.
const IMAGE_MAX_HEIGHT_PX = 700;
const GREY = "6E6E6E";
// Sections that only track progress in the markdown; the notes section recomputes the limits.
const SKIP = new Set(["Limits check (delete before submission)", "Figure status (delete before submission)"]);

const lines = fs.readFileSync(SOURCE, "utf8").split("\n");

function stripMarkup(text) {
  return text.replace(/<sup>[^<]*<\/sup>/g, "").replace(/\*\*/g, "").replace(/`/g, "");
}
const wordCount = (text) => stripMarkup(text).replace(/#/g, "").split(/\s+/).filter(Boolean).length;

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|<sup>[^<]*<\/sup>|`[^`]+`)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("<sup>")) out.push(new TextRun({ text: t.slice(5, -6), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { data: buf, width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

function image(relPath) {
  const { data, width, height } = pngSize(path.join(VANCOUVER, relPath));
  const w = Math.min(IMAGE_WIDTH_PX, width, Math.floor((IMAGE_MAX_HEIGHT_PX * width) / height));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 120, after: 120 },
    children: [new ImageRun({ type: "png", data, transformation: { width: w, height: Math.round((w * height) / width) } })],
  });
}

// Group the markdown into {h2, h3, blocks}; a block is a paragraph, list item, or image.
const sections = [];
let current = null;
let para = [];
function flush() {
  if (para.length && current) current.blocks.push({ kind: "para", text: para.join(" "), h3: current.h3 });
  para = [];
}
for (const raw of lines) {
  const line = raw.trimEnd();
  if (line.startsWith("## ")) {
    flush();
    current = { h2: line.slice(3), h3: null, blocks: [] };
    sections.push(current);
  } else if (line.startsWith("### ")) {
    flush();
    current.h3 = line.slice(4);
    current.blocks.push({ kind: "h3", text: current.h3 });
  } else if (!current || line.startsWith("# ")) {
    continue;
  } else if (line === "") {
    flush();
  } else if (/^!\[.*\]\((.*)\)$/.test(line)) {
    flush();
    current.blocks.push({ kind: "image", path: line.match(/\((.*)\)/)[1], h3: current.h3 });
  } else if (line.startsWith("- ")) {
    flush();
    current.blocks.push({ kind: "bullet", text: line.slice(2).replace(/^\[ \] /, "☐ ") });
  } else if (/^\d+\. /.test(line)) {
    flush();
    current.blocks.push({ kind: "ref", text: line });
  } else if (line.startsWith("|")) {
    continue;
  } else {
    para.push(line);
    // Author and affiliation lines are separate lines, not one sentence-per-line paragraph.
    if (current.h2 === "Authors") flush();
  }
}
flush();

const byName = (prefix) => sections.find((s) => s.h2.startsWith(prefix));
const title = byName("Title").blocks.find((b) => b.kind === "para").text;

// Live limit counts, so the notes never go stale.
const bodyText = byName("Body").blocks.filter((b) => b.kind !== "image").map((b) => b.text).join(" ");
const synopsisText = byName("Synopsis").blocks.map((b) => b.text.replace(/\*\*[^*]+:\*\*/, "")).join(" ");
const impactText = byName("Impact").blocks.map((b) => b.text).join(" ");
const captions = [];
for (const b of byName("Figures").blocks) {
  if (b.kind === "para" && /^Figure \d/.test(b.h3 || "")) captions.push({ fig: b.h3, chars: b.text.length });
}

const children = [new Paragraph({ heading: HeadingLevel.TITLE, children: runs(title) })];
for (const s of sections) {
  if (SKIP.has(s.h2) || s.h2.startsWith("Title")) continue;
  const isNotes = s.h2.startsWith("Before submitting");
  children.push(new Paragraph({
    heading: HeadingLevel.HEADING_1,
    keepNext: true,
    children: runs(isNotes ? "Notes for co-authors (remove before submission)" : s.h2),
  }));
  if (isNotes) {
    const limits = [
      `Title ${title.length}/125 characters; impact ${wordCount(impactText)}/40 words; synopsis ${wordCount(synopsisText)}/100 words; body ${wordCount(bodyText)}/750 words including section headings.`,
      `Captions: ${captions.map((c) => `${c.fig} ${c.chars}`).join(", ")} (max 500 characters each).`,
    ];
    for (const l of limits) children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(l) }));
  }
  for (const b of s.blocks) {
    if (b.kind === "h3") {
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: runs(b.text) }));
    } else if (b.kind === "image") {
      children.push(image(b.path));
    } else if (b.kind === "bullet") {
      children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(b.text) }));
    } else if (b.kind === "ref") {
      children.push(new Paragraph({ indent: { left: 360, hanging: 360 }, spacing: { after: 60 }, children: runs(b.text) }));
    } else if (s.h2.startsWith("Figures") && /^Figure \d/.test(b.h3 || "")) {
      children.push(new Paragraph({
        keepNext: true,
        keepLines: true,
        children: [new TextRun({ text: `${b.h3}. `, bold: true }), ...runs(b.text)],
      }));
      children.push(new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun({ text: `Caption: ${b.text.length}/500 characters`, color: GREY, size: 18 })],
      }));
    } else if (s.h2.startsWith("Title")) {
      continue;
    } else {
      children.push(new Paragraph({ spacing: { after: 120 }, children: runs(b.text) }));
    }
  }
}

const doc = new Document({
  creator: "Arthur Allilaire",
  title,
  styles: { default: { document: { run: { font: "Arial", size: 21 } } } },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
    }],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT}`);
  console.log(`body ${wordCount(bodyText)} words, synopsis ${wordCount(synopsisText)}, impact ${wordCount(impactText)}, title ${title.length} chars`);
  console.log(captions.map((c) => `${c.fig}: ${c.chars}`).join(", "));
});
