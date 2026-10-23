// The LLM writes math with LaTeX delimiters: \( ... \) inline, \[ ... \] display.
// remark-math only recognises $ delimiters, and markdown would otherwise treat
// "\(" as an escaped "(" and drop the backslash. So convert the delimiters to
// $$ before markdown parsing:
//   \( x \)          -> $$x$$          (inline math)
//   \[ x \]          -> $$x$$          (on one line: inline-style)
//   \[ on its own line ... \] on its own line -> $$ fences (display block)
// Line structure and indentation are kept, so a display block inside a list
// item stays inside it. Code spans and fenced code blocks are left untouched.
//
// $$ is used even for inline math because single-dollar math is disabled
// (singleDollarTextMath: false): answers can contain prices like "$4.6M".

const CODE = /(```[\s\S]*?```|`[^`\n]*`)/g;

function convert(text) {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => `$$${body}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, body) => `$$${body.trim()}$$`);
}

export function normalizeMathDelimiters(markdown) {
  if (!markdown) return markdown;
  // Odd indices are code segments (captured by the split), even are prose
  return markdown
    .split(CODE)
    .map((part, i) => (i % 2 === 1 ? part : convert(part)))
    .join("");
}
