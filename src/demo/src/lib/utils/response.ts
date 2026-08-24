/**
 * Response processing utilities for formatting model output.
 * Handles code extraction, markdown formatting, math delimiters, and indentation.
 */

const FENCE_PLACEHOLDER = '\u0000FENCE';

/**
 * Convert LaTeX `\(...\)` / `\[...\]` to `$...$` / `$$...$$` for remark-math.
 * Matches the closing backslash-delimiter so nested parentheses stay intact.
 * Fenced code blocks are left unchanged.
 */
export function normalizeMathDelimiters(content: string): string {
  if (!content) return content;

  const fences: string[] = [];
  const withPlaceholders = content.replace(/```[\s\S]*?```/g, (block) => {
    const token = `${FENCE_PLACEHOLDER}${fences.length}\u0000`;
    fences.push(block);
    return token;
  });

  const converted = convertLatexDelimiters(withPlaceholders);

  return converted.replace(
    new RegExp(`${FENCE_PLACEHOLDER}(\\d+)\u0000`, 'g'),
    (_, i) => fences[Number(i)]
  );
}

function convertLatexDelimiters(content: string): string {
  let result = '';
  let i = 0;
  const n = content.length;

  while (i < n) {
    if (content[i] === '\\' && i + 1 < n) {
      const next = content[i + 1];
      if (next === '[' || next === '(') {
        const closer = next === '[' ? '\\]' : '\\)';
        const close = findClosingDelimiter(content, i + 2, closer);
        if (close !== -1) {
          const inner = content.slice(i + 2, close).trim();
          if (next === '[') {
            result += `\n$$\n${inner}\n$$\n`;
          } else {
            result += `$${inner}$`;
          }
          i = close + 2;
          continue;
        }
      }
    }
    result += content[i];
    i++;
  }

  return result;
}

function findClosingDelimiter(content: string, start: number, closer: string): number {
  for (let i = start; i < content.length - 1; i++) {
    if (content[i] !== closer[0] || content[i + 1] !== closer[1]) continue;

    let backslashes = 0;
    for (let j = i - 1; j >= start && content[j] === '\\'; j--) {
      backslashes++;
    }
    if (backslashes % 2 === 0) {
      return i;
    }
  }
  return -1;
}

/**
 * Detect whether text looks like Python/Qiskit (or similar) code.
 */
export function looksLikeCode(text: string): boolean {
  if (text.includes('\n')) {
    const codeIndicators = [
      /^from\s+/m,
      /^import\s+/m,
      /^def\s+/m,
      /^class\s+/m,
      /^\s*return\s+/m,
      /QuantumCircuit/,
      /Parameter\(/,
      /\.\w+\([^)]*\)/m,
    ];
    return codeIndicators.some((p) => p.test(text));
  }

  const singleLinePatterns = [
    /^return\s+\w+/,
    /^\w+\s*=\s*\w+\([^)]*\)/,
    /^\w+\.\w+\([^)]*\)$/,
    /\w+\s*=\s*\w+\([^)]*\)(?:\s+\w+\.|\s+\w+\s*=)/,
    /QuantumCircuit\(/,
    /Parameter\(/,
    /\.control\(/,
    /\.measure\(/,
  ];
  return singleLinePatterns.some((p) => p.test(text.trim()));
}

/**
 * Prepare model output for markdown rendering: math delimiters, then
 * wrap unfenced code in a python block when the whole response is code.
 */
export function prepareMarkdownContent(content: string): string {
  let prepared = normalizeMathDelimiters(content);

  if (
    !prepared.includes('```') &&
    !prepared.includes('$$') &&
    !prepared.includes('$') &&
    looksLikeCode(prepared)
  ) {
    prepared = prepared
      .replace(/(\w+\s*=\s*\w+\([^)]*\))\s+(\w+\.)/g, '$1\n$2')
      .replace(/(\w+\.[a-z_]+\([^)]*\))\s+(\w+\.)/g, '$1\n$2');
    prepared = '```python\n' + prepared + '\n```';
  }

  return prepared;
}

/**
 * Strip markdown fences and collapse whitespace for compact list previews.
 * If the question is only a code fence, keep a short excerpt (e.g. the def line).
 */
export function previewText(text: string, maxLength: number = 120): string {
  const fenceBodies: string[] = [];
  const withoutFences = text.replace(/```(?:\w+)?\s*\n?([\s\S]*?)```/g, (_, body) => {
    fenceBodies.push(body);
    return ' ';
  });

  let preview = withoutFences
    .replace(/`+/g, '')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (preview.length < 24 && fenceBodies.length > 0) {
    const lines = fenceBodies[0]
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && line !== 'pass' && !line.startsWith('#'));
    const excerpt = lines.find((line) => line.startsWith('def ')) || lines[0];
    if (excerpt) {
      preview = preview ? `${preview} ${excerpt}` : excerpt;
    }
  }

  if (preview.length <= maxLength) return preview;
  return preview.substring(0, maxLength).trimEnd() + '...';
}

/**
 * Extract code blocks from model response.
 * Handles markdown code blocks and detects code patterns.
 */
export function extractCodeFromResponse(response: string, entryPoint?: string): string {
  // Find all markdown code blocks
  const codeBlockRegex = /```(?:python)?\s*\n([\s\S]*?)```/g;
  const matches: string[] = [];
  let match;

  while ((match = codeBlockRegex.exec(response)) !== null) {
    // Preserve indentation - only trim trailing whitespace, not leading
    matches.push(match[1].replace(/\s+$/, ''));
  }

  if (matches.length === 0) {
    // No code blocks found - the response itself might be code
    // Preserve indentation by only trimming trailing whitespace
    return response.replace(/\s+$/, '');
  }

  if (matches.length === 1) {
    return matches[0];
  }

  // If multiple blocks, prefer one with entry point
  if (entryPoint) {
    const entryPointRegex = new RegExp(`def\\s+${escapeRegex(entryPoint)}\\s*\\(`);
    for (const block of matches) {
      if (entryPointRegex.test(block)) {
        return block;
      }
    }
  }

  // Return longest block
  return matches.reduce((a, b) => (a.length > b.length ? a : b));
}

/**
 * Detect if text contains Python code patterns.
 */
export function detectsPythonCode(text: string): boolean {
  const pythonPatterns = [
    /^from\s+\w+\s+import/m,
    /^import\s+\w+/m,
    /^def\s+\w+\s*\(/m,
    /^class\s+\w+/m,
    /^\s*@\w+/m, // decorators
    /QuantumCircuit\s*\(/,
    /\.h\s*\(/,
    /\.cx\s*\(/,
    /\.measure/,
    /qc\s*=\s*QuantumCircuit/,
  ];

  return pythonPatterns.some((pattern) => pattern.test(text));
}

/**
 * Format response with proper markdown code blocks.
 * Ensures code is properly fenced for rendering.
 */
export function formatResponseWithCodeBlocks(response: string): string {
  // If response already has code blocks, return as-is
  if (/```[\s\S]*```/.test(response)) {
    return response;
  }

  // Check if the entire response looks like code
  const lines = response.split('\n');
  const codeLines = lines.filter((line) => {
    const trimmed = line.trim();
    return (
      trimmed.startsWith('from ') ||
      trimmed.startsWith('import ') ||
      trimmed.startsWith('def ') ||
      trimmed.startsWith('class ') ||
      trimmed.startsWith('@') ||
      trimmed.startsWith('#') ||
      /^\s*\w+\s*=/.test(trimmed) ||
      /^\s*\w+\.\w+\(/.test(trimmed) ||
      /^\s*return\s/.test(trimmed) ||
      /^\s*if\s/.test(trimmed) ||
      /^\s*for\s/.test(trimmed) ||
      /^\s*while\s/.test(trimmed) ||
      /^\s*try:/.test(trimmed) ||
      /^\s*except/.test(trimmed) ||
      trimmed === '' ||
      trimmed === 'pass'
    );
  });

  // If most lines look like code, wrap entire response
  if (codeLines.length > lines.length * 0.7 && detectsPythonCode(response)) {
    return '```python\n' + response.trim() + '\n```';
  }

  // Try to detect inline code that should be blocks
  // Pattern: text followed by code on same line or multiple statements
  const inlineCodePattern =
    /(from\s+\w+\s+import\s+[\w,\s]+)\s+([\w]+\s*=\s*\w+\([^)]*\)(?:\s+[\w.]+\([^)]*\))*)/g;

  if (inlineCodePattern.test(response)) {
    // Split inline code into proper lines
    const formatted = response
      .replace(
        /(from\s+\w+\s+import\s+[\w,\s]+)/g,
        '\n```python\n$1'
      )
      .replace(
        /\s+([\w]+\s*=\s*\w+\([^)]*\))/g,
        '\n$1'
      )
      .replace(
        /(\s+[\w.]+\([^)]*\))(?=\s+[\w.]+\()/g,
        '$1\n'
      );

    // Clean up and close code block
    const lines = formatted.split('\n');
    let inCodeBlock = false;
    const result: string[] = [];

    for (const line of lines) {
      if (line.includes('```python')) {
        inCodeBlock = true;
      }
      result.push(line);
    }

    if (inCodeBlock) {
      result.push('```');
    }

    return result.join('\n');
  }

  return response;
}

/**
 * Process streaming chunk to maintain markdown structure.
 * Handles partial code blocks during streaming.
 */
export function processStreamingContent(
  fullContent: string,
  previousContent: string
): { content: string; isInCodeBlock: boolean } {
  // Count code block markers
  const openMarkers = (fullContent.match(/```/g) || []).length;
  const isInCodeBlock = openMarkers % 2 === 1;

  return {
    content: fullContent,
    isInCodeBlock,
  };
}

/**
 * Normalize code indentation.
 * Similar to _normalize_body_indentation in synthetic.py
 * 
 * Handles the common pattern where model outputs function completion code with:
 * - First line at 0 indentation
 * - Subsequent lines with extra indentation (e.g., 4 spaces)
 */
export function normalizeIndentation(code: string, targetIndent: number = 0): string {
  const lines = code.split('\n');
  const nonEmptyLines = lines
    .map((line, idx) => ({ line, idx }))
    .filter(({ line }) => line.trim().length > 0);

  if (nonEmptyLines.length === 0) {
    return code;
  }

  // Get first non-empty line's indentation
  const firstNonEmpty = nonEmptyLines[0];
  const firstIndent = getIndent(firstNonEmpty.line);

  // Check for the common pattern: first line at 0, rest at 4+
  if (firstIndent === 0 && nonEmptyLines.length > 1) {
    const subsequentIndents = nonEmptyLines.slice(1).map(({ line }) => getIndent(line));
    const minSubsequent = Math.min(...subsequentIndents);

    // If subsequent lines have extra indentation, they should align with first line
    if (minSubsequent > 0) {
      const result: string[] = [];
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (!line.trim()) {
          result.push('');
        } else if (i === firstNonEmpty.idx) {
          // First line gets target indent
          result.push(' '.repeat(targetIndent) + line.trim());
        } else {
          // Subsequent lines: remove extra base indent, add target
          const currentIndent = getIndent(line);
          const relative = currentIndent - minSubsequent;
          const newIndent = ' '.repeat(targetIndent + Math.max(0, relative));
          result.push(newIndent + line.trim());
        }
      }
      return result.join('\n');
    }
  }

  // Standard case: subtract min indent and add target
  const minIndent = Math.min(
    ...nonEmptyLines.map(({ line }) => getIndent(line))
  );

  return lines
    .map((line) => {
      if (line.trim().length === 0) {
        return '';
      }
      const currentIndent = getIndent(line);
      const relativeIndent = currentIndent - minIndent;
      const newIndent = ' '.repeat(targetIndent + relativeIndent);
      return newIndent + line.trim();
    })
    .join('\n');
}

/**
 * Get the indentation level of a line.
 */
function getIndent(line: string): number {
  const match = line.match(/^(\s*)/);
  return match ? match[1].length : 0;
}

/**
 * Post-process complete response for display.
 * Applies formatting, code detection, and normalization.
 */
export function postProcessResponse(response: string): string {
  if (!response || response.trim().length === 0) {
    return response;
  }

  // First, try to format with proper code blocks
  let processed = formatResponseWithCodeBlocks(response);

  // Normalize indentation within code blocks
  processed = processed.replace(
    /```python\n([\s\S]*?)```/g,
    (match, code) => {
      const normalized = normalizeIndentation(code.trim());
      return '```python\n' + normalized + '\n```';
    }
  );

  return processed;
}

function escapeRegex(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

