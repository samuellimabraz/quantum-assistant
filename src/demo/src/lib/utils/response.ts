/**
 * Response processing utilities for formatting model output.
 * Handles code extraction, markdown formatting, and indentation normalization.
 */

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

