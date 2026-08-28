/**
 * THESEUS KaTeX Math Renderer
 * ===========================
 * Typesets mathematical formulas and equations using KaTeX.
 * Features safe fallback to styled monospace text if KaTeX fails or LaTeX is invalid.
 */

import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MathRendererProps {
  equation: string;
  inline?: boolean;
  className?: string;
}

export const MathRenderer: React.FC<MathRendererProps> = ({
  equation,
  inline = false,
  className = '',
}) => {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!containerRef.current || !equation) return;

    try {
      katex.render(equation, containerRef.current, {
        displayMode: !inline,
        throwOnError: false,
        output: 'htmlAndMathml',
      });
    } catch {
      // Fallback: raw text rendering if KaTeX fails
      if (containerRef.current) {
        containerRef.current.innerText = equation;
      }
    }
  }, [equation, inline]);

  return (
    <span
      ref={containerRef}
      className={`font-mono text-[#ffaa00] font-bold ${className}`}
    >
      {equation}
    </span>
  );
};
