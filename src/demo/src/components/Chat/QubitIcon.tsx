'use client';

interface QubitIconProps {
    size?: number;
    className?: string;
}

/**
 * Minimalist qubit icon representing a Bloch sphere.
 * Simple, clean design suitable for a professional interface.
 */
export function QubitIcon({ size = 24, className = '' }: QubitIconProps) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className={className}
        >
            {/* Outer circle - Bloch sphere */}
            <circle
                cx="12"
                cy="12"
                r="9"
                stroke="currentColor"
                strokeWidth="1.5"
                fill="none"
                opacity="0.6"
            />

            {/* Equator ellipse */}
            <ellipse
                cx="12"
                cy="12"
                rx="9"
                ry="3"
                stroke="currentColor"
                strokeWidth="1"
                fill="none"
                opacity="0.4"
            />

            {/* Vertical meridian */}
            <ellipse
                cx="12"
                cy="12"
                rx="3"
                ry="9"
                stroke="currentColor"
                strokeWidth="1"
                fill="none"
                opacity="0.4"
            />

            {/* State vector arrow */}
            <line
                x1="12"
                y1="12"
                x2="16"
                y2="6"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
            />

            {/* State point */}
            <circle
                cx="16"
                cy="6"
                r="2"
                fill="currentColor"
            />

            {/* Center point */}
            <circle
                cx="12"
                cy="12"
                r="1.5"
                fill="currentColor"
                opacity="0.5"
            />
        </svg>
    );
}

