import { formatGB } from "../utils/format";

interface Props {
  totalGb: number;
  freeGb: number;
  fitSizeGb?: number;
}

export function MemoryGauge({ totalGb, freeGb, fitSizeGb }: Props) {
  const usedGb = totalGb - freeGb;
  const usedPct = totalGb > 0 ? (usedGb / totalGb) * 100 : 0;
  const radius = 52;
  const stroke = 8;
  const circumference = 2 * Math.PI * radius;
  const usedOffset = circumference - (usedPct / 100) * circumference;

  const color =
    usedPct > 90 ? "var(--color-danger)" :
    usedPct > 70 ? "var(--color-warning)" :
    "var(--color-accent)";

  return (
    <div className="bg-bg-card border border-border rounded-xl p-5 flex flex-col items-center animate-fade-in">
      <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary mb-4 self-start">
        VRAM Usage
      </h2>

      <div className="relative w-32 h-32">
        <svg
          viewBox="0 0 128 128"
          className="w-full h-full -rotate-90"
        >
          {/* Track */}
          <circle
            cx="64" cy="64" r={radius}
            fill="none"
            stroke="var(--color-border)"
            strokeWidth={stroke}
          />
          {/* Used arc */}
          <circle
            cx="64" cy="64" r={radius}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={circumference}
            strokeDashoffset={usedOffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold font-[JetBrains_Mono] text-text-primary">
            {usedPct.toFixed(0)}%
          </span>
          <span className="text-[10px] text-text-muted">used</span>
        </div>
      </div>

      <div className="mt-4 w-full space-y-1.5 text-xs">
        <div className="flex justify-between">
          <span className="text-text-muted">Used</span>
          <span className="text-text-secondary font-[JetBrains_Mono]">{formatGB(usedGb)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Free</span>
          <span className="text-text-secondary font-[JetBrains_Mono]">{formatGB(freeGb)}</span>
        </div>
        <div className="flex justify-between border-t border-border pt-1.5">
          <span className="text-text-muted">Total</span>
          <span className="text-text-primary font-[JetBrains_Mono]">{formatGB(totalGb)}</span>
        </div>
        {fitSizeGb !== undefined && fitSizeGb > 0 && (
          <div className="flex justify-between">
            <span className="text-text-muted">Model</span>
            <span className="text-accent font-[JetBrains_Mono]">{formatGB(fitSizeGb)}</span>
          </div>
        )}
      </div>
    </div>
  );
}
