import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

interface Props {
  history: number[];
  currentTokS: number;
  peakTokS: number;
}

export function ThroughputChart({ history, currentTokS, peakTokS }: Props) {
  const chartData = history.map((v, i) => ({ idx: i, tokS: v }));

  return (
    <div className="bg-bg-card border border-border rounded-xl p-5 animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-text-secondary">
          Throughput
        </h2>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <span className="text-xs text-text-muted block">Current</span>
            <span className="text-sm font-[JetBrains_Mono] text-accent-bright">
              {currentTokS > 0 ? `${currentTokS.toFixed(1)}` : "—"}
            </span>
          </div>
          <div className="text-right">
            <span className="text-xs text-text-muted block">Peak</span>
            <span className="text-sm font-[JetBrains_Mono] text-text-primary">
              {peakTokS > 0 ? `${peakTokS.toFixed(1)}` : "—"}
            </span>
          </div>
          <span className="text-[10px] text-text-muted">tok/s</span>
        </div>
      </div>

      <div className="h-40">
        {chartData.length > 1 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="tokGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="idx" hide />
              <YAxis
                tick={{ fontSize: 10, fill: "#55556a" }}
                axisLine={false}
                tickLine={false}
                domain={[0, "auto"]}
              />
              <Tooltip
                contentStyle={{
                  background: "#141420",
                  border: "1px solid #2e2e48",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "#e2e2ea",
                }}
                formatter={(value) => [`${Number(value).toFixed(1)} tok/s`, ""]}
                labelFormatter={() => ""}
              />
              <Area
                type="monotone"
                dataKey="tokS"
                stroke="#818cf8"
                strokeWidth={2}
                fill="url(#tokGradient)"
                dot={false}
                animationDuration={300}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-text-muted text-sm">
            Waiting for inference data...
          </div>
        )}
      </div>
    </div>
  );
}
