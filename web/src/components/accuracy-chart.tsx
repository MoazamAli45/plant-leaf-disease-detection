/**
 * Lab vs field accuracy per model (mean ± std over seeds). Static SVG; the same numbers are
 * in the table next to it, so the chart is not the only way to read them.
 */
type Point = { model: string; lab: { mean: number; sd: number }; field: { mean: number; sd: number } };

const ROW = 44;
const LEFT = 150;
const RIGHT = 56;
const WIDTH = 640;

export function AccuracyChart({ data }: { data: Point[] }) {
  const plotW = WIDTH - LEFT - RIGHT;
  const height = data.length * ROW + 40;
  const x = (v: number) => LEFT + (v / 100) * plotW;
  const series = [
    { key: "lab" as const, label: "Lab test (leak-free)", color: "var(--series-1)", dy: -7 },
    { key: "field" as const, label: "Field photos (PlantDoc)", color: "var(--series-2)", dy: 7 },
  ];
  const summary = data
    .map((d) => `${d.model}: lab ${d.lab.mean.toFixed(1)}%, field ${d.field.mean.toFixed(1)}%`)
    .join("; ");
  return (
    <figure className="space-y-2">
      <div className="flex flex-wrap gap-4 text-sm" aria-hidden>
        {series.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-2">
            <span className="inline-block size-3 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${height}`} className="w-full" role="img" aria-label={`Accuracy by model. ${summary}`}>
        {[0, 25, 50, 75, 100].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={8} y2={height - 26} stroke="var(--border)" strokeWidth={1} />
            <text x={x(t)} y={height - 8} textAnchor="middle" fontSize={11} fill="var(--muted-foreground)">
              {t}%
            </text>
          </g>
        ))}
        {data.map((d, i) => {
          const cy = 26 + i * ROW;
          return (
            <g key={d.model}>
              <text x={LEFT - 10} y={cy + 4} textAnchor="end" fontSize={12} fill="var(--foreground)">
                {d.model}
              </text>
              {series.map((s) => {
                const v = d[s.key];
                return (
                  <g key={s.key}>
                    <title>{`${d.model}, ${s.label}: ${v.mean.toFixed(2)}% ± ${v.sd.toFixed(2)}`}</title>
                    <line x1={x(Math.max(0, v.mean - v.sd))} x2={x(Math.min(100, v.mean + v.sd))} y1={cy + s.dy} y2={cy + s.dy} stroke={s.color} strokeWidth={2} />
                    <circle cx={x(v.mean)} cy={cy + s.dy} r={5} fill={s.color} stroke="var(--background)" strokeWidth={2} />
                    <text x={x(Math.min(100, v.mean + v.sd)) + 8} y={cy + s.dy + 4} fontSize={11} fill="var(--foreground)">
                      {v.mean.toFixed(1)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <figcaption className="text-sm text-muted-foreground">
        Top-1 accuracy, mean ± standard deviation over 3 training seeds. Chance is 2.6%.
      </figcaption>
    </figure>
  );
}
