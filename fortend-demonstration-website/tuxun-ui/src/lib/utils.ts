export function cn(...classes: (string | boolean | undefined | null)[]) {
  return classes.filter(Boolean).join(' ')
}

export function formatCoord(val: number, decimals = 6): string {
  return val.toFixed(decimals)
}

export function formatPercent(val: number): string {
  return (val * 100).toFixed(1) + '%'
}

export function formatDist(m: number): string {
  if (m >= 1000) return (m / 1000).toFixed(2) + ' km'
  return m.toFixed(0) + ' m'
}

export function formatDuration(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}
