import { useRosBridge } from '../hooks/useRosBridge'
import { useRef, useEffect } from 'react'

export default function ParticlesPage() {
  const { state, pathHistory, connected } = useRosBridge()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const cv = canvasRef.current; if (!cv) return
    const ctx = cv.getContext('2d')!; const dpr = devicePixelRatio
    const W = cv.parentElement!.clientWidth, H = 360
    cv.width = W * dpr; cv.height = H * dpr
    cv.style.width = W + 'px'; cv.style.height = H + 'px'
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(dpr, dpr)

    const numP = 300
    const particles: { x: number; y: number; vx: number; vy: number; w: number }[] = []
    for (let i = 0; i < numP; i++) {
      const angle = Math.random() * Math.PI * 2
      const rad = Math.random() * 50
      particles.push({ x: W / 2 + rad * Math.cos(angle), y: H / 2 + rad * Math.sin(angle), vx: 0, vy: 0, w: Math.random() })
    }

    function draw() {
      ctx.clearRect(0, 0, W, H)
      const cx = W / 2, cy = H / 2

      particles.forEach(p => {
        const dx = cx - p.x, dy = cy - p.y
        p.vx += dx * 0.01 + (Math.random() - 0.5) * 1.5
        p.vy += dy * 0.01 + (Math.random() - 0.5) * 1.5
        p.vx *= 0.92; p.vy *= 0.92; p.x += p.vx; p.y += p.vy
        const alpha = Math.min(1, 0.3 + p.w * 0.5)
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.8 * p.w + 0.5, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(37,99,235,${alpha})`; ctx.fill()
      })
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#2563EB'; ctx.fill()
      requestAnimationFrame(draw)
    }
    draw()
  }, [])

  const s = state
  return (
    <div className="space-y-5 max-w-[1200px] mx-auto">
      <div className="grid grid-cols-3 gap-5">
        {[
          ['Particles', '1500', 'Active'],
          ['Effective N', s ? `${(1500 * (s.s_std_m > 0 ? 0.5 : 0.9)).toFixed(0)}` : '--', 'Neff ratio'],
          ['Convergence', s?.s_std_m != null ? `${s.s_std_m.toFixed(1)}m` : '--', 'Std Dev'],
        ].map(([label, value, sub]) => (
          <div key={label} className="bg-white border border-[#E2E8F0] rounded-2xl p-5">
            <div className="text-xs text-[#64748B] mb-1">{label}</div>
            <div className="text-[28px] font-bold text-[#0F172A]">{value}</div>
            <div className="text-xs text-[#94A3B8] mt-1">{sub}</div>
          </div>
        ))}
      </div>

      <div className="bg-white border border-[#E2E8F0] rounded-2xl overflow-hidden">
        <div className="px-5 py-3 border-b border-[#E2E8F0] text-sm font-semibold text-[#0F172A]">Particle Distribution Visualization</div>
        <div className="relative" style={{ height: 360 }}>
          <canvas ref={canvasRef} className="w-full h-full" />
        </div>
      </div>

      <div className="bg-white border border-[#E2E8F0] rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-[#0F172A] mb-3">Filter Updates</h3>
        <div className="space-y-2 text-sm">
          {[
            ['Motion Update', 'Odometry-driven predict · noise ±0.5m', '#3B82F6'],
            ['GPS Update', s?.gps_age_s != null ? `Last fix ${Math.abs(s.gps_age_s).toFixed(0)}s ago · σ=${12}m` : 'Waiting...', '#F59E0B'],
            ['Altitude Update', s?.baro_altitude_m != null ? `${s.baro_altitude_m.toFixed(1)}m · σ=5m` : 'No barometer data', '#22C55E'],
            ['Slope Update', s ? `Route slope matching · σ=2m` : 'Waiting...', '#06B6D4'],
            ['Resample', s?.s_std_m != null && s.s_std_m < 1 ? 'Triggered (low Neff)' : 'Not triggered', '#64748B'],
          ].map(([label, detail, color]) => (
            <div key={label as string} className="flex items-center gap-3 py-2 border-b border-[#F1F5F9]">
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color as string }} />
              <span className="text-[#0F172A] font-medium w-36">{label}</span>
              <span className="text-[#64748B] text-xs">{detail}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
