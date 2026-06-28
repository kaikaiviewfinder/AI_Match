import { useEffect, useRef } from 'react'

interface RoutePoint { x: number; y: number; z: number; s: number; lon: number; lat: number }
interface Props {
  routePoints: RoutePoint[]
  pathHistory: PathPoint[]
  gpsFixes: { lat: number; lon: number }[]
  gpsToLocal: (lon: number, lat: number, olon: number, olat: number) => { x: number; y: number }
  width?: number; height?: number
}

export default function TrajectoryMap({ routePoints, pathHistory, gpsFixes, gpsToLocal, width, height }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const cv = ref.current; if (!cv) return
    const ctx = cv.getContext('2d')!; const dpr = devicePixelRatio || 1
    const W = cv.parentElement!.clientWidth, H = cv.parentElement!.clientHeight
    cv.width = W * dpr; cv.height = H * dpr
    cv.style.width = W + 'px'; cv.style.height = H + 'px'
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(dpr, dpr)

    const allPts: { x: number; y: number }[] = []
    routePoints.forEach(p => allPts.push({ x: p.x, y: p.y }))
    pathHistory.forEach(p => allPts.push({ x: p.x, y: p.y }))

    function w2c(px: number, py: number) {
      if (allPts.length < 2) return { cx: W / 2, cy: H / 2 }
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
      allPts.forEach(p => {
        if (Math.abs(p.x) > 1e6 || Math.abs(p.y) > 1e6) return
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y
      })
      const rX = (maxX - minX) || 1, rY = (maxY - minY) || 1
      const s = Math.min((W - 60) / rX, (H - 50) / rY)
      return { cx: 30 + (px - minX) * s, cy: H - 25 - (py - minY) * s }
    }

    ctx.clearRect(0, 0, W, H)
    ctx.strokeStyle = '#F1F5F9'; ctx.lineWidth = 0.5
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke() }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }

    // Route line
    if (routePoints.length > 1) {
      ctx.beginPath()
      const r0 = w2c(routePoints[0].x, routePoints[0].y)
      ctx.moveTo(r0.cx, r0.cy)
      for (let i = 1; i < routePoints.length; i++) {
        const r = w2c(routePoints[i].x, routePoints[i].y); ctx.lineTo(r.cx, r.cy)
      }
      ctx.strokeStyle = '#CBD5E1'; ctx.lineWidth = 1.5
      ctx.setLineDash([4, 8]); ctx.stroke(); ctx.setLineDash([])
    }

    // GPS dots
    if (routePoints.length > 0 && gpsFixes.length > 0) {
      const o = routePoints[0]
      gpsFixes.forEach(gf => {
        const l = gpsToLocal(gf.lon, gf.lat, o.lon, o.lat)
        const gc = w2c(l.x, l.y)
        ctx.beginPath(); ctx.arc(gc.cx, gc.cy, 2.2, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(245,158,11,0.5)'; ctx.fill()
      })
    }

    // Path history
    if (pathHistory.length > 1) {
      ctx.beginPath()
      const ph0 = w2c(pathHistory[0].x, pathHistory[0].y)
      ctx.moveTo(ph0.cx, ph0.cy)
      for (let i = 1; i < pathHistory.length; i++) {
        const ph = w2c(pathHistory[i].x, pathHistory[i].y); ctx.lineTo(ph.cx, ph.cy)
      }
      ctx.strokeStyle = 'rgba(37,99,235,0.3)'; ctx.lineWidth = 2
      ctx.setLineDash([6, 3]); ctx.stroke(); ctx.setLineDash([])
    }

    // Recent matched (last 80)
    const st = Math.max(0, pathHistory.length - 80)
    if (st < pathHistory.length - 1) {
      ctx.beginPath()
      const rs = w2c(pathHistory[st].x, pathHistory[st].y)
      ctx.moveTo(rs.cx, rs.cy)
      for (let i = st + 1; i < pathHistory.length; i++) {
        const rp = w2c(pathHistory[i].x, pathHistory[i].y); ctx.lineTo(rp.cx, rp.cy)
      }
      ctx.strokeStyle = '#2563EB'; ctx.lineWidth = 2.5; ctx.stroke()
    }

    // Current position
    if (pathHistory.length > 0) {
      const last = pathHistory[pathHistory.length - 1]
      const lp = w2c(last.x, last.y)
      const t = Date.now() / 600
      const pulse = 1 + 0.25 * Math.sin(t)
      ctx.beginPath(); ctx.arc(lp.cx, lp.cy, 6, 0, Math.PI * 2)
      ctx.fillStyle = '#2563EB'; ctx.fill()
      ctx.beginPath(); ctx.arc(lp.cx, lp.cy, 13 * pulse, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(37,99,235,${0.1 * pulse})`; ctx.fill()
      // direction triangle
      const yaw = 0
      ctx.beginPath()
      ctx.moveTo(lp.cx + 10 * Math.cos(yaw), lp.cy + 10 * Math.sin(yaw))
      ctx.lineTo(lp.cx - 5 * Math.cos(yaw - 0.6), lp.cy - 5 * Math.sin(yaw - 0.6))
      ctx.lineTo(lp.cx - 5 * Math.cos(yaw + 0.6), lp.cy - 5 * Math.sin(yaw + 0.6))
      ctx.closePath(); ctx.fillStyle = '#2563EB'; ctx.fill()
    }
  }, [routePoints, pathHistory, gpsFixes])

  return <canvas ref={ref} className="w-full h-full rounded-xl" />
}
