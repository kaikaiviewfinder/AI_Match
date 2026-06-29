import { useEffect, useRef, useMemo } from 'react'

interface RoutePoint { x: number; y: number; z: number; s: number; lon: number; lat: number }
interface PathPoint { x: number; y: number; z: number; s: number; progress: number; confidence: number }

interface Props {
  routePoints: RoutePoint[]
  pathHistory: PathPoint[]
  gpsFixes: { lat: number; lon: number }[]
  gpsToLocal: (lon: number, lat: number, olon: number, olat: number) => { x: number; y: number }
  alignedVioPath?: { x: number; y: number; z: number }[]
}

export default function TrajectoryMap({ routePoints, pathHistory, gpsFixes, gpsToLocal, alignedVioPath }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  // Pre-compute GPS local coordinates
  const gpsLocal = useMemo(() => {
    if (routePoints.length === 0) return []
    const o = routePoints[0]
    return gpsFixes.map(gf => ({ ...gpsToLocal(gf.lon, gf.lat, o.lon, o.lat), lon: gf.lon, lat: gf.lat }))
  }, [gpsFixes, routePoints, gpsToLocal])

  useEffect(() => {
    const cv = ref.current; if (!cv) return
    const ctx = cv.getContext('2d')!; const dpr = devicePixelRatio || 1
    const W = cv.parentElement!.clientWidth, H = cv.parentElement!.clientHeight
    cv.width = W * dpr; cv.height = H * dpr
    cv.style.width = W + 'px'; cv.style.height = H + 'px'
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(dpr, dpr)

    // Build bounding box from ALL data: route + path + GPS + VIO
    const allPts: { x: number; y: number }[] = []
    routePoints.forEach(p => allPts.push({ x: p.x, y: p.y }))
    pathHistory.forEach(p => allPts.push({ x: p.x, y: p.y }))
    gpsLocal.forEach(p => allPts.push({ x: p.x, y: p.y }))
    if (alignedVioPath) alignedVioPath.forEach(p => allPts.push({ x: p.x, y: p.y }))

    function w2c(px: number, py: number) {
      if (allPts.length < 2) return { cx: W / 2, cy: H / 2 }
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
      allPts.forEach(p => {
        if (Math.abs(p.x) > 1e6 || Math.abs(p.y) > 1e6) return
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y
      })
      // Add 15% padding to ensure GPS dots are visible
      const padX = (maxX - minX) * 0.15 || 10
      const padY = (maxY - minY) * 0.15 || 10
      const rX = (maxX - minX + 2 * padX) || 1, rY = (maxY - minY + 2 * padY) || 1
      const s = Math.min((W - 60) / rX, (H - 50) / rY)
      return { cx: 30 + (px - minX + padX) * s, cy: H - 25 - (py - minY + padY) * s }
    }

    // Draw
    ctx.clearRect(0, 0, W, H)

    // Grid
    ctx.strokeStyle = '#F1F5F9'; ctx.lineWidth = 0.5
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke() }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }

    if (routePoints.length < 2) {
      ctx.fillStyle = '#94A3B8'; ctx.font = '14px Inter, sans-serif'
      ctx.textAlign = 'center'; ctx.fillText('Loading route data...', W / 2, H / 2)
      return
    }

    // 1. Known route (grey dashed)
    ctx.beginPath()
    const r0 = w2c(routePoints[0].x, routePoints[0].y)
    ctx.moveTo(r0.cx, r0.cy)
    for (let i = 1; i < routePoints.length; i++) {
      const r = w2c(routePoints[i].x, routePoints[i].y); ctx.lineTo(r.cx, r.cy)
    }
    ctx.strokeStyle = '#CBD5E1'; ctx.lineWidth = 1.5
    ctx.setLineDash([5, 8]); ctx.stroke(); ctx.setLineDash([])

    // 2. GPS dots (orange circles with glow) - with distance lines to route
    if (gpsLocal.length > 0) {
      gpsLocal.forEach(gp => {
        // Find nearest route point distance
        let minD2 = Infinity, nearestX = gp.x, nearestY = gp.y
        for (const rp of routePoints) {
          const d2 = (rp.x - gp.x) ** 2 + (rp.y - gp.y) ** 2
          if (d2 < minD2) { minD2 = d2; nearestX = rp.x; nearestY = rp.y }
        }
        const gc = w2c(gp.x, gp.y)
        const nc = w2c(nearestX, nearestY)
        const dist = Math.sqrt(minD2)

        // Only draw if within reasonable range
        if (dist < 200) {
          // Faint line from GPS to nearest route point
          ctx.beginPath(); ctx.moveTo(gc.cx, gc.cy); ctx.lineTo(nc.cx, nc.cy)
          ctx.strokeStyle = 'rgba(245,158,11,0.12)'; ctx.lineWidth = 0.5; ctx.stroke()

          // GPS dot with glow
          ctx.beginPath(); ctx.arc(gc.cx, gc.cy, 4.5, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(245,158,11,0.25)'; ctx.fill()
          ctx.beginPath(); ctx.arc(gc.cx, gc.cy, 2.8, 0, Math.PI * 2)
          ctx.fillStyle = '#F59E0B'; ctx.fill()
          ctx.strokeStyle = 'rgba(245,158,11,0.4)'; ctx.lineWidth = 1; ctx.stroke()
        }
      })
    }

    // 3. VINS raw VIO trajectory (green)
    if (alignedVioPath && alignedVioPath.length > 1) {
      ctx.beginPath()
      const v0 = w2c(alignedVioPath[0].x, alignedVioPath[0].y)
      ctx.moveTo(v0.cx, v0.cy)
      for (let i = 1; i < alignedVioPath.length; i++) {
        const v = w2c(alignedVioPath[i].x, alignedVioPath[i].y); ctx.lineTo(v.cx, v.cy)
      }
      ctx.strokeStyle = 'rgba(34,197,94,0.35)'; ctx.lineWidth = 2
      ctx.setLineDash([8, 5]); ctx.stroke(); ctx.setLineDash([])
    }

    // 4. Full path history (blue dashed)
    if (pathHistory.length > 1) {
      ctx.beginPath()
      const ph0 = w2c(pathHistory[0].x, pathHistory[0].y)
      ctx.moveTo(ph0.cx, ph0.cy)
      for (let i = 1; i < pathHistory.length; i++) {
        const ph = w2c(pathHistory[i].x, pathHistory[i].y); ctx.lineTo(ph.cx, ph.cy)
      }
      ctx.strokeStyle = 'rgba(59,130,246,0.25)'; ctx.lineWidth = 1.5
      ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([])
    }

    // 4. Recent matched path (solid blue, last 60 points)
    const st = Math.max(0, pathHistory.length - 60)
    if (st < pathHistory.length - 1) {
      ctx.beginPath()
      const rs = w2c(pathHistory[st].x, pathHistory[st].y)
      ctx.moveTo(rs.cx, rs.cy)
      for (let i = st + 1; i < pathHistory.length; i++) {
        const rp = w2c(pathHistory[i].x, pathHistory[i].y); ctx.lineTo(rp.cx, rp.cy)
      }
      ctx.strokeStyle = '#2563EB'; ctx.lineWidth = 3; ctx.stroke()
      // Soft glow
      ctx.strokeStyle = 'rgba(37,99,235,0.15)'; ctx.lineWidth = 8; ctx.stroke()
    }

    // 5. Current position marker
    if (pathHistory.length > 0) {
      const last = pathHistory[pathHistory.length - 1]
      const lp = w2c(last.x, last.y)
      const t = Date.now() / 600
      const pulse = 1 + 0.25 * Math.sin(t)

      // Outer pulse ring
      ctx.beginPath(); ctx.arc(lp.cx, lp.cy, 14 * pulse, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(37,99,235,${0.08 * pulse})`; ctx.fill()
      // Inner dot
      ctx.beginPath(); ctx.arc(lp.cx, lp.cy, 5, 0, Math.PI * 2)
      ctx.fillStyle = '#2563EB'; ctx.fill()
      ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 2; ctx.stroke()
    }

    // 6. Stats text overlay
    if (pathHistory.length > 0) {
      const last = pathHistory[pathHistory.length - 1]
      const gpsCount = gpsLocal.length
      const avgDist = gpsLocal.length > 0
        ? gpsLocal.reduce((sum, gp) => {
            let minD2 = Infinity
            for (const rp of routePoints) { minD2 = Math.min(minD2, (rp.x - gp.x) ** 2 + (rp.y - gp.y) ** 2) }
            return sum + Math.sqrt(minD2)
          }, 0) / gpsLocal.length
        : 0

      ctx.fillStyle = 'rgba(15,23,42,0.75)'; ctx.font = '11px Inter, sans-serif'
      ctx.textAlign = 'left'
      const lines = [
        `GPS: ${gpsCount} fixes · avg ${avgDist.toFixed(0)}m from route`,
        `Matched: s=${last.s.toFixed(0)}m · ±${last.s !== undefined ? '--' : '--'}m`,
        `Confidence: ${((last.confidence || 0) * 100).toFixed(0)}%`,
      ]
      lines.forEach((line, i) => {
        ctx.fillText(line, 10, H - 10 - (lines.length - 1 - i) * 16)
      })
    }

    if (pathHistory.length < 2 && gpsLocal.length === 0) {
      ctx.fillStyle = '#94A3B8'; ctx.font = '14px Inter, sans-serif'
      ctx.textAlign = 'center'; ctx.fillText('Waiting for position data...', W / 2, H / 2)
    }
  }, [routePoints, pathHistory, gpsLocal, alignedVioPath])

  return <canvas ref={ref} className="w-full h-full rounded-xl" />
}
