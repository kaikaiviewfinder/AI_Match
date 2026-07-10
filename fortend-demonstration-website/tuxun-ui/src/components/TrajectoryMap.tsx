import { useEffect, useRef, useMemo } from 'react'

interface RoutePoint { x: number; y: number; z: number; s: number; lon: number; lat: number }
interface PathPoint { x: number; y: number; z: number; s: number; progress: number; confidence: number }

interface Props {
  routePoints: RoutePoint[]
  pathHistory: PathPoint[]
  gpsFixes: { lat: number; lon: number }[]
  gpsToLocal: (lon: number, lat: number, olon: number, olat: number) => { x: number; y: number }
  alignedVioPath?: { x: number; y: number; z: number }[]
  /** Which layers to render. 'result'=route+matched+GPS+current,  'vio'=route+VIO only */
  variant?: 'result' | 'vio' | 'all'
  title?: string
}

export default function TrajectoryMap({ routePoints, pathHistory, gpsFixes, gpsToLocal, alignedVioPath, variant = 'result', title }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

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

    const show = {
      route: variant === 'result' || variant === 'vio' || variant === 'all',
      gps: variant === 'result' || variant === 'all',
      matched: variant === 'result' || variant === 'all',
      vio: variant === 'vio' || variant === 'all',
      curPos: variant === 'result' || variant === 'all',
    }

    // bounding box: always based on route to keep both maps at same scale
    const allPts: { x: number; y: number }[] = []
    routePoints.forEach(p => allPts.push({ x: p.x, y: p.y }))
    pathHistory.forEach(p => allPts.push({ x: p.x, y: p.y }))
    gpsLocal.forEach(p => allPts.push({ x: p.x, y: p.y }))
    if (alignedVioPath && show.vio) alignedVioPath.forEach(p => allPts.push({ x: p.x, y: p.y }))

    function w2c(px: number, py: number) {
      if (allPts.length < 2) return { cx: W / 2, cy: H / 2 }
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
      allPts.forEach(p => {
        if (Math.abs(p.x) > 1e6 || Math.abs(p.y) > 1e6) return
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y
      })
      const padX = (maxX - minX) * 0.15 || 10
      const padY = (maxY - minY) * 0.15 || 10
      const rX = (maxX - minX + 2 * padX) || 1, rY = (maxY - minY + 2 * padY) || 1
      const s = Math.min((W - 60) / rX, (H - 50) / rY)
      return { cx: 30 + (px - minX + padX) * s, cy: H - 25 - (py - minY + padY) * s }
    }

    ctx.clearRect(0, 0, W, H)

    // Grid
    ctx.strokeStyle = '#F1F5F9'; ctx.lineWidth = 0.5
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke() }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }

    // Title
    if (title) {
      ctx.fillStyle = '#0F172A'; ctx.font = 'bold 13px Inter, sans-serif'; ctx.textAlign = 'left'
      ctx.fillText(title, 12, 22)
    }

    if (routePoints.length < 2) {
      ctx.fillStyle = '#94A3B8'; ctx.font = '14px Inter, sans-serif'; ctx.textAlign = 'center'
      ctx.fillText('Loading route data...', W / 2, H / 2)
      return
    }

    // ─── 1. Known route (grey dashed) ───
    if (show.route) {
      ctx.beginPath()
      const r0 = w2c(routePoints[0].x, routePoints[0].y)
      ctx.moveTo(r0.cx, r0.cy)
      for (let i = 1; i < routePoints.length; i++) {
        const r = w2c(routePoints[i].x, routePoints[i].y); ctx.lineTo(r.cx, r.cy)
      }
      ctx.strokeStyle = '#CBD5E1'; ctx.lineWidth = 1.5
      ctx.setLineDash([5, 8]); ctx.stroke(); ctx.setLineDash([])
    }

    // ─── 2. GPS dots ───
    if (show.gps && gpsLocal.length > 0) {
      gpsLocal.forEach(gp => {
        let minD2 = Infinity, nearestX = gp.x, nearestY = gp.y
        for (const rp of routePoints) {
          const d2 = (rp.x - gp.x) ** 2 + (rp.y - gp.y) ** 2
          if (d2 < minD2) { minD2 = d2; nearestX = rp.x; nearestY = rp.y }
        }
        const gc = w2c(gp.x, gp.y); const nc = w2c(nearestX, nearestY)
        if (Math.sqrt(minD2) < 200) {
          ctx.beginPath(); ctx.moveTo(gc.cx, gc.cy); ctx.lineTo(nc.cx, nc.cy)
          ctx.strokeStyle = 'rgba(245,158,11,0.12)'; ctx.lineWidth = 0.5; ctx.stroke()
          ctx.beginPath(); ctx.arc(gc.cx, gc.cy, 4.5, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(245,158,11,0.25)'; ctx.fill()
          ctx.beginPath(); ctx.arc(gc.cx, gc.cy, 2.8, 0, Math.PI * 2)
          ctx.fillStyle = '#F59E0B'; ctx.fill()
          ctx.strokeStyle = 'rgba(245,158,11,0.4)'; ctx.lineWidth = 1; ctx.stroke()
        }
      })
    }

    // ─── 3. VIO raw trajectory (green, thick band) ───
    if (show.vio && alignedVioPath && alignedVioPath.length > 1) {
      // wide glow band
      ctx.beginPath()
      const v0 = w2c(alignedVioPath[0].x, alignedVioPath[0].y); ctx.moveTo(v0.cx, v0.cy)
      for (let i = 1; i < alignedVioPath.length; i++) {
        const v = w2c(alignedVioPath[i].x, alignedVioPath[i].y); ctx.lineTo(v.cx, v.cy)
      }
      ctx.strokeStyle = 'rgba(34,197,94,0.15)'; ctx.lineWidth = 7; ctx.stroke()
      // core line on top
      ctx.beginPath()
      const v00 = w2c(alignedVioPath[0].x, alignedVioPath[0].y); ctx.moveTo(v00.cx, v00.cy)
      for (let i = 1; i < alignedVioPath.length; i++) {
        const vv = w2c(alignedVioPath[i].x, alignedVioPath[i].y); ctx.lineTo(vv.cx, vv.cy)
      }
      ctx.strokeStyle = '#22C55E'; ctx.lineWidth = 2.2; ctx.stroke()
    }

    // ─── 4. Full matched path (solid blue, prominent) ───
    if (show.matched && pathHistory.length > 1) {
      ctx.beginPath()
      const ph0 = w2c(pathHistory[0].x, pathHistory[0].y); ctx.moveTo(ph0.cx, ph0.cy)
      for (let i = 1; i < pathHistory.length; i++) {
        const ph = w2c(pathHistory[i].x, pathHistory[i].y); ctx.lineTo(ph.cx, ph.cy)
      }
      ctx.strokeStyle = 'rgba(37,99,235,0.65)'; ctx.lineWidth = 2.5; ctx.stroke()
    }
    // 4b. Recent 60 pts — brighter
    if (show.matched) {
      const st = Math.max(0, pathHistory.length - 60)
      if (st < pathHistory.length - 1) {
        ctx.beginPath()
        const rs = w2c(pathHistory[st].x, pathHistory[st].y); ctx.moveTo(rs.cx, rs.cy)
        for (let i = st + 1; i < pathHistory.length; i++) {
          const rp = w2c(pathHistory[i].x, pathHistory[i].y); ctx.lineTo(rp.cx, rp.cy)
        }
        ctx.strokeStyle = '#2563EB'; ctx.lineWidth = 3.5; ctx.stroke()
        ctx.strokeStyle = 'rgba(37,99,235,0.12)'; ctx.lineWidth = 9; ctx.stroke()
      }
    }

    // ─── 5. Current position pulse (only on result view) ───
    if (show.curPos && pathHistory.length > 0) {
      const last = pathHistory[pathHistory.length - 1]
      const lp = w2c(last.x, last.y)
      const t = Date.now() / 600; const pulse = 1 + 0.25 * Math.sin(t)
      ctx.beginPath(); ctx.arc(lp.cx, lp.cy, 14 * pulse, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(37,99,235,${0.08 * pulse})`; ctx.fill()
      ctx.beginPath(); ctx.arc(lp.cx, lp.cy, 5, 0, Math.PI * 2)
      ctx.fillStyle = '#2563EB'; ctx.fill()
      ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 2; ctx.stroke()
    }

    // ─── 6. Info text ───
    if (show.vio && alignedVioPath && alignedVioPath.length > 1) {
      ctx.fillStyle = 'rgba(15,23,42,0.7)'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left'
      ctx.fillText(`VIO: ${alignedVioPath.length} pts`, 10, H - 10)
    }
    if (show.curPos && pathHistory.length > 0) {
      const last = pathHistory[pathHistory.length - 1]
      const gpsCount = gpsLocal.length
      ctx.fillStyle = 'rgba(15,23,42,0.7)'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left'
      ctx.fillText(`Matched: s=${last.s.toFixed(0)}m · GPS:${gpsCount} · Conf:${(last.confidence*100).toFixed(0)}%`, 10, H - 10)
    }
    if (show.route && !show.vio && !show.curPos && routePoints.length < 2) { /* fall through */ }
    if (pathHistory.length < 2 && gpsLocal.length === 0 && !(alignedVioPath && alignedVioPath.length > 0)) {
      ctx.fillStyle = '#94A3B8'; ctx.font = '14px Inter, sans-serif'; ctx.textAlign = 'center'
      ctx.fillText('Waiting for position data...', W / 2, H / 2)
    }
  }, [routePoints, pathHistory, gpsLocal, alignedVioPath, variant, title])

  return <canvas ref={ref} className="w-full h-full rounded-xl" />
}
