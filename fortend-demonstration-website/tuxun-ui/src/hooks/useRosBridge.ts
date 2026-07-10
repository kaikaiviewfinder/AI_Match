import { useSyncExternalStore, useCallback } from 'react'

export type RouteState = {
  s_m: number; route_length_m: number; progress: number
  segment_id: number; distance_to_end_m: number
  matched_x: number; matched_y: number; matched_z: number
  matched_lon: number; matched_lat: number; yaw_rad: number
  s_std_m: number; confidence: number
  last_vio_delta_m: number
  baro_altitude_m: number | null; altitude_error_m: number | null
  gps_age_s: number | null
}

export type GpsFix = { lat: number; lon: number; alt: number; time: number }
export type PathPoint = { x: number; y: number; z: number; s: number; progress: number; confidence: number }
export type VioPoint = { x: number; y: number; z: number }
export type AlignedVioPoint = { x: number; y: number; z: number }

const ROSBRIDGE_URL = 'ws://localhost:9090'

// ─────────────────────────────────────────────────────────────
// Module-level singleton: ONE WebSocket + ONE accumulation state,
// shared across every component that calls useRosBridge(). This keeps
// the trajectory intact when navigating between pages and avoids
// opening a new rosbridge connection per component.
// ─────────────────────────────────────────────────────────────

// Working (mutable) state
let wConnected = false
let wState: RouteState | null = null
const wGps: GpsFix[] = []
const wPath: PathPoint[] = []
let wVio: AlignedVioPoint[] = []

// VIO alignment: estimate the VINS-world→route rotation+scale ONCE, then FREEZE it.
// After freezing, new points are appended by integrating raw VIO deltas, so the
// already-drawn portion of the green line never moves.
const vioRaw: VioPoint[] = []
const pairs: { vx: number; vy: number; mx: number; my: number }[] = []
let firstMatch: { x: number; y: number; z: number } | null = null
let lastRaw: VioPoint | null = null
let align: { theta: number; scale: number } | null = null   // null until frozen

function estimateAlign(): { theta: number; scale: number } | null {
  const n = pairs.length
  if (n < 30) return null
  // require enough travel for a stable heading
  const first = pairs[0], last = pairs[n - 1]
  const matchSpan = Math.hypot(last.mx - first.mx, last.my - first.my)
  if (matchSpan < 6) return null
  let mvx = 0, mvy = 0, mmx = 0, mmy = 0
  for (const p of pairs) { mvx += p.vx; mvy += p.vy; mmx += p.mx; mmy += p.my }
  mvx /= n; mvy /= n; mmx /= n; mmy /= n
  let Sc = 0, Ss = 0, varP = 0
  for (const p of pairs) {
    const px = p.vx - mvx, py = p.vy - mvy
    const qx = p.mx - mmx, qy = p.my - mmy
    Sc += px * qx + py * qy
    Ss += px * qy - py * qx
    varP += px * px + py * py
  }
  if (varP < 1) return null
  const theta = Math.atan2(Ss, Sc)
  let scale = Math.sqrt(Sc * Sc + Ss * Ss) / varP
  if (!isFinite(scale) || scale <= 0) scale = 1
  if (scale < 0.3) scale = 0.3; else if (scale > 3) scale = 3
  return { theta, scale }
}

// Build the historical green line ONCE at freeze time by integrating raw deltas
// from the anchor (first matched position). Called only when align is set.
function rebuildVioFromRaw() {
  if (!align || !firstMatch || vioRaw.length < 1) return
  const cos = Math.cos(align.theta), sin = Math.sin(align.theta)
  wVio = [{ x: firstMatch.x, y: firstMatch.y, z: firstMatch.z }]
  for (let i = 1; i < vioRaw.length; i++) {
    const dx = (vioRaw[i].x - vioRaw[i - 1].x) * align.scale
    const dy = (vioRaw[i].y - vioRaw[i - 1].y) * align.scale
    const dz = (vioRaw[i].z - vioRaw[i - 1].z)
    const prev = wVio[wVio.length - 1]
    wVio.push({ x: prev.x + (cos * dx - sin * dy), y: prev.y + (sin * dx + cos * dy), z: prev.z + dz })
  }
  lastRaw = vioRaw[vioRaw.length - 1]
}

// ── React snapshot + throttled commit (~15 fps) ──
let snap = {
  connected: false, state: null as RouteState | null,
  gpsFixes: [] as GpsFix[], pathHistory: [] as PathPoint[], alignedVioPath: [] as AlignedVioPoint[],
}
const listeners = new Set<() => void>()
let commitTimer: ReturnType<typeof setTimeout> | null = null
function commitSoon() {
  if (commitTimer) return
  commitTimer = setTimeout(() => {
    commitTimer = null
    snap = {
      connected: wConnected, state: wState,
      gpsFixes: wGps.slice(), pathHistory: wPath.slice(), alignedVioPath: wVio.slice(),
    }
    listeners.forEach(l => l())
  }, 66)
}

function onState(d: RouteState) {
  wState = d
  wPath.push({ x: d.matched_x, y: d.matched_y, z: d.matched_z, s: d.s_m, progress: d.progress, confidence: d.confidence })
  if (wPath.length > 600) wPath.shift()
  if (!firstMatch && d.matched_x !== undefined) firstMatch = { x: d.matched_x, y: d.matched_y, z: d.matched_z }
  // Collect calibration correspondences only until the transform is frozen
  if (!align && vioRaw.length > 0 && d.matched_x !== undefined) {
    const v = vioRaw[vioRaw.length - 1]
    pairs.push({ vx: v.x, vy: v.y, mx: d.matched_x, my: d.matched_y })
    if (pairs.length > 800) pairs.shift()
    const est = estimateAlign()
    if (est) { align = est; rebuildVioFromRaw() }   // freeze + build history once
  }
  commitSoon()
}

function onGps(msg: any) {
  if (msg?.status?.status >= 0) {
    wGps.push({ lat: msg.latitude, lon: msg.longitude, alt: msg.altitude, time: Date.now() })
    if (wGps.length > 120) wGps.shift()
    commitSoon()
  }
}

function onOdom(msg: any) {
  const p = msg?.pose?.pose?.position
  if (!p || typeof p.x !== 'number') return
  vioRaw.push({ x: p.x, y: p.y, z: p.z })
  if (vioRaw.length > 2000) vioRaw.shift()
  // Once frozen, append incrementally so past points stay fixed
  if (align && lastRaw) {
    const cos = Math.cos(align.theta), sin = Math.sin(align.theta)
    const dx = (p.x - lastRaw.x) * align.scale
    const dy = (p.y - lastRaw.y) * align.scale
    const dz = (p.z - lastRaw.z)
    const prev = wVio.length ? wVio[wVio.length - 1] : (firstMatch || { x: 0, y: 0, z: 0 })
    wVio.push({ x: prev.x + (cos * dx - sin * dy), y: prev.y + (sin * dx + cos * dy), z: prev.z + dz })
    if (wVio.length > 1200) wVio.shift()
    commitSoon()
  }
  lastRaw = { x: p.x, y: p.y, z: p.z }
}

let started = false
function start() {
  if (started) return
  started = true
  let ws: WebSocket | null = null
  const connect = () => {
    ws = new WebSocket(ROSBRIDGE_URL)
    ws.onopen = () => {
      wConnected = true; commitSoon()
      const subs = [
        { topic: '/route_matcher/state', type: 'std_msgs/String' },
        { topic: '/gps', type: 'sensor_msgs/NavSatFix' },
        { topic: '/vins_estimator/odometry', type: 'nav_msgs/Odometry' },
      ]
      for (const s of subs) ws!.send(JSON.stringify({ op: 'subscribe', topic: s.topic, type: s.type }))
    }
    ws.onmessage = (ev) => {
      let parsed: any
      try { parsed = JSON.parse(ev.data) } catch { return }
      if (parsed.op !== 'publish') return
      if (parsed.topic === '/route_matcher/state') { try { onState(JSON.parse(parsed.msg.data)) } catch { /* */ } }
      else if (parsed.topic === '/gps') onGps(parsed.msg)
      else if (parsed.topic === '/vins_estimator/odometry') onOdom(parsed.msg)
    }
    ws.onclose = () => { wConnected = false; commitSoon(); setTimeout(connect, 1500) }
    ws.onerror = () => { wConnected = false; commitSoon(); ws?.close() }
  }
  connect()
}

function subscribe(cb: () => void) {
  listeners.add(cb)
  start()
  return () => { listeners.delete(cb) }
}
function getSnapshot() { return snap }

export function useRosBridge() {
  const s = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  const gpsToLocal = useCallback((lon: number, lat: number, originLon: number, originLat: number) => {
    const R = 6378137
    const lat0r = originLat * Math.PI / 180
    const dlon = (lon - originLon) * Math.PI / 180
    const dlat = (lat - originLat) * Math.PI / 180
    return { x: R * dlon * Math.cos(lat0r), y: R * dlat }
  }, [])
  return { ...s, gpsToLocal }
}
