import { useEffect, useRef, useState, useCallback } from 'react'
import * as ROSLIB from 'roslib'

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

export function useRosBridge() {
  const [connected, setConnected] = useState(false)
  const [state, setState] = useState<RouteState | null>(null)
  const [gpsFixes, setGpsFixes] = useState<GpsFix[]>([])
  const [pathHistory, setPathHistory] = useState<PathPoint[]>([])
  const [alignedVioPath, setAlignedVioPath] = useState<AlignedVioPoint[]>([])

  const rosRef = useRef<ROSLIB.Ros | null>(null)
  const vioRawRef = useRef<VioPoint[]>([])     // raw VIO in world frame
  const vioOffsetRef = useRef<{ x: number; y: number; z: number } | null>(null)
  const firstMatchRef = useRef<{ x: number; y: number; z: number } | null>(null)

  useEffect(() => {
    const ros = new ROSLIB.Ros({ url: 'ws://localhost:9090' })
    rosRef.current = ros

    ros.on('connection', () => setConnected(true))
    ros.on('close', () => setConnected(false))
    ros.on('error', () => setConnected(false))

    // Route matcher state
    const stateSub = new ROSLIB.Topic({
      ros, name: '/route_matcher/state', messageType: 'std_msgs/String'
    })
    stateSub.subscribe((msg: { data: string }) => {
      try {
        const d = JSON.parse(msg.data) as RouteState
        setState(d)
        setPathHistory(prev => {
          const next = [...prev, {
            x: d.matched_x, y: d.matched_y, z: d.matched_z,
            s: d.s_m, progress: d.progress, confidence: d.confidence
          }]
          return next.length > 600 ? next.slice(-600) : next
        })

        // Record first matched position for VIO alignment
        if (!firstMatchRef.current && d.matched_x !== undefined) {
          firstMatchRef.current = { x: d.matched_x, y: d.matched_y, z: d.matched_z }
        }

        // Align raw VIO to route frame using VIO delta direction
        // We use the matched path's local movement to rotate VIO steps
        const delta = d.last_vio_delta_m || 0
        if (vioRawRef.current.length > 0 && firstMatchRef.current) {
          const lastVio = vioRawRef.current[vioRawRef.current.length - 1]
          const prevLen = vioRawRef.current.length
          if (prevLen >= 2) {
            const prevVio = vioRawRef.current[prevLen - 2]
            const dvx = lastVio.x - prevVio.x
            const dvy = lastVio.y - prevVio.y
            const dvz = lastVio.z - prevVio.z
            const dvDist = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) || 1
            // Scale VIO step to match VIO delta distance, but keep direction
            const scale = delta / dvDist
            setAlignedVioPath(prev => {
              const lastAligned = prev.length > 0 ? prev[prev.length - 1] : firstMatchRef.current!
              const next = [...prev, {
                x: lastAligned.x + dvx * scale,
                y: lastAligned.y + dvy * scale,
                z: lastAligned.z + dvz * scale
              }]
              return next.length > 800 ? next.slice(-800) : next
            })
          }
        }
      } catch {}
    })

    // GPS
    const gpsSub = new ROSLIB.Topic({
      ros, name: '/gps', messageType: 'sensor_msgs/NavSatFix'
    })
    gpsSub.subscribe((msg: any) => {
      if (msg?.status?.status >= 0) {
        setGpsFixes(prev => {
          const next = [...prev, { lat: msg.latitude, lon: msg.longitude, alt: msg.altitude, time: Date.now() }]
          return next.length > 120 ? next.slice(-120) : next
        })
      }
    })

    // VINS raw odometry - record raw positions in world frame
    const vioSub = new ROSLIB.Topic({
      ros, name: '/vins_estimator/odometry', messageType: 'nav_msgs/Odometry'
    })
    vioSub.subscribe((msg: any) => {
      const p = msg?.pose?.pose?.position
      if (p && typeof p.x === 'number') {
        vioRawRef.current.push({ x: p.x, y: p.y, z: p.z })
        if (vioRawRef.current.length > 800) vioRawRef.current = vioRawRef.current.slice(-800)

        // If no offset yet and we have both VIO and matched data, compute offset
        if (!vioOffsetRef.current && firstMatchRef.current && vioRawRef.current.length >= 2) {
          const firstVio = vioRawRef.current[0]
          vioOffsetRef.current = {
            x: firstMatchRef.current.x - firstVio.x,
            y: firstMatchRef.current.y - firstVio.y,
            z: firstMatchRef.current.z - firstVio.z
          }
        }

        // If we have offset, publish aligned VIO
        if (vioOffsetRef.current) {
          const aligned = {
            x: p.x + vioOffsetRef.current.x,
            y: p.y + vioOffsetRef.current.y,
            z: p.z + vioOffsetRef.current.z
          }
          setAlignedVioPath(prev => {
            const next = [...prev, aligned]
            return next.length > 800 ? next.slice(-800) : next
          })
        }
      }
    })

    return () => { ros.close() }
  }, [])

  const gpsToLocal = useCallback((lon: number, lat: number, originLon: number, originLat: number) => {
    const R = 6378137
    const lat0r = originLat * Math.PI / 180
    const dlon = (lon - originLon) * Math.PI / 180
    const dlat = (lat - originLat) * Math.PI / 180
    return { x: R * dlon * Math.cos(lat0r), y: R * dlat }
  }, [])

  return { connected, state, gpsFixes, pathHistory, alignedVioPath, gpsToLocal }
}
