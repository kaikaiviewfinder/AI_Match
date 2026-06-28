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

export type GpsFix = {
  lat: number; lon: number; alt: number; time: number
}

export type PathPoint = {
  x: number; y: number; z: number; s: number; progress: number; confidence: number
}

export function useRosBridge() {
  const [connected, setConnected] = useState(false)
  const [state, setState] = useState<RouteState | null>(null)
  const [gpsFixes, setGpsFixes] = useState<GpsFix[]>([])
  const [pathHistory, setPathHistory] = useState<PathPoint[]>([])
  const rosRef = useRef<ROSLIB.Ros | null>(null)

  useEffect(() => {
    const ros = new ROSLIB.Ros({ url: 'ws://localhost:9090' })
    rosRef.current = ros

    ros.on('connection', () => setConnected(true))
    ros.on('close', () => setConnected(false))
    ros.on('error', () => setConnected(false))

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
      } catch {}
    })

    const gpsSub = new ROSLIB.Topic({
      ros, name: '/gps', messageType: 'sensor_msgs/NavSatFix'
    })
    gpsSub.subscribe((msg: any) => {
      if (msg?.status?.status >= 0) {
        setGpsFixes(prev => {
          const next = [...prev, {
            lat: msg.latitude, lon: msg.longitude,
            alt: msg.altitude, time: Date.now()
          }]
          return next.length > 120 ? next.slice(-120) : next
        })
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

  return { connected, state, gpsFixes, pathHistory, gpsToLocal }
}
