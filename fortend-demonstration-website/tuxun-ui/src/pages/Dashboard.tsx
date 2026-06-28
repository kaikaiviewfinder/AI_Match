import { useEffect, useState } from 'react'
import StatCard from '../components/StatCard'
import TrajectoryMap from '../components/TrajectoryMap'
import { useRosBridge } from '../hooks/useRosBridge'
import type { RouteState } from '../hooks/useRosBridge'
import { formatCoord, formatDist, formatPercent } from '../lib/utils'
import { MapPin, Crosshair, Satellite, ShieldCheck, Navigation, Gauge, Mountain, Clock } from 'lucide-react'

interface RoutePt { x: number; y: number; z: number; s: number; lon: number; lat: number }

export default function Dashboard() {
  const { connected, state, gpsFixes, pathHistory, gpsToLocal } = useRosBridge()
  const [routePts, setRoutePts] = useState<RoutePt[]>([])

  useEffect(() => {
    fetch('/route_table.csv')
      .then(r => r.text())
      .then(csv => {
        const pts: RoutePt[] = []
        csv.split('\n').forEach(line => {
          const c = line.split(',')
          if (c.length >= 11 && c[0] === 'point') {
            pts.push({ x: +c[2], y: +c[3], z: +c[4], s: +c[1], lon: +c[9], lat: +c[10] })
          }
        })
        setRoutePts(pts)
      }).catch(() => {})
  }, [])

  const s = state as RouteState | null
  return (
    <div className="space-y-6 max-w-[1600px] mx-auto">
      {/* Stat cards row */}
      <div className="grid grid-cols-4 gap-5">
        <StatCard
          label="Current Position" icon={MapPin} status="info"
          value={s ? formatDist(s.s_m) : '--'}
          sub={s ? `Segment ${s.segment_id} · ${formatPercent(s.progress)} of route` : 'Waiting for data...'}
        />
        <StatCard
          label="Localization Error" icon={Crosshair}
          status={s ? (s.s_std_m < 3 ? 'ok' : s.s_std_m < 8 ? 'warn' : 'error') : 'info'}
          value={s ? `±${s.s_std_m.toFixed(1)}m` : '--'}
          sub={s ? '1-sigma standard deviation' : ''}
        />
        <StatCard
          label="GPS Status" icon={Satellite}
          status={s?.gps_age_s != null ? (Math.abs(s.gps_age_s!) < 5 ? 'ok' : 'warn') : 'info'}
          value={s?.gps_age_s != null ? `${Math.abs(s.gps_age_s!).toFixed(0)}s ago` : 'No fix'}
          sub={s ? `${gpsFixes.length} fixes received` : ''}
        />
        <StatCard
          label="System Confidence" icon={ShieldCheck}
          status={s ? (s.confidence > 0.8 ? 'ok' : s.confidence > 0.5 ? 'warn' : 'error') : 'info'}
          value={s ? `${(s.confidence * 100).toFixed(0)}%` : '--'}
          sub={s ? `${pathHistory.length} position samples` : ''}
        />
      </div>

      {/* Main content: Video + Map + Altitude */}
      <div className="grid grid-cols-12 gap-5">
        {/* Video panel */}
        <div className="col-span-4 bg-white border border-[#E2E8F0] rounded-2xl overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-[#E2E8F0]">
            <span className="text-sm font-semibold text-[#0F172A]">Live Camera Feed</span>
            <span className="text-xs text-[#94A3B8]">VINS Feature Tracking</span>
          </div>
          <div className="aspect-video bg-black">
            <img
              src="http://localhost:8080/stream?topic=/cam0/image_raw&type=mjpeg&quality=60"
              className="w-full h-full object-contain"
              alt="Camera feed"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
            />
          </div>
          <div className="px-5 py-2.5 border-t border-[#E2E8F0] flex items-center justify-between text-xs text-[#64748B]">
            <span>/cam0/image_raw · 1280×720</span>
            <span className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${connected ? 'bg-[#22C55E]' : 'bg-[#EF4444]'}`} />
              {connected ? 'Streaming' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Map panel */}
        <div className="col-span-5 bg-white border border-[#E2E8F0] rounded-2xl overflow-hidden flex flex-col">
          <div className="flex items-center justify-between px-5 py-3 border-b border-[#E2E8F0]">
            <span className="text-sm font-semibold text-[#0F172A]">Trajectory Map</span>
            <span className="text-xs text-[#94A3B8]">
              {routePts.length > 0 ? `${formatDist(routePts[routePts.length-1]?.s || 0)} route` : 'Loading...'}
            </span>
          </div>
          <div className="flex-1 relative min-h-[360px]">
            <TrajectoryMap routePoints={routePts} pathHistory={pathHistory} gpsFixes={gpsFixes} gpsToLocal={gpsToLocal as any} />
          </div>
          <div className="px-5 py-2 border-t border-[#E2E8F0] flex gap-6 text-xs text-[#64748B]">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-[#CBD5E1] inline-block" /> Route</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-[#2563EB] inline-block" /> Matched</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-[#F59E0B] inline-block" /> GPS</span>
          </div>
        </div>

        {/* Altitude + Info panel */}
        <div className="col-span-3 space-y-5">
          {/* Altitude mini chart */}
          <div className="bg-white border border-[#E2E8F0] rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Mountain size={16} className="text-[#2563EB]" />
              <span className="text-sm font-semibold text-[#0F172A]">Altitude</span>
            </div>
            <div className="text-[28px] font-bold text-[#0F172A] tracking-tight">
              {s?.matched_z?.toFixed(1) || '--'}<span className="text-base font-normal text-[#94A3B8] ml-1">m</span>
            </div>
            <div className="mt-4 space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-[#64748B]">Route alt range</span>
                <span className="text-[#334155] font-medium">
                  {routePts.length > 1 ? `${Math.min(...routePts.map(p => p.z)).toFixed(0)} - ${Math.max(...routePts.map(p => p.z)).toFixed(0)}m` : '--'}
                </span>
              </div>
              <div className="w-full h-1.5 bg-[#F1F5F9] rounded-full overflow-hidden">
                {routePts.length > 1 && s && (() => {
                  const minZ = Math.min(...routePts.map(p => p.z))
                  const maxZ = Math.max(...routePts.map(p => p.z))
                  const pct = ((s.matched_z - minZ) / (maxZ - minZ)) * 100
                  return <div className="h-full bg-[#2563EB] rounded-full transition-all" style={{ width: `${Math.max(2, Math.min(98, pct))}%` }} />
                })()}
              </div>
            </div>
          </div>

          {/* Position details */}
          <div className="bg-white border border-[#E2E8F0] rounded-2xl p-5 space-y-3">
            <div className="flex items-center gap-2 mb-2">
              <Navigation size={16} className="text-[#2563EB]" />
              <span className="text-sm font-semibold text-[#0F172A]">Position</span>
            </div>
            {[
              ['Latitude', s ? formatCoord(s.matched_lat) + '°N' : '--'],
              ['Longitude', s ? formatCoord(s.matched_lon) + '°E' : '--'],
              ['Route Progress', s ? formatDist(s.s_m) + ' / ' + formatDist(s.route_length_m) : '--'],
              ['Distance to End', s ? formatDist(s.distance_to_end_m) : '--'],
              ['Heading', s ? `${(s.yaw_rad * 180 / Math.PI).toFixed(1)}°` : '--'],
            ].map(([label, value]) => (
              <div key={label as string} className="flex justify-between text-sm">
                <span className="text-[#64748B]">{label}</span>
                <span className="text-[#334155] font-medium">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="bg-white border border-[#E2E8F0] rounded-2xl p-5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-semibold text-[#0F172A] flex items-center gap-2">
            <Gauge size={16} className="text-[#2563EB]" /> Route Progress
          </span>
          <span className="text-sm font-bold text-[#2563EB]">{s ? formatPercent(s.progress) : '0%'}</span>
        </div>
        <div className="w-full h-2.5 bg-[#F1F5F9] rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-[#2563EB] to-[#06B6D4] rounded-full transition-all duration-500"
               style={{ width: s ? `${Math.max(1, s.progress * 100)}%` : '0%' }} />
        </div>
        <div className="flex justify-between mt-2 text-xs text-[#94A3B8]">
          <span>0m</span>
          <span>{s ? formatDist(s.route_length_m) : '--'}</span>
        </div>
      </div>
    </div>
  )
}
