import { useEffect, useState } from 'react'
import { useRosBridge } from '../hooks/useRosBridge'
import TrajectoryMap from '../components/TrajectoryMap'

export default function TrajectoryPage() {
  const { connected, state, gpsFixes, pathHistory, gpsToLocal } = useRosBridge()
  const [routePts, setRoutePts] = useState<any[]>([])

  useEffect(() => {
    fetch('/route_table.csv').then(r => r.text()).then(csv => {
      const pts: any[] = []
      csv.split('\n').forEach(line => {
        const c = line.split(',')
        if (c.length >= 11 && c[0] === 'point') pts.push({ x: +c[2], y: +c[3], z: +c[4], s: +c[1], lon: +c[9], lat: +c[10] })
      })
      setRoutePts(pts)
    }).catch(() => {})
  }, [])

  const s = state
  return (
    <div className="space-y-5 max-w-[1600px] mx-auto h-full flex flex-col">
      <div className="grid grid-cols-3 gap-4 text-sm">
        {[
          ['VIO Raw', '#3B82F6', '--'],
          ['Matched', '#2563EB', s ? `${(s.s_std_m).toFixed(1)}m error` : '--'],
          ['GPS Track', '#F59E0B', `${gpsFixes.length} fixes`],
        ].map(([label, color, detail]) => (
          <div key={label as string} className="bg-white border border-[#E2E8F0] rounded-xl p-4 flex items-center gap-3">
            <div className="w-3 h-3 rounded-full" style={{ background: color as string }} />
            <div><div className="font-medium text-[#0F172A]">{label}</div><div className="text-xs text-[#64748B]">{detail}</div></div>
          </div>
        ))}
      </div>
      <div className="flex-1 bg-white border border-[#E2E8F0] rounded-2xl overflow-hidden min-h-[500px]">
        <TrajectoryMap routePoints={routePts} pathHistory={pathHistory} gpsFixes={gpsFixes} gpsToLocal={gpsToLocal as any} />
      </div>
      <div className="bg-white border border-[#E2E8F0] rounded-2xl p-4 flex items-center gap-6">
        <span className="text-sm font-semibold text-[#0F172A]">Statistics</span>
        <span className="text-xs text-[#64748B]">Route length: {routePts.length > 1 ? (routePts[routePts.length-1]?.s || 0).toFixed(0) + 'm' : '--'}</span>
        <span className="text-xs text-[#64748B]">Path samples: {pathHistory.length}</span>
        <span className="text-xs text-[#64748B]">GPS fixes: {gpsFixes.length}</span>
        <span className="text-xs text-[#64748B]">ROS: {connected ? '● Online' : '○ Offline'}</span>
      </div>
    </div>
  )
}
