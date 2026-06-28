import { useRosBridge } from '../hooks/useRosBridge'
import { Camera, Cpu, Satellite, Thermometer } from 'lucide-react'

const sensors = [
  { name: 'Camera', topic: '/cam0/image_raw', freq: '30 Hz', icon: Camera, status: 'ok' },
  { name: 'IMU', topic: '/imu0', freq: '100 Hz', icon: Cpu, status: 'ok' },
  { name: 'GPS', topic: '/gps', freq: '1 Hz', icon: Satellite, status: 'ok' },
  { name: 'BME280', topic: '/baro_altitude', freq: '20 Hz', icon: Thermometer, status: 'warn' },
]

export default function SensorsPage() {
  const { connected, state, gpsFixes } = useRosBridge()

  return (
    <div className="space-y-6 max-w-[1200px] mx-auto">
      <div className="grid grid-cols-4 gap-5">
        {sensors.map(s => (
          <div key={s.name} className="bg-white border border-[#E2E8F0] rounded-2xl p-5 transition-all hover:border-[#CBD5E1]">
            <div className="flex items-start justify-between mb-3">
              <s.icon size={20} className="text-[#2563EB]" />
              <span className={`w-2 h-2 rounded-full ${s.status === 'ok' ? 'bg-[#22C55E]' : 'bg-[#F59E0B]'}`} />
            </div>
            <h3 className="text-sm font-semibold text-[#0F172A]">{s.name}</h3>
            <div className="mt-2 space-y-1 text-xs text-[#64748B]">
              <div className="flex justify-between"><span>Topic</span><span className="text-[#334155] font-mono">{s.topic}</span></div>
              <div className="flex justify-between"><span>Frequency</span><span className="text-[#334155]">{s.freq}</span></div>
              <div className="flex justify-between"><span>Status</span><span className={s.status === 'ok' ? 'text-[#22C55E]' : 'text-[#F59E0B]'}>{s.status === 'ok' ? 'Active' : 'No Data'}</span></div>
            </div>
          </div>
        ))}
      </div>

      {/* Live data */}
      <div className="bg-white border border-[#E2E8F0] rounded-2xl p-6">
        <h3 className="text-sm font-semibold text-[#0F172A] mb-4">Live Sensor Data</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          {state && [
            ['VIO Step', `${state.last_vio_delta_m?.toFixed(3)}m`],
            ['Barometer Alt', state.baro_altitude_m != null ? `${state.baro_altitude_m.toFixed(1)}m` : 'No data'],
            ['GPS Age', state.gps_age_s != null ? `${Math.abs(state.gps_age_s).toFixed(1)}s ago` : '--'],
            ['Altitude Error', state.altitude_error_m != null ? `${state.altitude_error_m.toFixed(1)}m` : '--'],
            ['GPS Fixes', `${gpsFixes.length}`],
            ['ROS Status', connected ? 'Connected' : 'Disconnected'],
          ].map(([k, v]) => (
            <div key={k as string} className="flex justify-between py-2 border-b border-[#F1F5F9]">
              <span className="text-[#64748B]">{k}</span><span className="text-[#334155] font-medium">{v}</span>
            </div>
          ))}
          {!state && <div className="col-span-2 text-center text-[#94A3B8] py-8">Waiting for ROS data...</div>}
        </div>
      </div>
    </div>
  )
}
