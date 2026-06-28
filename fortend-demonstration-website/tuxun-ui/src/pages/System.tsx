import { useRosBridge } from '../hooks/useRosBridge'

export default function SystemPage() {
  const { connected } = useRosBridge()
  const nodes = [
    { name: '/vins_node', pkg: 'vins', status: 'ok' },
    { name: '/route_matcher', pkg: 'hiking_route_localization', status: 'ok' },
    { name: '/route_visualizer', pkg: 'hiking_route_localization', status: 'ok' },
    { name: '/rosbridge_websocket', pkg: 'rosbridge_server', status: 'ok' },
    { name: '/web_video_server', pkg: 'web_video_server', status: 'ok' },
    { name: '/static_transform_publisher', pkg: 'tf', status: 'ok' },
    { name: '/image_transport_republish', pkg: 'image_transport', status: 'ok' },
  ]

  return (
    <div className="space-y-5 max-w-[1200px] mx-auto">
      <div className="grid grid-cols-2 gap-5">
        <div className="bg-white border border-[#E2E8F0] rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-[#0F172A] mb-4">ROS Nodes</h3>
          <div className="space-y-1">
            {nodes.map(n => (
              <div key={n.name} className="flex items-center justify-between py-2 border-b border-[#F1F5F9] text-sm">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#22C55E]" />
                  <span className="text-[#334155] font-mono text-xs">{n.name}</span>
                </div>
                <span className="text-[#94A3B8] text-xs">{n.pkg}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white border border-[#E2E8F0] rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-[#0F172A] mb-4">System Resources</h3>
          <div className="space-y-3 text-sm">
            {[
              ['ROS Master', connected ? 'Connected' : 'Disconnected', connected ? '#22C55E' : '#EF4444'],
              ['WebSocket', 'ws://localhost:9090', '#22C55E'],
              ['Frontend', 'Vite + React + TypeScript', '#2563EB'],
              ['Video Server', 'http://localhost:8080', '#22C55E'],
              ['Route CSV', '/home/kai/AI/AI_Data/Outdoor-1/route_table.csv', '#22C55E'],
            ].map(([k, v, c]) => (
              <div key={k} className="flex justify-between py-2 border-b border-[#F1F5F9]">
                <span className="text-[#64748B]">{k}</span>
                <span className="text-[#334155] font-medium text-xs">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
