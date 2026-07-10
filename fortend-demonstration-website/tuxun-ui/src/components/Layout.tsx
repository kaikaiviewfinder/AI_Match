import { ReactNode, useState } from 'react'
import { cn } from '../lib/utils'
import {
  LayoutDashboard, Route, Cpu, Activity, Database, Settings, Info,
  Menu, X, MapPin, Satellite, Thermometer
} from 'lucide-react'

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'trajectory', label: 'Trajectory Analysis', icon: Route },
  { id: 'sensors', label: 'Sensor Status', icon: Activity },
  { id: 'particles', label: 'Particle Filter', icon: Cpu },
  { id: 'system', label: 'System Config', icon: Settings },
  { id: 'about', label: 'About', icon: Info },
]

interface Props {
  active: string
  onNavigate: (id: string) => void
  connected: boolean
  state: any
  children: ReactNode
}

export default function Layout({ active, onNavigate, connected, state, children }: Props) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="flex h-screen bg-[#F8FAFC] overflow-hidden">
      {/* Sidebar */}
      <aside className={cn(
        'flex flex-col bg-white border-r border-[#E2E8F0] transition-all duration-300 flex-shrink-0',
        collapsed ? 'w-[68px]' : 'w-[240px]'
      )}>
        {/* Logo */}
        <div className={cn('flex items-center gap-3 px-5 h-16 border-b border-[#E2E8F0]', collapsed && 'justify-center px-0')}>
          <div className="w-8 h-8 rounded-lg bg-[#2563EB] flex items-center justify-center flex-shrink-0">
            <MapPin size={18} className="text-white" />
          </div>
          {!collapsed && <span className="font-bold text-[#0F172A] text-base tracking-tight">图寻 TU XUN</span>}
        </div>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="mx-3 mt-3 p-1.5 rounded-lg hover:bg-[#EFF6FF] text-[#64748B] transition-colors self-end"
        >
          {collapsed ? <Menu size={18} /> : <X size={18} />}
        </button>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                active === item.id
                  ? 'bg-[#EFF6FF] text-[#2563EB]'
                  : 'text-[#64748B] hover:bg-[#F8FAFC] hover:text-[#334155]'
              )}
            >
              <item.icon size={18} />
              {!collapsed && <span>{item.label}</span>}
            </button>
          ))}
        </nav>

        {/* Bottom status */}
        {!collapsed && (
          <div className="px-4 py-3 border-t border-[#E2E8F0] space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <div className={cn('w-2 h-2 rounded-full', connected ? 'bg-[#22C55E]' : 'bg-[#EF4444]')} />
              <span className="text-[#64748B]">ROS {connected ? 'Online' : 'Offline'}</span>
            </div>
            {state && (
              <div className="text-xs text-[#64748B] space-y-0.5">
                <div className="flex justify-between"><span>Position</span><span className="text-[#334155] font-medium">{state.s_m?.toFixed(0)}m</span></div>
                <div className="flex justify-between"><span>Confidence</span><span className="text-[#22C55E] font-medium">{((state.confidence || 0) * 100).toFixed(0)}%</span></div>
              </div>
            )}
          </div>
        )}
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-white border-b border-[#E2E8F0] flex items-center justify-between px-8 flex-shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-[#0F172A]">
              {navItems.find(i => i.id === active)?.label || 'Dashboard'}
            </h2>
          </div>
          <div className="flex items-center gap-6 text-xs text-[#64748B]">
            <div className="flex items-center gap-1.5">
              <Satellite size={14} />
              <span>GPS {state?.gps_age_s != null ? `${Math.abs(state.gps_age_s).toFixed(0)}s ago` : '--'}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Thermometer size={14} />
              <span>Alt {state?.matched_z?.toFixed(1) || '--'}m</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Activity size={14} />
              <span>{(state?.s_std_m || 0).toFixed(1)}m</span>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto p-8">
          {children}
        </main>

        {/* Footer */}
        <footer className="h-9 bg-white border-t border-[#E2E8F0] flex items-center justify-between px-8 text-[11px] text-[#64748B] flex-shrink-0">
          <span>图寻 TU XUN v1.0 · Multi-Sensor Fusion Localization</span>
          <span>{new Date().toLocaleTimeString()}</span>
        </footer>
      </div>
    </div>
  )
}
