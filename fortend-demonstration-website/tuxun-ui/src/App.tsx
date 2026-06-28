import { useState, Suspense } from 'react'
import Layout from './components/Layout'
import { useRosBridge } from './hooks/useRosBridge'
import Dashboard from './pages/Dashboard'
import TrajectoryPage from './pages/Trajectory'
import SensorsPage from './pages/Sensors'
import ParticlesPage from './pages/Particles'
import SystemPage from './pages/System'

function AboutPage() {
  return (
    <div className="max-w-[800px] mx-auto space-y-6">
      <div className="bg-white border border-[#E2E8F0] rounded-2xl p-8 text-center">
        <h1 className="text-[32px] font-bold text-[#0F172A] tracking-tight mb-2">图寻 · TU XUN</h1>
        <p className="text-[#64748B] text-lg">Multi-Sensor Fusion Hiking Localization System</p>
        <div className="mt-6 grid grid-cols-3 gap-4 text-sm">
          {[
            ['VINS-Fusion', 'Visual-Inertial Odometry'],
            ['Particle Filter', 'Route Matching Engine'],
            ['ROS Noetic', 'Robot Operating System'],
            ['BME280', 'Barometric Altitude'],
            ['GPS/BDS', 'Satellite Positioning'],
            ['DK-2500', 'Embedded Platform'],
          ].map(([k, v]) => (
            <div key={k} className="bg-[#F8FAFC] rounded-xl p-3">
              <div className="font-semibold text-[#0F172A]">{k}</div>
              <div className="text-xs text-[#94A3B8] mt-0.5">{v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState('dashboard')
  const { connected, state } = useRosBridge()

  const pages: Record<string, JSX.Element> = {
    dashboard: <Dashboard />,
    trajectory: <TrajectoryPage />,
    sensors: <SensorsPage />,
    particles: <ParticlesPage />,
    system: <SystemPage />,
    about: <AboutPage />,
  }

  return (
    <Layout active={page} onNavigate={setPage} connected={connected} state={state}>
      <Suspense fallback={<div className="flex items-center justify-center h-64 text-[#94A3B8]">Loading...</div>}>
        {pages[page] || <Dashboard />}
      </Suspense>
    </Layout>
  )
}
