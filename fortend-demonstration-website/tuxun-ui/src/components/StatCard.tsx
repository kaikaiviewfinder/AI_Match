import { cn } from '../lib/utils'
import { type FC } from 'react'

interface Props {
  label: string; value: string; sub?: string
  icon: React.ComponentType<{ size?: number; className?: string }>; status?: 'ok' | 'warn' | 'error' | 'info'
  className?: string
}

const statusColors = {
  ok: { bg: 'bg-[#F0FDF4]', text: 'text-[#22C55E]', icon: 'text-[#22C55E]' },
  warn: { bg: 'bg-[#FFFBEB]', text: 'text-[#F59E0B]', icon: 'text-[#F59E0B]' },
  error: { bg: 'bg-[#FEF2F2]', text: 'text-[#EF4444]', icon: 'text-[#EF4444]' },
  info: { bg: 'bg-[#EFF6FF]', text: 'text-[#2563EB]', icon: 'text-[#2563EB]' },
}

export default function StatCard({ label, value, sub, icon: Icon, status = 'info', className }: Props) {
  const c = statusColors[status]
  return (
    <div className={cn('bg-white border border-[#E2E8F0] rounded-2xl p-6 transition-all duration-200 hover:border-[#CBD5E1] hover:shadow-sm', className)}>
      <div className="flex items-start justify-between mb-3">
        <span className="text-sm text-[#64748B] font-medium">{label}</span>
        <div className={cn('w-9 h-9 rounded-xl flex items-center justify-center', c.bg)}>
          <Icon size={18} className={c.icon} />
        </div>
      </div>
      <div className={cn('text-[28px] font-bold tracking-tight', c.text)}>{value}</div>
      {sub && <div className="text-xs text-[#94A3B8] mt-1">{sub}</div>}
    </div>
  )
}
