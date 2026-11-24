import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { ClipboardList, FileClock, Gauge, LogOut, PlusCircle, Stethoscope } from 'lucide-react'
import { Suspense } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthProvider'

const NAV = [
  { to: '/worklist', label: 'Worklist', icon: ClipboardList },
  { to: '/studies/new', label: 'New study', icon: PlusCircle },
  { to: '/model', label: 'Model & validation', icon: Gauge },
  { to: '/audit', label: 'Audit log', icon: FileClock },
]

export function Logo({ dark = false }: { dark?: boolean }) {
  return (
    <span className="flex items-center gap-2">
      <span className="grid size-8 place-items-center rounded-lg bg-brand-600 text-white">
        <Stethoscope aria-hidden className="size-4.5" />
      </span>
      <span className={clsx('text-base font-semibold tracking-tight', dark ? 'text-white' : 'text-slate-900')}>
        LungXray
      </span>
    </span>
  )
}

export function AppShell() {
  const { user, signOut, mode } = useAuth()
  const queryClient = useQueryClient()
  const handleSignOut = () => {
    queryClient.clear() // never show the previous user's data after signing out
    void signOut()
  }
  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-slate-800 bg-slate-900 px-3 py-5 md:flex">
        <div className="px-2">
          <Logo dark />
        </div>
        <nav className="mt-8 flex flex-col gap-1" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to !== '/worklist'}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors',
                  isActive ? 'bg-slate-800 text-white' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200',
                )
              }
            >
              <Icon aria-hidden className="size-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto rounded-lg border border-slate-800 p-3">
          <p className="truncate text-sm font-medium text-slate-200">{user?.name}</p>
          <p className="truncate text-xs text-slate-500">{mode === 'demo' ? 'Demo mode' : user?.email}</p>
          <button type="button" onClick={handleSignOut} className="mt-3 flex items-center gap-1.5 text-xs text-slate-400 hover:text-white">
            <LogOut aria-hidden className="size-3.5" /> Sign out
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-center text-xs text-amber-900">
          <strong>Research use only.</strong> Not a medical device. AI findings must be confirmed by a qualified
          radiologist.
        </div>
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 md:hidden">
          <Logo />
          <nav className="flex gap-1" aria-label="Main mobile">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} aria-label={label} className="rounded-md p-2 text-slate-600 hover:bg-slate-100">
                <Icon aria-hidden className="size-4.5" />
              </NavLink>
            ))}
          </nav>
        </header>
        <main className="flex-1 px-4 py-6 md:px-8">
          <Suspense fallback={<p className="text-sm text-slate-500">Loading…</p>}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  )
}
