import { lazy } from 'react'
import { Navigate, Outlet, Route, Routes } from 'react-router-dom'
import { useAuth } from './auth/AuthProvider'
import { AppShell } from './components/AppShell'
import { SignInPage } from './pages/SignInPage'

// Route-level code splitting: each page is downloaded only when first visited.
const WorklistPage = lazy(() => import('./pages/WorklistPage').then((m) => ({ default: m.WorklistPage })))
const NewStudyPage = lazy(() => import('./pages/NewStudyPage').then((m) => ({ default: m.NewStudyPage })))
const StudyPage = lazy(() => import('./pages/StudyPage').then((m) => ({ default: m.StudyPage })))
const ModelPage = lazy(() => import('./pages/ModelPage').then((m) => ({ default: m.ModelPage })))
const AuditPage = lazy(() => import('./pages/AuditPage').then((m) => ({ default: m.AuditPage })))

function RequireAuth() {
  const { ready, isAuthenticated } = useAuth()
  if (!ready) return <p className="p-8 text-sm text-slate-500">Signing in…</p>
  if (!isAuthenticated) return <Navigate to="/" replace />
  return <Outlet />
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<SignInPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppShell />}>
          <Route path="/worklist" element={<WorklistPage />} />
          <Route path="/studies/new" element={<NewStudyPage />} />
          <Route path="/studies/:id" element={<StudyPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/audit" element={<AuditPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
