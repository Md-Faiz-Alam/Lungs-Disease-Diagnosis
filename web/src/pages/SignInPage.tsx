import { Activity, FileText, Flame, Gauge, LockKeyhole, ScanLine } from 'lucide-react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthProvider'
import { Logo } from '../components/AppShell'

const FEATURES = [
  { icon: ScanLine, title: 'DICOM & image upload', text: 'Accepts hospital DICOM files; patient identifiers are stripped before anything is stored.' },
  { icon: Activity, title: 'Triage worklist', text: 'Studies ranked urgent → needs review → routine so critical cases are read first.' },
  { icon: Flame, title: 'Explainable findings', text: 'Grad-CAM heatmaps show which regions drove each calibrated AI finding.' },
  { icon: FileText, title: 'Clinician sign-off', text: 'Confirm or override every finding, with a full audit trail and PDF report.' },
  { icon: Gauge, title: 'Honest validation', text: 'Leakage-audited test set, confidence intervals and documented limitations.' },
  { icon: LockKeyhole, title: 'Enterprise security', text: 'Microsoft Entra ID sign-in, per-clinician data isolation, managed identities.' },
]

export function SignInPage() {
  const { isAuthenticated, signIn, mode, ready } = useAuth()
  if (isAuthenticated) return <Navigate to="/worklist" replace />

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <Logo dark />
        <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs text-amber-200">
          Research use only · not a medical device
        </span>
      </header>

      <section className="mx-auto grid max-w-6xl items-center gap-12 px-6 pb-16 pt-10 lg:grid-cols-2">
        <div>
          <p className="text-sm font-semibold uppercase tracking-widest text-brand-500">AI chest X-ray triage</p>
          <h1 className="mt-3 text-4xl font-semibold leading-tight tracking-tight text-white sm:text-5xl">
            Read the most urgent chest X-rays first.
          </h1>
          <p className="mt-5 max-w-xl text-lg text-slate-400">
            LungXray flags COVID-19 and pneumonia patterns on chest radiographs, prioritises your worklist, and shows
            exactly why — so clinicians stay in control of every decision.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <button type="button" className="btn-primary px-5 py-2.5 text-base" disabled={!ready} onClick={() => void signIn()}>
              {mode === 'entra' ? (
                <>
                  <MicrosoftMark /> Sign in with Microsoft
                </>
              ) : (
                'Open demo workspace'
              )}
            </button>
            <span className="text-sm text-slate-500">
              {mode === 'entra' ? 'Organisational account required' : 'No account needed · sample X-rays included'}
            </span>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-2xl">
          <div className="grid grid-cols-3 gap-3 text-center">
            <Stat value="97.7%" label="Balanced accuracy" />
            <Stat value="0.997" label="Macro AUC" />
            <Stat value="97.7%" label="Disease sensitivity" />
          </div>
          <p className="mt-4 text-xs leading-relaxed text-slate-500">
            Measured on a held-out test set after removing 45 images duplicated from training. Performance at other
            hospitals has not been established — see Model &amp; validation after signing in.
          </p>
        </div>
      </section>

      <section className="border-t border-slate-800 bg-slate-900/40">
        <div className="mx-auto grid max-w-6xl gap-6 px-6 py-14 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(({ icon: Icon, title, text }) => (
            <div key={title}>
              <Icon aria-hidden className="size-5 text-brand-500" />
              <h2 className="mt-3 font-semibold text-white">{title}</h2>
              <p className="mt-1 text-sm text-slate-400">{text}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-xl bg-slate-950/60 p-4">
      <p className="text-2xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-xs text-slate-500">{label}</p>
    </div>
  )
}

function MicrosoftMark() {
  return (
    <svg aria-hidden viewBox="0 0 21 21" className="size-4">
      <rect x="1" y="1" width="9" height="9" fill="#f25022" />
      <rect x="11" y="1" width="9" height="9" fill="#7fba00" />
      <rect x="1" y="11" width="9" height="9" fill="#00a4ef" />
      <rect x="11" y="11" width="9" height="9" fill="#ffb900" />
    </svg>
  )
}
