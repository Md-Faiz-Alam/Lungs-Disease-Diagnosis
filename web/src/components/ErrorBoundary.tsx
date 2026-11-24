import { AlertOctagon } from 'lucide-react'
import { Component, type ErrorInfo, type ReactNode } from 'react'

interface State {
  error: Error | null
}

/** Last-resort crash screen: a rendering bug shows a recoverable message instead of a blank page. */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Unhandled UI error', error, info.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div role="alert" className="grid min-h-screen place-items-center bg-slate-50 p-6">
        <div className="card max-w-md p-6 text-center">
          <AlertOctagon aria-hidden className="mx-auto size-10 text-urgent" />
          <h1 className="mt-3 text-lg font-semibold">Something went wrong</h1>
          <p className="mt-1 text-sm text-slate-600">
            The page hit an unexpected error. Your studies are safe; reload to continue.
          </p>
          <button type="button" className="btn-primary mt-5" onClick={() => window.location.assign('/worklist')}>
            Reload workspace
          </button>
        </div>
      </div>
    )
  }
}
