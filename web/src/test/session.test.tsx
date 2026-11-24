import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { getDemoSessionId, resetDemoSession } from '../lib/demoSession'

describe('demo workspace id', () => {
  it('is a stable UUID v4 until reset', () => {
    const id = getDemoSessionId()
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
    expect(getDemoSessionId()).toBe(id)
    resetDemoSession()
    expect(getDemoSessionId()).not.toBe(id)
  })
})

describe('ErrorBoundary', () => {
  it('shows a recovery screen instead of a blank page', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const Boom = () => {
      throw new Error('boom')
    }
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>,
    )
    expect(screen.getByRole('alert')).toHaveTextContent('Something went wrong')
    spy.mockRestore()
  })
})
