import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ReviewBadge, TriageBadge } from '../components/Badges'
import { ProbabilityBars } from '../components/ProbabilityBars'

describe('badges', () => {
  it('renders triage and review labels', () => {
    render(
      <>
        <TriageBadge triage="urgent" />
        <ReviewBadge status="disagreed" />
      </>,
    )
    expect(screen.getByText('Urgent')).toBeInTheDocument()
    expect(screen.getByText('Overridden')).toBeInTheDocument()
  })
})

describe('ProbabilityBars', () => {
  it('lists classes from most to least likely with accessible meters', () => {
    render(<ProbabilityBars probabilities={{ COVID19: 0.02, NORMAL: 0.1, PNEUMONIA: 0.88 }} predicted="PNEUMONIA" />)
    const meters = screen.getAllByRole('meter')
    expect(meters.map((m) => m.getAttribute('aria-label'))).toEqual([
      'Pneumonia pattern',
      'No acute finding',
      'COVID-19 pattern',
    ])
    expect(meters[0]).toHaveAttribute('aria-valuenow', '88')
  })
})
