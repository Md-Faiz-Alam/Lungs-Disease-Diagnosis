import { describe, expect, it } from 'vitest'
import type { Study } from '../api/types'
import { pct, sortWorklist, timeAgo } from '../lib/format'
import { MAX_ZOOM, imageFilter, initialViewer, viewerReducer } from '../lib/viewer'

const study = (over: Partial<Study>): Study => ({
  id: 'ST-20260101-00000000',
  owner: 'u',
  created_at: '2026-01-01T10:00:00+00:00',
  filename: 'x.png',
  source: 'image',
  prediction: 'NORMAL',
  confidence: 0.9,
  probabilities: { COVID19: 0.05, NORMAL: 0.9, PNEUMONIA: 0.05 },
  needs_review: false,
  triage: 'routine',
  warnings: [],
  dicom_metadata: {},
  image_width: 100,
  image_height: 100,
  model_arch: 'resnet18',
  model_run_id: 'r',
  review_status: 'pending',
  review_label: '',
  review_note: '',
  reviewed_at: '',
  reviewer: '',
  ...over,
})

describe('worklist ordering', () => {
  it('puts unreviewed urgent studies first, then review, routine, then signed-off', () => {
    const sorted = sortWorklist([
      study({ id: 'routine', triage: 'routine' }),
      study({ id: 'done-urgent', triage: 'urgent', review_status: 'agreed' }),
      study({ id: 'review', triage: 'review' }),
      study({ id: 'urgent-old', triage: 'urgent', created_at: '2026-01-01T09:00:00+00:00' }),
      study({ id: 'urgent-new', triage: 'urgent', created_at: '2026-01-01T11:00:00+00:00' }),
    ])
    expect(sorted.map((s) => s.id)).toEqual(['urgent-new', 'urgent-old', 'review', 'routine', 'done-urgent'])
  })
})

describe('formatting', () => {
  it('formats percentages and relative times', () => {
    expect(pct(0.9772)).toBe('97.7%')
    const now = new Date('2026-01-01T12:00:00Z')
    expect(timeAgo('2026-01-01T11:59:30Z', now)).toBe('just now')
    expect(timeAgo('2026-01-01T11:00:00Z', now)).toBe('1 h ago')
    expect(timeAgo('2025-12-30T12:00:00Z', now)).toBe('2 d ago')
  })
})

describe('viewer reducer', () => {
  it('clamps zoom and keeps the cursor point fixed', () => {
    let s = initialViewer
    for (let i = 0; i < 50; i++) s = viewerReducer(s, { type: 'zoomBy', factor: 2 })
    expect(s.zoom).toBe(MAX_ZOOM)
    const z = viewerReducer(initialViewer, { type: 'zoomBy', factor: 2, cx: 100, cy: 0 })
    expect(z.panX).toBe(-100) // point at +100px stays under the cursor after 2x zoom
  })

  it('maps drags to window (contrast) and level (brightness) and resets', () => {
    const s = viewerReducer(initialViewer, { type: 'windowLevel', dx: 150, dy: -150 })
    expect(s.contrast).toBeCloseTo(1.5)
    expect(s.brightness).toBeCloseTo(1.5)
    expect(imageFilter({ ...s, invert: true })).toContain('invert(1)')
    const reset = viewerReducer({ ...s, showHeatmap: false }, { type: 'reset' })
    expect(reset.contrast).toBe(1)
    expect(reset.showHeatmap).toBe(false) // overlay preference survives a view reset
  })
})
