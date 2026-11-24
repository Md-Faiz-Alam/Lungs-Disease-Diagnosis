/** Pure state for the radiograph viewer: zoom/pan, window/level (brightness/contrast), overlay. */

export interface ViewerState {
  zoom: number
  panX: number
  panY: number
  /** "level" — CSS brightness multiplier */
  brightness: number
  /** "window" — CSS contrast multiplier */
  contrast: number
  invert: boolean
  showHeatmap: boolean
  heatmapOpacity: number
}

export const MIN_ZOOM = 0.5
export const MAX_ZOOM = 8

export const initialViewer: ViewerState = {
  zoom: 1,
  panX: 0,
  panY: 0,
  brightness: 1,
  contrast: 1,
  invert: false,
  showHeatmap: true,
  heatmapOpacity: 0.45,
}

export const WL_PRESETS = {
  Default: { brightness: 1, contrast: 1 },
  Lung: { brightness: 1.1, contrast: 1.6 },
  Bone: { brightness: 0.9, contrast: 2.1 },
  'Soft tissue': { brightness: 1.25, contrast: 0.8 },
} as const
export type WLPreset = keyof typeof WL_PRESETS

export type ViewerAction =
  | { type: 'zoomBy'; factor: number; cx?: number; cy?: number }
  | { type: 'pan'; dx: number; dy: number }
  | { type: 'windowLevel'; dx: number; dy: number }
  | { type: 'preset'; name: WLPreset }
  | { type: 'toggleInvert' }
  | { type: 'toggleHeatmap' }
  | { type: 'heatmapOpacity'; value: number }
  | { type: 'reset' }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export function viewerReducer(s: ViewerState, a: ViewerAction): ViewerState {
  switch (a.type) {
    case 'zoomBy': {
      const zoom = clamp(s.zoom * a.factor, MIN_ZOOM, MAX_ZOOM)
      const k = zoom / s.zoom
      // keep the point under the cursor fixed (cx/cy relative to the viewport centre)
      const cx = a.cx ?? 0
      const cy = a.cy ?? 0
      return { ...s, zoom, panX: cx - (cx - s.panX) * k, panY: cy - (cy - s.panY) * k }
    }
    case 'pan':
      return { ...s, panX: s.panX + a.dx, panY: s.panY + a.dy }
    case 'windowLevel':
      // radiology convention: horizontal drag = window (contrast), vertical drag = level (brightness)
      return {
        ...s,
        contrast: clamp(s.contrast * (1 + a.dx / 300), 0.2, 4),
        brightness: clamp(s.brightness * (1 - a.dy / 300), 0.2, 3),
      }
    case 'preset':
      return { ...s, ...WL_PRESETS[a.name] }
    case 'toggleInvert':
      return { ...s, invert: !s.invert }
    case 'toggleHeatmap':
      return { ...s, showHeatmap: !s.showHeatmap }
    case 'heatmapOpacity':
      return { ...s, heatmapOpacity: clamp(a.value, 0, 1) }
    case 'reset':
      return { ...initialViewer, showHeatmap: s.showHeatmap, heatmapOpacity: s.heatmapOpacity }
  }
}

export const imageFilter = (s: ViewerState) =>
  `brightness(${s.brightness.toFixed(3)}) contrast(${s.contrast.toFixed(3)})${s.invert ? ' invert(1)' : ''}`
