import clsx from 'clsx'
import { Contrast, Flame, Maximize, Minus, Plus, RotateCcw } from 'lucide-react'
import { useReducer, useRef, type PointerEvent, type ReactNode, type WheelEvent } from 'react'
import { useBlobUrl } from '../lib/useBlobUrl'
import { WL_PRESETS, imageFilter, initialViewer, viewerReducer, type WLPreset } from '../lib/viewer'

interface Props {
  studyId: string
  alt: string
}

export function Viewer({ studyId, alt }: Props) {
  const image = useBlobUrl(`/api/studies/${studyId}/image.png`)
  const heatmap = useBlobUrl(`/api/studies/${studyId}/gradcam.png`)
  const [s, dispatch] = useReducer(viewerReducer, initialViewer)
  const drag = useRef<{ x: number; y: number; mode: 'pan' | 'wl' } | null>(null)
  const frame = useRef<HTMLDivElement>(null)

  const onWheel = (e: WheelEvent) => {
    const rect = frame.current?.getBoundingClientRect()
    const cx = rect ? e.clientX - rect.left - rect.width / 2 : 0
    const cy = rect ? e.clientY - rect.top - rect.height / 2 : 0
    dispatch({ type: 'zoomBy', factor: e.deltaY < 0 ? 1.15 : 1 / 1.15, cx, cy })
  }
  const onPointerDown = (e: PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, mode: e.button === 2 || e.shiftKey ? 'wl' : 'pan' }
  }
  const onPointerMove = (e: PointerEvent) => {
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    d.x = e.clientX
    d.y = e.clientY
    dispatch(d.mode === 'pan' ? { type: 'pan', dx, dy } : { type: 'windowLevel', dx, dy })
  }
  const endDrag = () => {
    drag.current = null
  }

  const layer = 'absolute inset-0 m-auto max-h-full max-w-full select-none object-contain'
  const transform = `translate(${s.panX}px, ${s.panY}px) scale(${s.zoom})`

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-slate-800 bg-black">
      <div className="flex flex-wrap items-center gap-1 border-b border-slate-800 bg-slate-900 px-2 py-1.5 text-slate-300">
        <ToolButton label="Zoom in" onClick={() => dispatch({ type: 'zoomBy', factor: 1.25 })}>
          <Plus className="size-4" />
        </ToolButton>
        <ToolButton label="Zoom out" onClick={() => dispatch({ type: 'zoomBy', factor: 0.8 })}>
          <Minus className="size-4" />
        </ToolButton>
        <ToolButton label="Fit to view" onClick={() => dispatch({ type: 'reset' })}>
          <Maximize className="size-4" />
        </ToolButton>
        <span className="mx-1 h-5 w-px bg-slate-700" />
        <label className="sr-only" htmlFor="wl-preset">
          Window/level preset
        </label>
        <select
          id="wl-preset"
          className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-xs text-slate-200"
          defaultValue="Default"
          onChange={(e) => dispatch({ type: 'preset', name: e.target.value as WLPreset })}
        >
          {Object.keys(WL_PRESETS).map((p) => (
            <option key={p}>{p}</option>
          ))}
        </select>
        <ToolButton label="Invert" active={s.invert} onClick={() => dispatch({ type: 'toggleInvert' })}>
          <Contrast className="size-4" />
        </ToolButton>
        <span className="mx-1 h-5 w-px bg-slate-700" />
        <ToolButton label="Grad-CAM overlay" active={s.showHeatmap} onClick={() => dispatch({ type: 'toggleHeatmap' })}>
          <Flame className="size-4" />
        </ToolButton>
        <label className="flex items-center gap-2 text-xs text-slate-400">
          <span className="hidden sm:inline">Opacity</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={s.heatmapOpacity}
            disabled={!s.showHeatmap}
            aria-label="Grad-CAM opacity"
            onChange={(e) => dispatch({ type: 'heatmapOpacity', value: Number(e.target.value) })}
            className="w-20 accent-brand-500"
          />
        </label>
        <ToolButton label="Reset window/level" onClick={() => dispatch({ type: 'preset', name: 'Default' })}>
          <RotateCcw className="size-4" />
        </ToolButton>
        <span className="ml-auto hidden font-mono text-[11px] text-slate-500 lg:inline">
          {Math.round(s.zoom * 100)}% · W {s.contrast.toFixed(2)} · L {s.brightness.toFixed(2)}
        </span>
      </div>

      <div
        ref={frame}
        className="relative min-h-[420px] flex-1 cursor-grab touch-none overflow-hidden active:cursor-grabbing"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onContextMenu={(e) => e.preventDefault()}
        role="img"
        aria-label={alt}
      >
        {image.error && <p className="absolute inset-0 grid place-items-center text-sm text-red-300">{image.error}</p>}
        {!image.url && !image.error && (
          <p className="absolute inset-0 grid place-items-center text-sm text-slate-500">Loading image…</p>
        )}
        <div className="absolute inset-0" style={{ transform, transformOrigin: 'center' }}>
          {image.url && (
            <img src={image.url} alt="" draggable={false} className={layer} style={{ filter: imageFilter(s) }} />
          )}
          {heatmap.url && s.showHeatmap && (
            <img
              src={heatmap.url}
              alt=""
              draggable={false}
              className={clsx(layer, 'pointer-events-none mix-blend-normal')}
              style={{ opacity: s.heatmapOpacity }}
            />
          )}
        </div>
        <p className="pointer-events-none absolute bottom-2 left-3 text-[11px] text-slate-500">
          Scroll: zoom · Drag: pan · Right-drag or Shift-drag: window/level
        </p>
      </div>
    </div>
  )
}

function ToolButton({
  label,
  onClick,
  active,
  children,
}: {
  label: string
  onClick: () => void
  active?: boolean
  children: ReactNode
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      aria-pressed={active}
      onClick={onClick}
      className={clsx(
        'rounded-md p-1.5 transition-colors hover:bg-slate-800 hover:text-white',
        active && 'bg-slate-800 text-brand-100',
      )}
    >
      {children}
    </button>
  )
}
