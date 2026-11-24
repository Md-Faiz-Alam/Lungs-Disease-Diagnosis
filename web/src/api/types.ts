export type Triage = 'urgent' | 'review' | 'routine'
export type ReviewStatus = 'pending' | 'agreed' | 'disagreed'
export type ClassLabel = 'COVID19' | 'NORMAL' | 'PNEUMONIA'

export interface Study {
  id: string
  owner: string
  created_at: string
  filename: string
  source: 'dicom' | 'image'
  prediction: ClassLabel
  confidence: number
  probabilities: Record<ClassLabel, number>
  needs_review: boolean
  triage: Triage
  warnings: string[]
  dicom_metadata: Record<string, string>
  image_width: number
  image_height: number
  model_arch: string
  model_run_id: string
  review_status: ReviewStatus
  review_label: string
  review_note: string
  reviewed_at: string
  reviewer: string
}

export interface ModelInfo {
  arch: string
  run_id: string
  created_at: string
  classes: ClassLabel[]
  temperature: number
  referral_threshold: number
  metrics: {
    accuracy?: number
    balanced_accuracy?: number
    macro_f1?: number
    macro_roc_auc?: number
    ece?: number
    abnormal_sensitivity?: number
    per_class_recall?: Record<string, number>
  }
}

export interface AuditEvent {
  user_id: string
  user_name: string
  action: string
  study_id: string
  detail: string
  ts: string
}

export interface Me {
  id: string
  name: string
  email: string
  is_demo: boolean
  auth_mode: 'demo' | 'entra'
}
