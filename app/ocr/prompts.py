"""Versioned prompts shared by local vision OCR providers."""

OCR_TRANSCRIPTION_PROMPT_VERSION = "ocr-transcription-v1"

OCR_TRANSCRIPTION_PROMPT = """Transcribe only visible student-written answer text.
Preserve spelling, grammar, capitalization, punctuation, and meaningful line breaks
exactly where visible. Do not correct English, grade, explain, add commentary, or
invent unreadable words. Exclude printed examination text and teacher marks or
corrections where distinguishable. Return the transcription only. If no
student-written answer text is visible, return an empty response."""
