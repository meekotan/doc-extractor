from django.db import models


class ExtractionJob(models.Model):
    """Tracks a single invoice extraction request."""

    DOCUMENT_CODE_04021 = "04021"

    STATUS_PENDING    = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_SUCCESS    = "success"
    STATUS_FAILED     = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING,    "Pending"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_SUCCESS,    "Success"),
        (STATUS_FAILED,     "Failed"),
    ]

    document_code = models.CharField(max_length=20)
    ocr_draft = models.TextField(help_text="Raw OCR text of the document")
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_PENDING)
    result = models.JSONField(null=True, blank=True, help_text="Extracted structured data")
    error = models.TextField(blank=True, default="")
    # Metrics
    duration_ms = models.IntegerField(null=True, blank=True, help_text="Total wall-clock time in milliseconds")
    metrics = models.JSONField(null=True, blank=True, help_text="Per-step timing and accuracy metrics")
    # LLM model used for this extraction (primary model id or profile alias)
    model_id = models.CharField(max_length=100, blank=True, default="", help_text="LLM model used for extraction (e.g. 'gpt-oss:120b', 'gemini-2.5-flash')")
    # LangExtract visualizer data — serialized AnnotatedDocument dict
    viz_data = models.JSONField(null=True, blank=True, help_text="Serialized LangExtract AnnotatedDocument for the visualizer")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"ExtractionJob({self.document_code}, {self.status}, id={self.pk})"
