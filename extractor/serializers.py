from rest_framework import serializers
from .models import ExtractionJob


class ExtractionRequestSerializer(serializers.Serializer):
    document_code = serializers.CharField()
    ocr_draft = serializers.CharField()
    # Optional model override.  Accepts a profile alias ("gpt-oss", "gemini")
    # or any raw model ID (e.g. "gpt-oss:120b", "gemini-2.5-flash").
    # When omitted, the pipeline uses LLM_MODEL_PRIMARY from settings.
    model = serializers.CharField(required=False, allow_blank=True, default="")

    def validate_document_code(self, value):
        if value != ExtractionJob.DOCUMENT_CODE_04021:
            raise serializers.ValidationError(
                f"This endpoint only handles document_code '{ExtractionJob.DOCUMENT_CODE_04021}'. "
                f"Got '{value}'."
            )
        return value


class ExtractionJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExtractionJob
        fields = [
            "id", "document_code", "model_id", "status",
            "result", "error", "duration_ms", "metrics",
            "created_at", "updated_at",
        ]
        read_only_fields = fields
