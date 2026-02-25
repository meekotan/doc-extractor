import logging

from django.http import HttpResponse
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Avg
from langextract import data_lib

from .extraction import run_invoice_extraction
from .health import check_database, check_llm_api
from .models import ExtractionJob
from .serializers import ExtractionRequestSerializer, ExtractionJobSerializer
from .visualizer import build_visualization_html

logger = logging.getLogger(__name__)


def _serialize_annotated_doc(annotated_doc) -> dict | None:
    try:
        if annotated_doc is None:
            return None
        return data_lib.annotated_document_to_dict(annotated_doc)
    except Exception as exc:
        logger.warning("Could not serialize AnnotatedDocument: %s", exc)
        return None


class InvoiceExtractView(APIView):
    """
    POST /api/extract/
    Body: {"document_code": "04021", "ocr_draft": "<raw OCR text>"}

    Runs the extraction pipeline synchronously and returns the result.
    """

    def post(self, request):
        serializer = ExtractionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        model_spec = serializer.validated_data.get("model", "") or ""

        job = ExtractionJob.objects.create(
            document_code=serializer.validated_data["document_code"],
            ocr_draft=serializer.validated_data["ocr_draft"],
            status=ExtractionJob.STATUS_PROCESSING,
        )

        try:
            output = run_invoice_extraction(job.ocr_draft, model_id=model_spec or None)
        except Exception as exc:
            logger.exception("Unhandled exception in extraction pipeline (job_id=%s)", job.pk)
            job.status = ExtractionJob.STATUS_FAILED
            job.error = str(exc)
            job.save(update_fields=["status", "error", "updated_at"])
            return Response(ExtractionJobSerializer(job).data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        metrics = output.get("metrics", {})
        duration_ms = int(metrics.get("t_total_s", 0) * 1000)
        viz_data = _serialize_annotated_doc(output.get("annotated_doc"))
        model_id_used = output.get("model_id", "")

        if "error" in output:
            job.status = ExtractionJob.STATUS_FAILED
            job.error = output["error"]
            job.metrics = metrics
            job.duration_ms = duration_ms
            job.viz_data = viz_data
            job.model_id = model_id_used
            job.save(update_fields=["status", "error", "metrics", "duration_ms", "viz_data", "model_id", "updated_at"])
            return Response(ExtractionJobSerializer(job).data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        job.status = ExtractionJob.STATUS_SUCCESS
        job.result = output["result"]
        job.metrics = metrics
        job.duration_ms = duration_ms
        job.viz_data = viz_data
        job.model_id = model_id_used
        job.save(update_fields=["status", "result", "metrics", "duration_ms", "viz_data", "model_id", "updated_at"])
        return Response(ExtractionJobSerializer(job).data, status=status.HTTP_200_OK)


class ExtractionJobDetailView(APIView):
    """
    GET /api/jobs/<pk>/
    """

    def get(self, request, pk):
        try:
            job = ExtractionJob.objects.get(pk=pk)
        except ExtractionJob.DoesNotExist:
            return Response({"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND)

        return Response(ExtractionJobSerializer(job).data)


class MetricsSummaryView(APIView):
    """
    GET /api/metrics/
    Aggregate stats across all stored ExtractionJob records.
    """

    def get(self, request):
        qs = ExtractionJob.objects.all()
        total = qs.count()
        if total == 0:
            return Response({"total_jobs": 0, "message": "No jobs recorded yet."})

        success_count = qs.filter(status=ExtractionJob.STATUS_SUCCESS).count()
        failed_count = qs.filter(status=ExtractionJob.STATUS_FAILED).count()

        timed_qs = qs.filter(duration_ms__isnull=False)
        avg_duration = timed_qs.aggregate(avg=Avg("duration_ms"))["avg"] or 0

        fallback_count = sum(
            1 for j in qs.exclude(metrics=None)
            if j.metrics.get("fallback_used", False)
        )

        avg_items = sum(
            j.metrics.get("items_extracted", 0)
            for j in qs.exclude(metrics=None)
        ) / max(timed_qs.count(), 1)

        field_totals: dict = {}
        field_job_count = 0
        for job in qs.filter(status=ExtractionJob.STATUS_SUCCESS).exclude(metrics=None):
            rates = job.metrics.get("field_fill_rates", {})
            if rates:
                field_job_count += 1
                for field, rate in rates.items():
                    field_totals[field] = field_totals.get(field, 0.0) + rate

        avg_field_fill = (
            {f: round(v / field_job_count, 3) for f, v in field_totals.items()}
            if field_job_count else {}
        )

        step_keys = ["t_clean_s", "t_primary_llm_s", "t_validate_s", "t_fallback_llm_s", "t_finalize_s", "t_total_s"]
        step_totals = {k: 0.0 for k in step_keys}
        step_count = 0
        for job in timed_qs.exclude(metrics=None):
            step_count += 1
            for k in step_keys:
                step_totals[k] += job.metrics.get(k, 0.0)

        avg_step_times = (
            {k: round(v / step_count, 3) for k, v in step_totals.items()}
            if step_count else {}
        )

        return Response({
            "total_jobs": total,
            "success_count": success_count,
            "failed_count": failed_count,
            "success_rate": round(success_count / total, 3),
            "fallback_used_count": fallback_count,
            "fallback_rate": round(fallback_count / total, 3),
            "avg_duration_ms": round(avg_duration, 1),
            "avg_items_extracted": round(avg_items, 2),
            "avg_step_times_s": avg_step_times,
            "avg_field_fill_rates": avg_field_fill,
        })


class VisualizationView(APIView):
    """
    GET /api/jobs/<pk>/visualize/
    Returns a standalone interactive HTML page (LangExtract visualizer).
    """

    def get(self, request, pk):
        try:
            job = ExtractionJob.objects.get(pk=pk)
        except ExtractionJob.DoesNotExist:
            return Response({"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND)

        if not job.viz_data:
            return Response(
                {"error": "No visualization data available for this job."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            html = build_visualization_html(job.viz_data, job_id=pk)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            logger.exception("Failed to build visualization for job_id=%s", pk)
            return Response({"error": "Failed to render visualizer."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return HttpResponse(html, content_type="text/html; charset=utf-8")


class HealthCheckView(APIView):
    """
    GET /api/health/
    Checks: database, llm_api.
    HTTP 200 when all pass; HTTP 503 when any fails.
    """

    def get(self, request):
        db  = check_database()
        llm = check_llm_api()

        all_ok = all(c["status"] == "ok" for c in [db, llm])
        http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE

        return Response(
            {
                "status":   "ok" if all_ok else "degraded",
                "database": db,
                "llm_api":  llm,
            },
            status=http_status,
        )
