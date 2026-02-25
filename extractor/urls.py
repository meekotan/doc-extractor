from django.urls import path
from .views import InvoiceExtractView, ExtractionJobDetailView, MetricsSummaryView, HealthCheckView, VisualizationView

urlpatterns = [
    path("extract/",                  InvoiceExtractView.as_view(),       name="invoice-extract"),
    path("jobs/<int:pk>/",            ExtractionJobDetailView.as_view(),  name="job-detail"),
    path("jobs/<int:pk>/visualize/",  VisualizationView.as_view(),        name="job-visualize"),
    path("metrics/",                  MetricsSummaryView.as_view(),       name="metrics-summary"),
    path("health/",                   HealthCheckView.as_view(),          name="health-check"),
]
