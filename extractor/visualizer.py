"""
LangExtract visualizer integration.

Converts the stored viz_data (a serialized AnnotatedDocument dict) into a
standalone HTML page using lx.visualize().

Usage
-----
>>> from extractor.visualizer import build_visualization_html
>>> html = build_visualization_html(job.viz_data, job_id=job.pk)
>>> # Return as HttpResponse with content_type="text/html"
"""

import logging

import langextract as lx
from langextract import data_lib

logger = logging.getLogger(__name__)

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LangExtract Visualizer — Job {job_id}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px 24px;
      background: #f5f5f5;
    }}
    h1 {{
      font-size: 18px;
      color: #1565c0;
      margin-bottom: 4px;
    }}
    .meta {{
      font-size: 13px;
      color: #555;
      margin-bottom: 20px;
    }}
    .container {{
      background: #fff;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 1px 4px rgba(0,0,0,.12);
    }}
  </style>
</head>
<body>
  <h1>LangExtract Extraction Visualizer</h1>
  <div class="meta">Job ID: <strong>{job_id}</strong> &nbsp;|&nbsp; Extractions: <strong>{count}</strong></div>
  <div class="container">
    {viz_html}
  </div>
</body>
</html>
"""


def build_visualization_html(viz_data: dict, job_id: int) -> str:
    """
    Turn the stored viz_data dict back into an AnnotatedDocument and render
    the LangExtract interactive HTML visualizer.

    Args:
        viz_data:  The JSON dict previously produced by
                   ``data_lib.annotated_document_to_dict()``.
        job_id:    Used only for the page title.

    Returns:
        A complete standalone HTML page as a string.

    Raises:
        ValueError: If viz_data is empty or has no extractions.
    """
    if not viz_data:
        raise ValueError("viz_data is empty — no visualizer data stored for this job.")

    annotated_doc = data_lib.dict_to_annotated_document(viz_data)

    if not annotated_doc.extractions:
        raise ValueError("AnnotatedDocument contains no extractions to visualize.")

    # lx.visualize() returns HTML string when not in Jupyter
    viz_html = lx.visualize(annotated_doc, animation_speed=1.5, gif_optimized=False)

    # In some environments it returns an IPython HTML object — unwrap it
    if hasattr(viz_html, "data"):
        viz_html = viz_html.data

    count = len(annotated_doc.extractions)
    return _PAGE_TEMPLATE.format(job_id=job_id, count=count, viz_html=viz_html)
