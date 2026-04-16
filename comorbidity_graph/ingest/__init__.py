"""Load raw prediction CSVs and (later) the MIMIC-IV diagnosis file."""

from comorbidity_graph.ingest.load_predictions import load_predictions

__all__ = ["load_predictions"]
