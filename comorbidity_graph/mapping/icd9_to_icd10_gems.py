"""CMS General Equivalence Mappings: ICD-9-CM -> ICD-10-CM.

Deferred per the approved plan ("ICD-10 only for now, keep v9 as a backup").
When we want to add ICD-9 MIMIC-IV rows back in, this module will:
  1. load the CMS GEMs table,
  2. translate each ICD-9 code to one or more ICD-10-CM codes,
  3. reduce to the 3-char ICD-10 node granularity.
Not implemented yet.
"""

from __future__ import annotations


def translate_icd9_to_icd10(icd9_code: str) -> list[str]:
    raise NotImplementedError("ICD-9 -> ICD-10 GEMs translation deferred.")
