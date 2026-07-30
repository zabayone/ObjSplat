# Documentation

- [`ObjSplat_Benchmark_Metrics.docx`](ObjSplat_Benchmark_Metrics.docx):
  complete catalogue of benchmark fields, units, definitions, availability, and
  scientific interpretation.

The DOCX is generated reproducibly with:

```bash
PYTHONPATH=/path/to/python-docx \
python benchmark/tools/build_metrics_docx.py
```

The source script uses the stable schemas in `benchmark/schemas.py`; regenerate
the document whenever a schema column is appended.
