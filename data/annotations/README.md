# Annotations

This directory contains manual annotations of biosignature sites.

## Annotation Format

Annotations are stored as JSON files with the following schema:

```json
{
  "image_id": "NLF_1234_...",
  "sol": 1234,
  "target_name": "Cheyava Falls",
  "labels": ["leopard_spots", "organic_signatures", "halos"],
  "bounding_boxes": [
    {"x": 100, "y": 200, "width": 50, "height": 50, "label": "halos"}
  ],
  "annotator": "expert_1",
  "confidence": 0.9,
  "notes": "Distinctive redox features observed"
}
```

## Known Biosignature Sites

- **Cheyava Falls** (Sol 1174): Leopard-spot patterns with organic signatures
- **Wildcat Ridge** (Sol 528): Strongest organic detection in Jezero Crater
- **Delta deposits**: Sedimentary structures in ancient river delta
