```bash
(localllm310_stage2) shen0574@agc10 [~/GopherEye_cleaned_V6_V8_dataset] %  python how_to_use/assemble_model_demo.py   --ckpt_dir qwen3-1.7B-greenhouse-ngld-faithful-stage2   --image ex_image.jpg
```

```bash
 raw rext
Disease: Downy Mildew.

Indicators:
- Yellowing primarily on the adaxial surface
- Chlorotic patches are vein-bounded
- Interveinal chlorosis is more pronounced near the base
- Necrotic spots appear as small, circular areas
- Leaf edges show slight curling or distortion
- Some veins remain green despite surrounding discoloration
- Discoloration appears to follow a consistent pattern across the leaf

Recommended checks:
- Inspect the abaxial side for any fuzzy growth.
- Check for moisture accumulation on the underside of leaves.
- Examine nearby plants for similar symptoms.
- Look for water-soaked areas under direct sunlight.
- Assess the plant's overall health by checking for wilting.

Evidence:
The described grape leaf exhibits characteristic symptoms indicative of downy mildew. The primary feature is yellowing (chlorosis) predominantly on the upper surface, which often follows a specific pattern along the veins. This chlorosis tends to be confined within the interveinal spaces, suggesting that it does not cross over into the veins themselves. Small necrotic lesions can form around these chlorotic regions, appearing as dark brown or yellowish spots. Additionally, there is evidence of slight curliness at the leaf margins, indicating

 parsed text
{
  "disease": "downy mildew",
  "indicators": [
    "yellowing primarily on the adaxial surface",
    "chlorotic patches are vein-bounded",
    "interveinal chlorosis is more pronounced near the base",
    "necrotic spots appear as small, circular areas",
    "leaf edges show slight curling or distortion",
    "some veins remain green despite surrounding discoloration",
    "discoloration appears to follow a consistent pattern across the leaf"
  ],
  "recommended_checks": [
    "inspect the abaxial side for any fuzzy growth.",
    "check for moisture accumulation on the underside of leaves.",
    "examine nearby plants for similar symptoms.",
    "look for water-soaked areas under direct sunlight.",
    "assess the plant's overall health by checking for wilting."
  ],
  "evidence": "the described grape leaf exhibits characteristic symptoms indicative of downy mildew. the primary feature is yellowing (chlorosis) predominantly on the upper surface, which often follows a specific pattern along the veins. this chlorosis tends to be confined within the interveinal spaces, suggesting that it does not cross over into the veins themselves. small necrotic lesions can form around these chlorotic regions, appearing as dark brown or yellowish spots. additionally, there is evidence of slight curliness at the leaf margins, indicating",
  "has_disease_section": true,
  "has_indicators_section": true,
  "has_checks_section": true,
  "has_evidence_section": true
}
```
