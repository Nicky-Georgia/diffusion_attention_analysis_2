# Expected annotation structure

Attention localization and concept-dictionary stages expect binary PNG masks aligned to generated images:

```text
annotations/
  single_dog_seed_0000/
    dog.png
    grass.png
  red_car_seed_0000/
    car.png
```

Mask filenames are matched to prompt labels / target words. You can create these masks manually for the curated subset or with GroundingDINO + SAM / CLIPSeg and then inspect them.
