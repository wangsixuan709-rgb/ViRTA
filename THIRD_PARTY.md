# Third-party runtime sources

The publication source keeps only the runtime files used by this project, including required tokenizer vocabularies, architecture configuration files and frost textures. Original source headers are retained.

- FineLIP: retained `model/` runtime and upstream README. Repository and exact checkout revision are recorded in `third_party_sources.json`. The supplied checkout has no standalone license file; no new license is assigned to it here.
- Chinese-CLIP: retained `cn_clip/clip/`, package initializer and the original `MIT-LICENSE.txt`. Repository and exact checkout revision are recorded in `third_party_sources.json`.
- NAFNet: `nafnet_original.py` retains its Megvii copyright and paper attribution.
- Image corruptions: `vprm_corruption/` retains its upstream attribution and required frost textures. It is used by restoration pretraining.

Model weights, third-party training suites, examples, notebooks, deployment helpers, repository histories and evaluation scripts are excluded from the vendored runtime.
