# Citation locator audit

We retain numeric citations in manuscript prose. We place page locators in
the bibliography, using `p.`, `pp.` and `pdf p.`. PDF locators count physical
pages in the linked version, starting at one; they are not inferred offsets
into a different proceedings edition.

## Verified passage locators

| Source | Bibliography locator | Passage used |
|---|---|---|
| C2PA specification 2.2 | pp. 47, 59, 63 | Manifest container, signature algorithms and trust model; physical PDF pages 52, 64, 68 |
| Cox et al., book | pp. 19-21 | Owner identification and proof-of-ownership sections, verified against the book contents |
| Craver et al. | pp. 576-578 | Ownership ambiguity and counterclaims |
| Zhao et al. | pp. 8645-8646 | Regeneration attack and method |
| Jiang et al. | pp. 1168-1169 | Introduction to black-box/query-access evasion |
| WAVES | p. 1456 | Benchmark purpose |
| Gloaguen et al. | p. 19562 | Spoofing-discrimination problem |
| Kirchenbauer et al. | p. 17061 | Original text-watermark construction, not a spoofing attack |
| Mueller et al. | p. 20937 | Black-box image-forgery access assumptions |
| Gneiting and Raftery | p. 360 | Proper scoring-rule definition |
| Cox et al., spread spectrum | pp. 1678-1679 | Insertion and similarity detection |
| Chen and Wornell | p. 1426 | QIM construction |
| Al-Haj | p. 745 | JPEG robustness comparison |
| SDXL | pdf p. 1-3 | Model overview and architecture |
| Liu et al. | pdf p. 2-3 | Clean-noise controllable regeneration |
| Saberi et al. | pdf p. 1-3 | Robustness limits and practical attacks |
| Yi et al. | pdf p. 1-2 | Spoofing/scrubbing framework |
| Sadasivan et al. | pdf p. 1, 4 | Detectability limits and spoofing contribution |
| WMCopier | pdf p. 1-2, 4-5, 23 | Unconditional model, shallow inversion and original DWT-DCT bit accuracy |
| Kinakh et al. | pdf p. 3 | Copy/removal algorithm |
| HiDDeN | pdf p. 1 | Learned robustness to listed distortions |
| StegaStamp | pdf p. 3 | Training with physical-recapture perturbations |
| Tree-Ring | pdf p. 5 | Initial-noise watermark construction |
| Stable Signature | pdf p. 4 | Generative decoder fine-tuning |
| Artificial fingerprinting | pdf p. 3 | Fingerprinted training-data construction |
| SSIM author-hosted preprint | pdf p. 5-6, 8 | SSIM construction and mean image-quality index |

We corrected the missing SPIE page range for *The watermark copy attack*
to pp. 371-380. We retained verified full publication ranges for general
method/dataset citations or when we could not independently verify a narrower
claim page: Rombach et al., Goldwasser et al., Bellare et al., Petitcolas et al.,
Kutter et al., Skellam, Barni et al., Wallace, COCO, DiffusionDB, Wilson,
Efron and Clopper-Pearson. These ranges identify the publication, not a claim
that we checked every page or isolated the narrowest supporting passage.

## Nonpaginated primary sources

We removed the optional invisible-watermark issue-thread reference and its
lossless-roundtrip anecdote. We retained our implementation-based finding
without claiming first discovery.

We retain three essential implementation references: invisible-watermark,
original CompVis issuance code and the Diffusers SDXL watermarker. Source
repositories have no genuine PDF pagination, and a scholarly paper does not
replace the exact code we audited. We retain repository/version locators
instead of fabricating page numbers.

We retain Article 50(2) and Recital 133 as provision locators for the AI Act.
The legal text was verified, but our direct official-PDF retrieval returned
an empty response, so we did not invent physical page numbers.

## Primary PDF locations used for additional direct checks

- [C2PA 2.2](https://spec.c2pa.org/specifications/specifications/2.2/specs/_attachments/C2PA_Specification.pdf)
- [SDXL](https://arxiv.org/pdf/2307.01952)
- [Saberi et al.](https://arxiv.org/pdf/2310.00076)
- [Sadasivan et al.](https://arxiv.org/pdf/2303.11156)
- [WMCopier](https://arxiv.org/pdf/2503.22330)
- [HiDDeN](https://arxiv.org/pdf/1807.09937)
- [SSIM author-hosted preprint](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
