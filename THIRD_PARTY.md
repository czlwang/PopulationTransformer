# Third-party notices and license scope

The root [MIT License](LICENSE) applies to original PopulationTransformer code
and documentation. Third-party material retains its applicable terms, including
the exceptions and unresolved sources below. No MIT permission is granted here
for material whose rights are held by someone else.

This review accompanies the license branch based on commit
`dadb55b21daf2809b123d49a9594ff6eebea2c40` and was prepared on 2026-09-13.
Reference versions below identify material inspected, not necessarily the exact
revisions originally copied. This document does not certify that all upstream
permissions have been established.

## Transformer layers: PyTorch and Buomsoo Kim's tutorial

- **Local code:** `_get_activation_fn`, `TransformerEncoderLayer`, `_get_clones`,
  and `TransformerEncoder` in `models/pt_model_custom.py`.
- **Credited source:** Buomsoo Kim's
  ["Attention in Neural Networks - 21. Transformer (5)"](https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/),
  published 2020-04-27. The site displays "© 2022 Buomsoo Kim" and
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
  ([legal terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)).
- **Earlier source:** the tutorial adapts PyTorch's transformer implementation.
  Reference: [`torch/nn/modules/transformer.py` at v1.5.0](https://github.com/pytorch/pytorch/blob/v1.5.0/torch/nn/modules/transformer.py).
- **Preserved PyTorch terms:** [LICENSES/PyTorch-LICENSE.txt](LICENSES/PyTorch-LICENSE.txt)
  contains the full copyright notices, BSD-style conditions, and disclaimer from
  that version.

The local classes adapt the tutorial's attention-returning encoder and integrate
it into PopT. Preserve attribution and modification notices and observe the
noncommercial restriction for material covered by the tutorial's grant. Neither
the root MIT license nor PyTorch's license overrides rights in the tutorial's
additions. No endorsement by the upstream authors is implied. A fully permissive
release would require resolving any applicable noncommercial restriction.

## SciPy-derived normalization helpers

- **Local code:** `_first` and `zscore` in `preprocessors/stft.py`.
- **Source cited in code:** [`scipy/stats/_stats_py.py` at v1.9.0](https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py).
- **Copyright:** Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
- **License:** BSD-3-Clause; full upstream notice in
  [LICENSES/SciPy-LICENSE.txt](LICENSES/SciPy-LICENSE.txt).

The local implementation simplifies normalization and replaces zero standard
deviations with one. These adapted portions retain the SciPy terms.

## Sources requiring further provenance or permission confirmation

These entries preserve existing attribution without assigning an unsupported MIT
license to the borrowed material:

| Local code | Existing source or author credit | Remaining question |
| --- | --- | --- |
| `preprocessors/superlet.py` | Gregor Mönke, [tensionhead](https://github.com/tensionhead); implementation of Moca et al.'s superlet method | Establish the exact source revision and applicable implementation license. A paper citation alone is not a code license. |
| `util/tensorboard_utils.py::plot_to_tensorboard` | [Martin Mundt's TensorBoard figures tutorial](https://martin-mundt.com/tensorboard-figures/) | Confirm the copied source's terms; the page could not be retrieved during this review. |
| `data/utils.py::compute_m5_hash` and associated helper | [Stack Overflow answer 3431835](https://stackoverflow.com/a/3431835) | Confirm the author and revision actually used and any applicable attribution/share-alike requirements. |
| `models/transformer_encoder_input.py::PositionalEncoding` | [PyTorch forum discussion](https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2) | The cited post contains no code; identify the actual snippet and applicable terms for any forum-specific contribution. |

Stack Overflow's [licensing policy](https://stackoverflow.com/help/licensing)
assigns different CC BY-SA versions according to the contribution/revision date;
the currently displayed website license is not enough to identify the terms of a
historically copied answer. No specific version is assigned here without that
provenance.

Installed packages, including PyTorch, SciPy, and `warmup_scheduler`, retain their
own package licenses. This inventory concerns incorporated source and does not
replace those packages' notices.

## Weights, data, and maintainer review

The root MIT license does not grant rights to separately distributed pretrained
weights, BrainBERT artifacts, or external datasets. Their providers need to state
the applicable terms independently.

Before merging the license change, maintainers should confirm that
"PopulationTransformer contributors" is the appropriate copyright attribution
and that the relevant contributors or institutions authorize the MIT grant for
their original contributions. Resolve or retain the third-party exceptions above;
do not describe the entire repository as unrestricted MIT code while they remain.
