# Third-party notices and license scope

The root [MIT License](LICENSE) applies to original PopulationTransformer code
and documentation. Third-party material retains its own terms and is not
relicensed by the MIT grant.

Reference versions below identify material inspected, not necessarily the exact
revisions originally copied.

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
additions. No endorsement by the upstream authors is implied.

## SciPy-derived normalization helpers

- **Local code:** `_first` and `zscore` in `preprocessors/stft.py`.
- **Source cited in code:** [`scipy/stats/_stats_py.py` at v1.9.0](https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py).
- **Copyright:** Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
- **License:** BSD-3-Clause; full upstream notice in
  [LICENSES/SciPy-LICENSE.txt](LICENSES/SciPy-LICENSE.txt).

The local implementation simplifies normalization and replaces zero standard
deviations with one. These adapted portions retain the SciPy terms.
