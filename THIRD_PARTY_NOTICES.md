# Third Party Notices

## PureJaxRL

The actor-critic interface used to train the released models follows the continuous-action PPO implementation in [PureJaxRL](https://github.com/luchris429/purejaxrl). The project changes the hidden network to three 512-unit GELU and LayerNorm layers and uses a density-based observation and swarm-allocation action interface. Frozen observation statistics are applied during evaluation.

The released `policy.py` provides a separate NumPy implementation of mean-action inference. It loads actor parameters and observation statistics, without restoring a critic, optimizer, or training process. The terminal-allocation and lower-level control modules in `src/` are project implementations, not PureJaxRL modules.

The relevant upstream references are:

- [Continuous-action PPO and ActorCritic interface](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py)
- [Observation-normalization wrappers](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py)
- [PureJaxRL license](https://github.com/luchris429/purejaxrl/blob/main/LICENSE)

PureJaxRL identifies its copyright as **Copyright 2023 Chris Lu** and is licensed under the Apache License, Version 2.0. The upstream license text is retained in [third_party/purejaxrl_LICENSE.txt](third_party/purejaxrl_LICENSE.txt).

License source checked on 7 November 2026: [official raw license](https://raw.githubusercontent.com/luchris429/purejaxrl/main/LICENSE). Its source SHA-256 is `00387162b333dff9ed44a187bf7bc9b11ea49b0021c3910b0020574cb5205520`. The included text has the same contents with a final newline; its SHA-256 is `a51fa4d095e3d7b2edd1114ab677eee751a29410032b2688e19f858d6c76ef8b`.

This notice preserves upstream attribution. It does not apply a new repository-wide license to the original project code or model weights.

## Runtime dependencies

NumPy, SciPy, and OSQP are installed separately under their respective upstream licenses. They are not vendored in this repository. The optional native kernels are built from the C++ sources in `src/`; precompiled binaries are not included.
