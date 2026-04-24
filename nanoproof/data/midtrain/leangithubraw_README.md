---
license: other
license_name: mixed
license_link: LICENSE
language:
  - en
  - lean
task_categories:
  - text-generation
tags:
  - lean4
  - theorem-proving
  - code
pretty_name: Lean-Github-Raw
size_categories:
  - 10K<n<100K
---

# Lean-Github-Raw

Raw `.lean` source files collected from a curated set of public Lean 4
GitHub repositories. Each row contains the full file contents together
with the upstream URL and the pinned commit hash, so every sample is
traceable to a specific revision.

## Repository selection

The list of repositories was taken from **LEAN-GitHub** (Wu et al., 2024):

- Paper: https://arxiv.org/abs/2407.17227
- Original dataset: https://huggingface.co/datasets/internlm/Lean-Github

We reuse their repository selection, but the contents here are our own
fresh scrape of the Lean source - not a mirror of `internlm/Lean-Github`.
A few repositories from the original list are excluded (unavailable
upstream, or known to contaminate evaluation benchmarks); see the build
script in nanoproof for the exact selection.

Dataset built and published from https://github.com/Kripner/nanoproof.

## Schema

| column   | type   | description                                      |
|----------|--------|--------------------------------------------------|
| `text`   | string | full `.lean` file contents (UTF-8)               |
| `url`    | string | `https://github.com/<repo>/blob/<commit>/<path>` |
| `commit` | string | git commit hash the file was read from           |

## Files

- `leangithubraw.parquet` - the full dataset (~142 MB of text), row-group
  size 1024 for efficient streaming. The last 4 row groups are held out
  as the validation split.
- `repo_*.parquet` - per-repository shards used during the build; safe to
  ignore unless you want to slice by source repo.
- `licenses/<repo>/` - a copy of the LICENSE / LICENCE / COPYING file(s)
  from each source repository, preserving the original filename. Repos
  that did not ship any license file have a `NO_LICENSE.md` describing
  the default-copyright status (see "License" below).

## License

The compilation itself (the selection, schema, and metadata) is released
under the MIT License, matching the nanoproof project. See the root
`LICENSE` file.

**Each `.lean` sample retains the license of its upstream repository.**
Consult `licenses/<repo>/` for the original license text, and the `url`
and `commit` columns to resolve the exact source file. Repositories that
published no LICENSE file are, per GitHub default, under standard
copyright: all rights reserved by the original authors. See
https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository#choosing-the-right-license

## Citation

If you use this dataset, please cite LEAN-GitHub whose repository
selection we build on:

```bibtex
@misc{wu2024leangithubcompilinggithublean,
      title={LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover},
      author={Zijian Wu and Jiayu Wang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2407.17227},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.17227},
}
```
