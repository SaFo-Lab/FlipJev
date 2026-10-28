# Agent-trajectory benchmarks

The trajectory-judge experiments use two public benchmarks. They are not shipped with this repository; download them
into this directory (or set `TRAJECTORY_DATA` to where they live) so that the loaders find them at:

```
data/trajectories/
  R-Judge/data/<scenario>/<file>.json        # R-Judge
  ATBench/ATBench500/test.json               # ATBench500
```

## R-Judge

571 labelled records of agent interactions over 27 scenarios. Clone the benchmark repository and copy its `data`
directory here:

```
git clone https://github.com/Lordog/R-Judge
cp -r R-Judge/data data/trajectories/R-Judge/data
```

The trajectory-judge study uses only the unsafe records whose metadata marks them as injections (an outside party's
text sits inside a tool output).

## ATBench500

The 500-trajectory release introduced with AgentDoG. Download the benchmark from its Hugging Face dataset and keep
the `ATBench500/test.json` file:

```
huggingface-cli download AI45Research/ATBench --repo-type dataset --local-dir data/trajectories/ATBench
```

## Trajectory ids

The trajectories attacked in the paper are listed by case id in `results/per_document/trajectories/<benchmark>/asr/`
and `results/universal/trajectories/<benchmark>/selection.json`. The released result files keep only case ids,
suffixes, probabilities and scores; no trajectory text is shipped.
