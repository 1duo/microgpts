# microgpts

Port of Karpathy's tiny `microgpt.py` to different programming language for fun.

## Run

Just run `run.sh`, e.g.,:

```bash
./Python/run.sh
./Mojo/run.sh
```

Each language folder has its own `run.sh`.

## Meaningless Benchmark For Fun

10 runs each, randomized different order per round. Python is baseline (`1.00x`).

| PL | Avg Runtime (s) | Speed vs. Python (X) |
|---|---:|---:|
| C | 0.746 | 87.70x |
| Rust | 0.998 | 65.52x |
| Mojo | 1.501 | 43.57x |
| Zig | 4.214 | 15.52x |
| Go | 6.961 | 9.39x |
| C++ | 8.549 | 7.65x |
| TypeScript | 9.882 | 6.62x |
| Kotlin | 11.731 | 5.57x |
| Swift | 20.492 | 3.19x |
| Python | 65.396 | 1.00x |
