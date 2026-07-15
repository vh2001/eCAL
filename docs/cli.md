# CLI Reference

The `ecal` console script is installed alongside the package
({py:func}`ecal.cli.main`).

```bash
ecal --version
ecal --help
```

## `ecal estimate`

Estimate the energy cost of an AI model lifecycle.

```bash
ecal estimate --model MLP --layers 3 --epochs 50 --hardware apple_m2
```

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model architecture: `MLP`, `CNN`, `KAN`, or `Transformer` |
| `--layers` | 3 | Number of layers |
| `--din` | 10 | Input dimension |
| `--dout` | 2 | Output dimension |
| `--epochs` | 50 | Training epochs |
| `--samples` | 1000 | Number of training samples |
| `--sample-size` | 10 | Sample size / number of features |
| `--inferences` | 10000 | Number of inference calls |
| `--hardware` | *(none)* | Hardware profile name (see [Configuration](configuration.md)) |
| `--json` | off | Output the result as JSON instead of a formatted table |

### Transformer-specific flags

| Flag | Default |
|---|---|
| `--context-length` | 10 |
| `--embedding-size` | 16 |
| `--num-heads` | 2 |
| `--decoder-blocks` | 3 |
| `--feed-forward-size` | 32 |
| `--vocab-size` | 2 |

### CNN-specific flags

| Flag | Default |
|---|---|
| `--conv-layers` | 3 |
| `--pool-layers` | 3 |

### KAN-specific flags

| Flag | Default |
|---|---|
| `--grid-size` | 10 |

## `ecal profiles`

List all available hardware profiles:

```bash
ecal profiles
```

## Global flags

| Flag | Description |
|---|---|
| `--version` | Print the installed `ecal` version and exit |
| `-v`, `--verbose` | Enable verbose (debug-level) logging |
