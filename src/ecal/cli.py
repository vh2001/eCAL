"""Command-line interface for eCAL."""

import argparse
import json
import sys

from ecal._version import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ecal",
        description="eCAL: Estimate the energy cost of the AI lifecycle (J/bit)",
    )
    parser.add_argument("--version", action="version", version=f"ecal {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command")

    # --- estimate subcommand ---
    est = subparsers.add_parser("estimate", help="Estimate energy for a model")
    est.add_argument("--model", required=True, choices=["MLP", "CNN", "KAN", "Transformer"],
                     help="Model architecture")
    est.add_argument("--layers", type=int, default=3, help="Number of layers (default: 3)")
    est.add_argument("--din", type=int, default=10, help="Input dimension (default: 10)")
    est.add_argument("--dout", type=int, default=2, help="Output dimension (default: 2)")
    est.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    est.add_argument("--samples", type=int, default=1000, help="Number of training samples (default: 1000)")
    est.add_argument("--sample-size", type=int, default=10, help="Sample size / features (default: 10)")
    est.add_argument("--inferences", type=int, default=10000, help="Number of inferences (default: 10000)")
    est.add_argument("--hardware", type=str, default=None, help="Hardware profile name")
    est.add_argument("--json", action="store_true", dest="output_json", help="Output as JSON")

    # Transformer-specific
    est.add_argument("--context-length", type=int, default=10)
    est.add_argument("--embedding-size", type=int, default=16)
    est.add_argument("--num-heads", type=int, default=2)
    est.add_argument("--decoder-blocks", type=int, default=3)
    est.add_argument("--feed-forward-size", type=int, default=32)
    est.add_argument("--vocab-size", type=int, default=2)

    # CNN-specific
    est.add_argument("--conv-layers", type=int, default=3)
    est.add_argument("--pool-layers", type=int, default=3)

    # KAN-specific
    est.add_argument("--grid-size", type=int, default=10)

    # --- profiles subcommand ---
    subparsers.add_parser("profiles", help="List available hardware profiles")

    args = parser.parse_args(argv)

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.command == "estimate":
        _run_estimate(args)
    elif args.command == "profiles":
        _run_profiles()
    else:
        parser.print_help()
        sys.exit(1)


def _run_estimate(args):
    from ecal.api import estimate

    model_params = {}
    model = args.model.upper()

    if model == "MLP":
        model_params = {"num_layers": args.layers, "din": args.din, "dout": args.dout}
    elif model == "CNN":
        model_params = {
            "num_cnv_layers": args.conv_layers,
            "num_pool_layers": args.pool_layers,
            "i_r": args.sample_size, "i_c": 1,
            "k_r": 3, "k_c": 1, "c_in": 1,
        }
    elif model == "KAN":
        model_params = {
            "num_layers": args.layers, "grid_size": args.grid_size,
            "din": args.din, "dout": args.dout,
        }
    elif model == "TRANSFORMER":
        model_params = {
            "context_length": args.context_length,
            "embedding_size": args.embedding_size,
            "num_heads": args.num_heads,
            "num_decoder_blocks": args.decoder_blocks,
            "feed_forward_size": args.feed_forward_size,
            "vocab_size": args.vocab_size,
        }

    result = estimate(
        model_type=args.model,
        model_params=model_params,
        num_samples=args.samples,
        sample_size=args.sample_size,
        num_epochs=args.epochs,
        num_inferences=args.inferences,
        hardware=args.hardware,
    )

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print("\nEnergy Consumption Results (in Joules):")
        print("-" * 45)
        for key, value in result.items():
            if key == "total_bits":
                continue
            if isinstance(value, float):
                pct = value / result["total"] * 100 if result["total"] > 0 else 0
                print(f"  {key:20s}: {value:12.6f} J ({pct:6.2f}%)")
            else:
                print(f"  {key:20s}: {value}")
        print(f"\n  eCAL: {result['ecal_j_per_bit']:.10e} J/bit")


def _run_profiles():
    from ecal.hardware.profiles import list_profiles

    profiles = list_profiles()
    print(f"\nAvailable hardware profiles ({len(profiles)}):")
    print("-" * 70)
    print(f"  {'Name':<22s} {'FP32 FLOPS':>14s} {'TDP (W)':>10s} {'Device':<8s}")
    print("-" * 70)
    for key, p in sorted(profiles.items()):
        print(f"  {key:<22s} {p.flops_per_second_fp32:>14.2e} {p.tdp_watts:>10.0f} {p.device:<8s}")


if __name__ == "__main__":
    main()
