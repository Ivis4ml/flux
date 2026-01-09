"""
Command-line interface for Flux.

Provides commands for training, testing, and utility operations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable debug logging.
        log_file: Optional file to write logs to.
    """
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=format_str, handlers=handlers)


def cmd_train(args: argparse.Namespace) -> int:
    """Run training command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from flux.core.config import FluxConfig
    from flux.trainer import FluxTrainer

    # Load config
    if args.config:
        config = FluxConfig.from_yaml(args.config)
    else:
        config = FluxConfig(
            model_path=args.model_path,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    # Load prompts
    prompts = load_prompts(args.data)

    print(f"Starting training with {len(prompts)} prompts")
    print(f"Model: {config.model_path}")
    print(f"Output: {config.output_dir}")
    print(f"Steps: {config.num_steps}")

    # Create trainer and run
    trainer = FluxTrainer(config)

    try:
        result = trainer.fit(
            prompts=prompts,
            num_steps=args.num_steps or config.num_steps,
            checkpoint_interval=args.checkpoint_interval,
        )

        print("\n" + "=" * 50)
        print("Training Complete")
        print("=" * 50)
        print(f"Total steps: {result.total_steps}")
        print(f"Final loss: {result.final_loss:.4f}")
        print(f"Time: {result.total_time_seconds:.1f}s")
        print(f"Samples/s: {result.samples_per_second:.2f}")
        print(f"Checkpoint: {result.checkpoint_path}")

        return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted")
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return 1


def cmd_test(args: argparse.Namespace) -> int:
    """Run testing/scoring command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from flux.core.config import FluxConfig
    from flux.trainer import FluxTrainer

    # Load config
    if args.config:
        config = FluxConfig.from_yaml(args.config)
    else:
        config = FluxConfig(model_path=args.model_path)

    # Load prompts
    prompts = load_prompts(args.data)

    print(f"Testing on {len(prompts)} prompts")

    # Create trainer
    trainer = FluxTrainer(config)

    try:
        # Load checkpoint if provided
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        # Run scoring
        metrics = trainer.evaluate(prompts)

        print("\n" + "=" * 50)
        print("Test Results")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        # Save results if output specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults saved to {args.output}")

        trainer.teardown()
        return 0

    except Exception as e:
        print(f"\nTesting failed: {e}")
        trainer.teardown()
        return 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Run generation command (for testing).

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from flux.rollout.sglang_client import GenerationRequest, SGLangClient, SGLangConfig

    # Create client
    config = SGLangConfig(
        server_urls=[args.server_url] if args.server_url else None,
    )
    client = SGLangClient(config=config)

    print(f"Connecting to {client._server_urls}")

    try:
        import asyncio

        async def run_generation():
            await client.connect()

            # Read prompts
            if args.prompt:
                prompts = [args.prompt]
            elif args.data:
                prompts = load_prompts(args.data)[:args.num_samples]
            else:
                prompts = ["Hello, how are you?"]

            print(f"\nGenerating {len(prompts)} responses...")

            for i, prompt in enumerate(prompts):
                result = await client.generate(
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )

                print(f"\n--- Sample {i+1} ---")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Response: {result.response[:500]}...")
                print(f"Tokens: {result.completion_tokens}")
                print(f"Time: {result.elapsed_ms:.0f}ms")

            await client.close()

        asyncio.run(run_generation())
        return 0

    except Exception as e:
        print(f"\nGeneration failed: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show system and package info.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    import platform

    import torch

    print("Flux System Information")
    print("=" * 50)

    # Python info
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    # PyTorch info
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Flux version
    try:
        from flux import __version__
        print(f"\nFlux version: {__version__}")
    except ImportError:
        print("\nFlux version: development")

    # Check optional dependencies
    print("\nOptional dependencies:")
    deps = [
        ("transformers", "Transformers"),
        ("megatron", "Megatron-LM"),
        ("sglang", "SGLang"),
        ("uvloop", "uvloop"),
        ("wandb", "Weights & Biases"),
    ]
    for module, name in deps:
        try:
            __import__(module)
            print(f"  {name}: installed")
        except ImportError:
            print(f"  {name}: not installed")

    return 0


def load_prompts(path: str) -> list[str]:
    """Load prompts from file.

    Args:
        path: Path to file containing prompts.

    Returns:
        List of prompt strings.

    Supports:
    - .txt: One prompt per line
    - .json: List of strings or list of dicts with "prompt" key
    - .jsonl: One JSON object per line with "prompt" key
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    if path.suffix == ".txt":
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                return data
            return [x.get("prompt", x.get("text", str(x))) for x in data]
        return [data.get("prompt", str(data))]

    elif path.suffix == ".jsonl":
        prompts = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    prompts.append(obj.get("prompt", obj.get("text", str(obj))))
        return prompts

    else:
        # Try to read as text
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="flux",
        description="Flux: Adaptive RLHF Training Framework",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run RLHF training")
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path or HuggingFace ID",
    )
    train_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training prompts file",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    train_parser.add_argument(
        "--num-steps",
        type=int,
        help="Number of training steps",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Steps between checkpoints",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run model testing")
    test_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    test_parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path or HuggingFace ID",
    )
    test_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to load",
    )
    test_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test prompts file",
    )
    test_parser.add_argument(
        "--output",
        type=str,
        help="Path to save test results",
    )

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate responses (testing)")
    gen_parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="SGLang server URL",
    )
    gen_parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate",
    )
    gen_parser.add_argument(
        "--data",
        type=str,
        help="Path to prompts file",
    )
    gen_parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate",
    )
    gen_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    setup_logging(args.verbose, args.log_file)

    # Dispatch to command
    if args.command == "train":
        return cmd_train(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
