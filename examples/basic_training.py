#!/usr/bin/env python3
"""
Basic Flux Training Example

This example demonstrates a simple RLHF training loop using Flux
with length-based rewards.

Usage:
    python examples/basic_training.py --model Qwen/Qwen3-8B --output ./outputs
"""

import argparse
import logging
from pathlib import Path

from flux import FluxConfig, FluxTrainer
from flux.rewards import LengthReward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Basic Flux Training")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/basic",
        help="Output directory",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=200,
        help="Target response length for reward",
    )
    args = parser.parse_args()

    # Create configuration
    config = FluxConfig(
        model_path=args.model,
        output_dir=args.output,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=1e-6,
        seed=42,
    )

    # Create reward function
    # LengthReward encourages responses close to target length
    reward_fn = LengthReward(
        target_length=args.target_length,
        reward_type="gaussian",  # Gaussian around target
        scale=1.0,
    )

    # Sample training prompts
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "Describe the benefits of renewable energy.",
        "What are the key principles of good software design?",
        "Explain how neural networks work.",
        "Write instructions for making a paper airplane.",
        "Describe the water cycle in nature.",
        "What makes a good leader?",
        "Explain the difference between AI and human intelligence.",
        "Write a poem about the changing seasons.",
    ] * 100  # Repeat for training data

    logger.info(f"Starting training with {len(prompts)} prompts")
    logger.info(f"Config: {args.num_steps} steps, batch size {args.batch_size}")

    # Create trainer and train
    with FluxTrainer(config, reward_function=reward_fn) as trainer:
        # Add logging callback
        def log_callback(state, metrics):
            if state.step % 100 == 0:
                logger.info(
                    f"Step {state.step}: "
                    f"loss={metrics.get('loss', 0):.4f}, "
                    f"reward={metrics.get('mean_reward', 0):.4f}"
                )

        trainer.add_callback(log_callback)

        # Train
        result = trainer.fit(prompts=prompts)

        # Log results
        logger.info(f"Training complete!")
        logger.info(f"Total steps: {result.total_steps}")
        logger.info(f"Final loss: {result.final_loss:.4f}")
        logger.info(f"Samples/second: {result.samples_per_second:.2f}")

        # Save final checkpoint
        trainer.save_checkpoint(Path(args.output) / "final_checkpoint")

    logger.info(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
