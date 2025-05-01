import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def parse_tensorboard_events(event_files: list) -> tuple[list[float], list[float]]:
    """Parse TensorBoard event files for loss and step values."""
    steps = []
    losses = []
    
    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        
        # Try different tags for loss values
        tags = event_acc.Tags()['scalars']
        loss_tags = [tag for tag in tags if 'loss' in tag.lower()]
        
        if not loss_tags:
            continue
            
        # Use the first loss tag found
        loss_tag = loss_tags[0]
        events = event_acc.Scalars(loss_tag)
        
        for event in events:
            steps.append(event.step)
            losses.append(event.value)
    
    # Sort by steps
    if steps:
        sorted_pairs = sorted(zip(steps, losses))
        steps, losses = zip(*sorted_pairs)
        steps = list(steps)
        losses = list(losses)
    
    return steps, losses

def plot_training_loss(steps: list[float], losses: list[float], output_file: str = None):
    """Plot training loss over steps."""
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, label="Training Loss", color="blue", linewidth=2)
    
    # Add markers for every 1000 steps if we have enough points
    if len(steps) > 1000:
        marker_steps = steps[::1000]
        marker_losses = losses[::1000]
        plt.scatter(marker_steps, marker_losses, color='red', s=50, zorder=5)
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add min/max annotations
    min_loss = min(losses)
    max_loss = max(losses)
    plt.annotate(f'Min: {min_loss:.4f}', 
                xy=(steps[losses.index(min_loss)], min_loss),
                xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Max: {max_loss:.4f}',
                xy=(steps[losses.index(max_loss)], max_loss),
                xytext=(10, -10), textcoords='offset points')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot training loss from TensorBoard logs")
    parser.add_argument("--log_dir", type=str, default="results/megatron_gpt_peft_adapter_tuning",
                      help="Directory containing TensorBoard event files")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save the plot (optional, will show plot if not provided)")
    args = parser.parse_args()
    
    # Find all event files
    event_files = glob.glob(f"{args.log_dir}/version_*/events.*")
    if not event_files:
        print(f"Error: No TensorBoard event files found in '{args.log_dir}'")
        return
    
    # Parse and plot
    steps, losses = parse_tensorboard_events(event_files)
    
    if not steps:
        print("No loss data found in the TensorBoard event files")
        print("Please check if the files contain loss metrics")
        return
    
    print(f"Found {len(steps)} data points")
    print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")
    print(f"Final loss: {losses[-1]:.4f} at step {steps[-1]}")
    
    plot_training_loss(steps, losses, args.output)

if __name__ == "__main__":
    main() 