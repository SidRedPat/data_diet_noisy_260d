import matplotlib.pyplot as plt


def plot_results(args, rec):
    # Create a figure with 2 rows and 2 columns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training and test loss
    ax1.plot(rec.train_step, rec.train_loss, label="Train Loss")
    ax1.plot(rec.test_step, rec.test_loss, label="Test Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Test Loss")
    ax1.legend()

    # Plot training and test accuracy
    ax2.plot(rec.train_step, rec.train_acc, label="Train Accuracy")
    ax2.plot(rec.test_step, rec.test_acc, label="Test Accuracy")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Test Accuracy")
    ax2.legend()

    if args.adaptive:
        # Plot pruning percentages over time
        ax3.plot(rec.prune_steps, rec.pruned_percentages)
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Pruning Percentage")
        ax3.set_title("Pruning Percentage Over Time")

        ax4.plot(rec.prune_steps, rec.pruned_sizes)
        ax4.set_xlabel("Steps")
        ax4.set_ylabel("Number of Pruned Samples")
        ax4.set_title("Pruned Samples Over Time")

    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/training_plots.png")
    plt.close()

    print("Training completed. Results and plots saved.")
