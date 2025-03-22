import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Visualize logo groupings from a CSV file.")
    parser.add_argument("grouping_file", help="Path to the CSV file with grouping info (first column: filepath, second column: group)")
    parser.add_argument("--logo_dir", help="Directory where logos are stored", default="logos")
    parser.add_argument("--output_dir", help="Directory to save the visualization image", default="visualizations")
    parser.add_argument("--max_logos_per_row", type=int, default=10, help="Maximum logos per group in a row")
    parser.add_argument("--max_groups", type=int, default=30, help="Maximum groups to plot")
    parser.add_argument("--logo_size", type=int, default=80, help="Thumbnail size (width and height)")
    args = parser.parse_args()

    # Load the CSV file; assume first column is filepath and second is group
    df = pd.read_csv(args.grouping_file)
    file_col = df.columns[0]
    group_col = df.columns[1]

    print(f"Number of groups: {df[group_col].nunique()}")

    # Select the top groups by frequency (limited to max_groups)
    top_groups = df[group_col].value_counts().nlargest(args.max_groups).index
    df = df[df[group_col].isin(top_groups)]
    unique_groups = df[group_col].unique()
    num_groups = len(unique_groups)

    # Create subplots: one row per group
    fig, axes = plt.subplots(num_groups, 1, figsize=(args.max_logos_per_row, num_groups * 2), constrained_layout=True)
    if num_groups == 1:
        axes = [axes]

    for idx, group in enumerate(unique_groups):
        group_data = df[df[group_col] == group]
        logo_files = group_data[file_col].values[:args.max_logos_per_row]
        ax = axes[idx]
        ax.set_title(f"Group {group} (Total: {len(group_data)})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        ax.set_xlim(0, args.max_logos_per_row)
        ax.set_ylim(0, 1)

        for i, filename in enumerate(logo_files):
            logo_path = os.path.join(args.logo_dir, os.path.basename(filename))
            try:
                img = Image.open(logo_path)
                img.thumbnail((args.logo_size, args.logo_size))
                x_pos = i
                y_pos = 0
                ax.imshow(img, extent=[x_pos, x_pos + 1, y_pos, y_pos + 1])
                ax.text(x_pos + 0.5, y_pos - 0.2, os.path.basename(filename),
                        fontsize=8, rotation=45, ha="right", va="top", color="black")
            except Exception as e:
                print(f"Error loading {logo_path}: {e}")

    # Ensure output directory exists and save the figure
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(args.grouping_file))[0] + "_clusters_visualization.png"
    output_path = os.path.join(args.output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Cluster visualization saved as {output_path}")

if __name__ == "__main__":
    main()