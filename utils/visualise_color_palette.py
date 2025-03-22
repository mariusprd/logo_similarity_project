import pandas as pd
import plotly.graph_objects as go

# Configuration
CSV_FILE = "logo_color_palette.csv"
LOGO_FILE = "logos/subway.co.id.png"  # Change to the logo you want to analyze

def load_logo_data(csv_file, logo_file):
    """Loads the color palette CSV and extracts data for one logo."""
    df = pd.read_csv(csv_file)
    logo_data = df[df["Filepath"] == logo_file]

    if logo_data.empty:
        print(f"❌ Error: No data found for {logo_file}")
        return None
    return logo_data.iloc[0]

def plot_color_pie_chart(logo_data, logo_file):
    """Creates a Pie Chart showing the color distribution of a logo."""
    colors = []
    confidences = []

    for i in range(1, 5):  # Extract up to 5 colors
        color_str = logo_data[f"Color_{i}"]
        confidence = logo_data[f"Confidence_{i}"]

        if pd.notna(color_str) and pd.notna(confidence):
            rgb = eval(color_str)  # Convert "(R, G, B)" string to tuple
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)  # Convert to HEX
            colors.append(hex_color)
            confidences.append(confidence)

    # Create Pie Chart
    fig = go.Figure(data=[go.Pie(
        labels=colors,
        values=confidences,
        marker=dict(colors=colors),
        textinfo='percent+label'
    )])

    # Configure layout
    fig.update_layout(
        title=f"Color Distribution for {logo_file.split('/')[-1]}"
    )

    fig.show()

# Load data and visualize
logo_data = load_logo_data(CSV_FILE, LOGO_FILE)
if logo_data is not None:
    plot_color_pie_chart(logo_data, LOGO_FILE)