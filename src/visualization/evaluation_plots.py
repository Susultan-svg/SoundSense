from pathlib import Path
import numpy as np
import plotly.graph_objects as go

def confusion_figure(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix,x=labels,y=labels,colorscale="Blues",text=matrix,texttemplate="%{text}",colorbar={"title":"Clips"}))
    fig.update_layout(title="Held-out predictions across five folds",xaxis_title="Predicted class",yaxis_title="True class",template="plotly_dark")
    return fig

def save_figure(fig, destination: Path) -> None:
    """Save Matplotlib figures as 300 dpi PNG and vector PDF."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"),bbox_inches="tight")
