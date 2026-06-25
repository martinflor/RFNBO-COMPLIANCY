import pandas as pd
import plotly.express as px
from pathlib import Path

SUMMARY_CSV = Path('outputs/Belgium_hourly_ppa_sourcing_cost_summary.csv')
OUT_HTML = Path('outputs/Belgium_hourly_ppa_sourcing_cost_summary_plot.html')

if not SUMMARY_CSV.exists():
    raise SystemExit(f"Summary CSV not found: {SUMMARY_CSV}")

df = pd.read_csv(SUMMARY_CSV)
plot_df = df.copy()
plot_df['rfnbo_marker_size'] = plot_df['overall_rfnbo_pct'].fillna(4.0).clip(lower=4.0)

fig = px.scatter(
    plot_df,
    x='ppa_to_electrolyser_ratio',
    y='total_cost_eur',
    color='extra_margin_eur_mwh',
    symbol='technology',
    category_orders={'technology': ['Solar', 'Wind Offshore', 'Wind Onshore', 'Solar + Wind Offshore', 'Solar + Wind Onshore']},
    symbol_sequence=['circle', 'square', 'diamond', 'x', 'cross'],
    size='rfnbo_marker_size',
    size_max=22,
    color_continuous_scale='Viridis',
    hover_data={
        'technology': True,
        'extra_margin_eur_mwh': ':.1f',
        'incremental_cost_eur': ':.2f',
        'total_cost_eur': ':.2f',
        'overall_rfnbo_pct': ':.1f',
        'ppa_share_pct': ':.1f',
        'baseline_cost_eur': ':.2f',
        'incremental_cost_eur_mwh': ':.2f',
        'absolute_cost_eur_mwh': ':.2f',
        'ppa_capacity_mw': ':.1f',
        'electrolyser_mw': ':.1f',
    },
    title='PPA total cost vs production-to-consumption ratio - Belgium (summary)',
    labels={
        'ppa_to_electrolyser_ratio': 'Production to consumption ratio',
        'total_cost_eur': 'Total cost [€]',
        'extra_margin_eur_mwh': 'Extra margin [€/MWh]',
        'overall_rfnbo_pct': 'RFNBO [%]',
    },
)

fig.update_traces(marker=dict(opacity=0.9, line=dict(width=0.5, color='rgba(0,0,0,0.35)')))
fig.update_layout(
    template='plotly_white',
    margin=dict(l=70, r=170, t=70, b=100),
    legend=dict(
        title_text='Technology',
        x=1.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='rgba(0,0,0,0.1)',
        borderwidth=1,
    ),
    coloraxis_colorbar=dict(
        title='Extra margin [€/MWh]',
        x=1.18,
        thickness=18,
    ),
)
fig.update_xaxes(tickmode='linear', dtick=0.25)
fig.update_yaxes(ticksuffix=' €')
fig.write_html(OUT_HTML)
print(OUT_HTML)
