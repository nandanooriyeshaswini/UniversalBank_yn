import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_data, train_models, get_prescriptive_insights, prepare_features

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UniversalBank Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME ──────────────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#0A2342',
    'accent': '#E8B84B',
    'accept': '#2ECC71',
    'reject': '#E74C3C',
    'neutral': '#3498DB',
    'bg': '#F8F9FA',
    'card': '#FFFFFF',
    'text': '#1A1A2E',
    'muted': '#6C757D',
    'grad1': '#0A2342',
    'grad2': '#1B4F8A',
}

PALETTE = [COLORS['primary'], COLORS['accent'], COLORS['accept'], COLORS['reject'],
           COLORS['neutral'], '#9B59B6', '#E67E22', '#1ABC9C']

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {COLORS['bg']};
        color: {COLORS['text']};
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {COLORS['grad1']} 0%, {COLORS['grad2']} 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    .main-header h1 {{
        font-family: 'Playfair Display', serif;
        color: white;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    .main-header p {{
        color: rgba(255,255,255,0.75);
        font-size: 0.95rem;
        margin: 0.25rem 0 0;
    }}
    .bank-icon {{
        font-size: 3.5rem;
        line-height: 1;
    }}
    
    .kpi-card {{
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #E8ECF0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        height: 100%;
    }}
    .kpi-label {{
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {COLORS['muted']};
        margin-bottom: 0.5rem;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['primary']};
        line-height: 1.1;
        font-family: 'Playfair Display', serif;
    }}
    .kpi-delta {{
        font-size: 0.82rem;
        margin-top: 0.4rem;
        color: {COLORS['muted']};
    }}
    .kpi-accent {{
        border-left: 4px solid {COLORS['accent']};
    }}
    .kpi-accept {{
        border-left: 4px solid {COLORS['accept']};
    }}
    .kpi-reject {{
        border-left: 4px solid {COLORS['reject']};
    }}
    .kpi-blue {{
        border-left: 4px solid {COLORS['neutral']};
    }}
    
    .section-header {{
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: {COLORS['primary']};
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS['accent']};
        margin-bottom: 1.5rem;
        margin-top: 1rem;
    }}
    
    .insight-card {{
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #E8ECF0;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }}
    .insight-title {{
        font-weight: 600;
        color: {COLORS['primary']};
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }}
    .insight-text {{
        color: {COLORS['muted']};
        font-size: 0.83rem;
    }}
    .badge-high {{
        background: #FFF3CD;
        color: #856404;
        border-radius: 99px;
        padding: 0.1rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 600;
    }}
    .badge-very-high {{
        background: #D4EDDA;
        color: #155724;
        border-radius: 99px;
        padding: 0.1rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 600;
    }}
    .badge-medium {{
        background: #D1ECF1;
        color: #0C5460;
        border-radius: 99px;
        padding: 0.1rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 600;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: #EEF1F5;
        padding: 4px;
        border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        font-size: 0.875rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: white;
        color: {COLORS['primary']};
        font-weight: 600;
    }}
    
    .stSidebar {{
        background: {COLORS['primary']};
    }}
    section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, {COLORS['primary']} 0%, #0F2F5A 100%);
        padding-top: 1rem;
    }}
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {{
        color: {COLORS['accent']} !important;
        font-family: 'Playfair Display', serif;
    }}
    
    .model-card {{
        background: white;
        border-radius: 14px;
        padding: 1.4rem;
        border: 1px solid #E8ECF0;
        text-align: center;
        height: 100%;
        transition: box-shadow 0.2s;
    }}
    .model-card:hover {{
        box-shadow: 0 6px 24px rgba(10,35,66,0.12);
    }}
    .model-card.best {{
        border: 2px solid {COLORS['accent']};
        background: linear-gradient(135deg, #FFFBF0, white);
    }}
    .model-name {{
        font-weight: 600;
        color: {COLORS['primary']};
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }}
    .model-auc {{
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
        color: {COLORS['primary']};
    }}
    .model-label {{
        font-size: 0.72rem;
        color: {COLORS['muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    
    .prescriptive-card {{
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #E8ECF0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border-left: 5px solid {COLORS['accent']};
    }}
    
    div[data-testid="stHorizontalBlock"] {{
        gap: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


# ─── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data("UniversalBank.csv")

@st.cache_resource
def get_models(df_hash):
    df = get_data()
    return train_models(df)

df_full = get_data()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 UniversalBank")
    st.markdown("### Dashboard Filters")
    st.markdown("---")

    income_min, income_max = int(df_full['Income'].min()), int(df_full['Income'].max())
    income_range = st.slider(
        "💰 Income Range (K$)",
        min_value=income_min, max_value=income_max,
        value=(income_min, income_max)
    )

    education_opts = {'All': [1, 2, 3], 'Undergrad': [1], 'Graduate': [2], 'Advanced': [3]}
    edu_sel = st.selectbox("🎓 Education Level", list(education_opts.keys()))

    family_opts = st.multiselect(
        "👨‍👩‍👧 Family Size",
        options=[1, 2, 3, 4], default=[1, 2, 3, 4],
        format_func=lambda x: f"{x} member{'s' if x > 1 else ''}"
    )

    loan_filter = st.radio(
        "🎯 Show Customers",
        ['All', 'Accepted Loan', 'Rejected Loan']
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Customers", f"{len(df_full):,}")
    st.metric("Loan Acceptance Rate", f"{df_full['Personal Loan'].mean()*100:.1f}%")
    st.markdown("---")
    st.markdown("<p style='color:rgba(255,255,255,0.5);font-size:0.75rem;'>UniversalBank Campaign Analysis • 2024</p>", unsafe_allow_html=True)

# ─── FILTER DATA ────────────────────────────────────────────────────────────────
df = df_full.copy()
df = df[(df['Income'] >= income_range[0]) & (df['Income'] <= income_range[1])]
df = df[df['Education'].isin(education_opts[edu_sel])]
if family_opts:
    df = df[df['Family'].isin(family_opts)]
if loan_filter == 'Accepted Loan':
    df = df[df['Personal Loan'] == 1]
elif loan_filter == 'Rejected Loan':
    df = df[df['Personal Loan'] == 0]

# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <div class="bank-icon">🏦</div>
    <div>
        <h1>UniversalBank Loan Analytics</h1>
        <p>Personal Loan Campaign · Descriptive · Diagnostic · Predictive · Prescriptive Analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI ROW ────────────────────────────────────────────────────────────────────
accepted = df[df['Personal Loan'] == 1]
rejected = df[df['Personal Loan'] == 0]
accept_rate = df['Personal Loan'].mean() * 100 if len(df) > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
kpis = [
    ("Total Customers", f"{len(df):,}", f"Filtered from {len(df_full):,}", "kpi-accent"),
    ("Loan Acceptors", f"{len(accepted):,}", f"{accept_rate:.1f}% acceptance rate", "kpi-accept"),
    ("Loan Rejectors", f"{len(rejected):,}", f"{100-accept_rate:.1f}% rejection rate", "kpi-reject"),
    ("Avg Income (K$)", f"${df['Income'].mean():.0f}K" if len(df) else "N/A", f"Acceptors: ${accepted['Income'].mean():.0f}K" if len(accepted) else "", "kpi-blue"),
    ("Avg CC Spend", f"${df['CCAvg'].mean():.2f}K" if len(df) else "N/A", f"Acceptors: ${accepted['CCAvg'].mean():.2f}K" if len(accepted) else "", "kpi-accent"),
]
for col, (label, val, delta, cls) in zip([col1, col2, col3, col4, col5], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Analysis",
    "🤖 Predictive Model",
    "💡 Prescriptive Insights",
    "🎯 Predict a Customer"
])


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">1. Loan Acceptance Overview</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        # Donut chart with drill-down
        loan_counts = df['Personal Loan'].value_counts()
        labels = ['Rejected', 'Accepted']
        values = [loan_counts.get(0, 0), loan_counts.get(1, 0)]
        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker=dict(colors=[COLORS['reject'], COLORS['accept']],
                        line=dict(color='white', width=3)),
            textinfo='label+percent',
            textfont=dict(size=13),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Proportion: %{percent}<extra></extra>'
        ))
        fig_donut.add_annotation(
            text=f"<b>{accept_rate:.1f}%</b><br>Accept Rate",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['primary']),
            align='center'
        )
        fig_donut.update_layout(
            title='Loan Acceptance Distribution',
            height=380, showlegend=True,
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        # Interactive drill-down: Education × Loan
        drill_by = st.selectbox(
            "🔎 Drill-down Breakdown by:",
            ['Education Level', 'Income Group', 'Family Size', 'Age Group'],
            key='drilldown'
        )
        drill_map = {
            'Education Level': ('Education_Label', ['Undergrad', 'Graduate', 'Advanced']),
            'Income Group': ('Income_Group', ['<50K', '50-100K', '100-150K', '150-200K', '200K+']),
            'Family Size': ('Family_Label', ['1', '2', '3', '4+']),
            'Age Group': ('Age_Group', ['<30', '30-40', '40-50', '50-60', '60+'])
        }
        col_key, order = drill_map[drill_by]
        drill_df = df.groupby([col_key, 'Personal Loan']).size().reset_index(name='Count')
        drill_df['Loan Status'] = drill_df['Personal Loan'].map({0: 'Rejected', 1: 'Accepted'})
        drill_df[col_key] = pd.Categorical(drill_df[col_key], categories=order, ordered=True)
        drill_df = drill_df.sort_values(col_key)

        fig_drill = px.bar(
            drill_df, x=col_key, y='Count', color='Loan Status',
            color_discrete_map={'Accepted': COLORS['accept'], 'Rejected': COLORS['reject']},
            barmode='group', title=f'Loan Acceptance by {drill_by}',
            text='Count'
        )
        fig_drill.update_traces(textposition='outside', textfont_size=11)
        fig_drill.update_layout(
            height=380, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20),
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center')
        )
        fig_drill.update_yaxes(gridcolor='#F0F0F0')
        st.plotly_chart(fig_drill, use_container_width=True)

    # ─── Summary Statistics ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">2. Summary Statistics of Numerical Variables</div>', unsafe_allow_html=True)

    num_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']
    summary = df.groupby('Personal Loan')[num_cols].describe().T
    summary.columns = ['Rejected — ' + c for c in summary.columns.get_level_values(1)] if isinstance(summary.columns, pd.MultiIndex) else summary.columns

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Customers Who Accepted the Loan (Personal Loan = 1)**")
        if len(accepted):
            st.dataframe(
                accepted[num_cols].describe().round(2).rename(index={'count': 'Count', 'mean': 'Mean', 'std': 'Std Dev', 'min': 'Min', '25%': 'Q1', '50%': 'Median', '75%': 'Q3', 'max': 'Max'}),
                use_container_width=True
            )
    with col_s2:
        st.markdown("**Customers Who Rejected the Loan (Personal Loan = 0)**")
        if len(rejected):
            st.dataframe(
                rejected[num_cols].describe().round(2).rename(index={'count': 'Count', 'mean': 'Mean', 'std': 'Std Dev', 'min': 'Min', '25%': 'Q1', '50%': 'Median', '75%': 'Q3', 'max': 'Max'}),
                use_container_width=True
            )

    # ─── Distribution Analysis ────────────────────────────────────────────────
    st.markdown('<div class="section-header">3. Distribution Analysis of Key Numeric Features</div>', unsafe_allow_html=True)

    dist_col = st.selectbox("Select Variable for Distribution", num_cols, key='dist_var')
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig_hist = go.Figure()
        for loan_val, label, color in [(0, 'Rejected', COLORS['reject']), (1, 'Accepted', COLORS['accept'])]:
            subset = df[df['Personal Loan'] == loan_val][dist_col].dropna()
            fig_hist.add_trace(go.Histogram(
                x=subset, name=label, opacity=0.7,
                marker_color=color, nbinsx=30,
                hovertemplate=f'<b>{label}</b><br>{dist_col}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
        fig_hist.update_layout(
            barmode='overlay', title=f'Distribution of {dist_col} by Loan Status',
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'),
            legend=dict(orientation='h', y=1.05),
            margin=dict(t=50, b=20)
        )
        fig_hist.update_yaxes(gridcolor='#F0F0F0')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d2:
        fig_box = go.Figure()
        for loan_val, label, color in [(0, 'Rejected', COLORS['reject']), (1, 'Accepted', COLORS['accept'])]:
            subset = df[df['Personal Loan'] == loan_val][dist_col].dropna()
            fig_box.add_trace(go.Box(
                y=subset, name=label, marker_color=color,
                boxmean='sd', line=dict(width=2)
            ))
        fig_box.update_layout(
            title=f'Box Plot: {dist_col} by Loan Status',
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20)
        )
        fig_box.update_yaxes(gridcolor='#F0F0F0')
        st.plotly_chart(fig_box, use_container_width=True)

    # ─── Correlation Heatmap ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">4. Correlation Heatmap</div>', unsafe_allow_html=True)

    corr_cols = ['Age', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
                 'Securities Account', 'CD Account', 'Online', 'CreditCard', 'Personal Loan']
    corr_matrix = df[corr_cols].corr().round(2)

    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, '#E74C3C'], [0.5, 'white'], [1, '#0A2342']],
        zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x} × %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    fig_corr.update_layout(
        title='Correlation Matrix — All Features',
        height=520, plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='DM Sans'), margin=dict(t=50, b=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Key correlation callout
    corr_with_target = corr_matrix['Personal Loan'].drop('Personal Loan').abs().sort_values(ascending=False)
    top3 = corr_with_target.head(3)
    cols_corr = st.columns(3)
    for i, (feat, corr_val) in enumerate(top3.items()):
        with cols_corr[i]:
            st.markdown(f"""
            <div class="kpi-card kpi-accent">
                <div class="kpi-label">Top Correlator #{i+1}</div>
                <div class="kpi-value">{corr_val:.2f}</div>
                <div class="kpi-delta">{feat} ↔ Personal Loan</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">What Differentiates Loan Acceptors from Rejectors?</div>', unsafe_allow_html=True)

    # ─── Income, CCAvg, Mortgage comparison ──────────────────────────────────
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Scatter: Income vs CCAvg colored by loan
        fig_scatter = px.scatter(
            df.sample(min(len(df), 2000), random_state=42),
            x='Income', y='CCAvg',
            color=df.sample(min(len(df), 2000), random_state=42)['Personal Loan'].map({0: 'Rejected', 1: 'Accepted'}),
            color_discrete_map={'Accepted': COLORS['accept'], 'Rejected': COLORS['reject']},
            opacity=0.6, size_max=8,
            title='Income vs Credit Card Spending by Loan Status',
            labels={'CCAvg': 'CC Avg Spend (K$)', 'color': 'Loan Status'}
        )
        fig_scatter.update_layout(
            height=380, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20)
        )
        fig_scatter.update_xaxes(gridcolor='#F0F0F0')
        fig_scatter.update_yaxes(gridcolor='#F0F0F0')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_v2:
        # Loan acceptance rate by income group
        income_group_df = df_full.copy()
        income_group_df['Income_Bucket'] = pd.cut(
            income_group_df['Income'],
            bins=[0, 40, 80, 120, 160, 200, 300],
            labels=['0-40K', '40-80K', '80-120K', '120-160K', '160-200K', '200K+']
        )
        income_rate = income_group_df.groupby('Income_Bucket')['Personal Loan'].agg(['mean', 'count']).reset_index()
        income_rate.columns = ['Income Group', 'Acceptance Rate', 'Count']
        income_rate['Acceptance Rate'] = (income_rate['Acceptance Rate'] * 100).round(1)

        fig_income_rate = px.bar(
            income_rate, x='Income Group', y='Acceptance Rate',
            color='Acceptance Rate',
            color_continuous_scale=[[0, '#E8ECF0'], [1, COLORS['primary']]],
            title='Loan Acceptance Rate by Income Bracket',
            text='Acceptance Rate'
        )
        fig_income_rate.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_income_rate.update_layout(
            height=380, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), coloraxis_showscale=False,
            margin=dict(t=50, b=20)
        )
        fig_income_rate.update_yaxes(gridcolor='#F0F0F0', title='Acceptance Rate (%)')
        st.plotly_chart(fig_income_rate, use_container_width=True)

    # ─── Education & Banking Services ────────────────────────────────────────
    col_v3, col_v4 = st.columns(2)

    with col_v3:
        edu_rate = df_full.groupby('Education_Label')['Personal Loan'].mean().reset_index()
        edu_rate.columns = ['Education', 'Acceptance Rate']
        edu_rate['Acceptance Rate'] = (edu_rate['Acceptance Rate'] * 100).round(1)
        edu_rate['Education'] = pd.Categorical(edu_rate['Education'], ['Undergrad', 'Graduate', 'Advanced'], ordered=True)
        edu_rate = edu_rate.sort_values('Education')

        fig_edu = px.bar(
            edu_rate, x='Education', y='Acceptance Rate',
            color='Education',
            color_discrete_sequence=[COLORS['neutral'], COLORS['accent'], COLORS['primary']],
            title='Loan Acceptance Rate by Education Level',
            text='Acceptance Rate'
        )
        fig_edu.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_edu.update_layout(
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), showlegend=False, margin=dict(t=50, b=20)
        )
        fig_edu.update_yaxes(gridcolor='#F0F0F0', title='Acceptance Rate (%)')
        st.plotly_chart(fig_edu, use_container_width=True)

    with col_v4:
        # Banking services and loan acceptance
        services = ['Securities Account', 'CD Account', 'Online', 'CreditCard']
        service_labels = ['Securities\nAccount', 'CD Account', 'Online\nBanking', 'Credit Card']
        service_rates = []
        for s in services:
            for val in [0, 1]:
                subset = df_full[df_full[s] == val]
                rate = subset['Personal Loan'].mean() * 100
                service_rates.append({
                    'Service': s.replace('\n', ' '),
                    'Has Service': 'Yes' if val == 1 else 'No',
                    'Acceptance Rate': round(rate, 1)
                })
        svc_df = pd.DataFrame(service_rates)

        fig_svc = px.bar(
            svc_df, x='Service', y='Acceptance Rate', color='Has Service',
            color_discrete_map={'Yes': COLORS['accept'], 'No': COLORS['reject']},
            barmode='group', title='Banking Services vs Loan Acceptance Rate',
            text='Acceptance Rate'
        )
        fig_svc.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_svc.update_layout(
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20),
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center')
        )
        fig_svc.update_yaxes(gridcolor='#F0F0F0', title='Acceptance Rate (%)')
        st.plotly_chart(fig_svc, use_container_width=True)

    # ─── Age Distribution ─────────────────────────────────────────────────────
    col_v5, col_v6 = st.columns(2)

    with col_v5:
        fig_age = go.Figure()
        for loan_val, label, color in [(0, 'Rejected', COLORS['reject']), (1, 'Accepted', COLORS['accept'])]:
            subset = df_full[df_full['Personal Loan'] == loan_val]['Age']
            fig_age.add_trace(go.Histogram(
                x=subset, name=label, opacity=0.7,
                marker_color=color, nbinsx=25
            ))
        fig_age.update_layout(
            barmode='overlay', title='Age Distribution by Loan Status',
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'),
            legend=dict(orientation='h', y=1.05),
            margin=dict(t=50, b=20)
        )
        fig_age.update_yaxes(gridcolor='#F0F0F0')
        st.plotly_chart(fig_age, use_container_width=True)

    with col_v6:
        # CC Avg vs Loan grouped violin
        fig_violin = go.Figure()
        for loan_val, label, color in [(0, 'Rejected', COLORS['reject']), (1, 'Accepted', COLORS['accept'])]:
            subset = df_full[df_full['Personal Loan'] == loan_val]['CCAvg']
            fig_violin.add_trace(go.Violin(
                y=subset, name=label, fillcolor=color,
                line_color='white', meanline_visible=True,
                opacity=0.85
            ))
        fig_violin.update_layout(
            title='Credit Card Spending Distribution by Loan Status',
            height=360, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20)
        )
        fig_violin.update_yaxes(gridcolor='#F0F0F0', title='CC Avg Spend (K$)')
        st.plotly_chart(fig_violin, use_container_width=True)

    # ─── Key Diagnostic Insights ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Diagnostic Findings</div>', unsafe_allow_html=True)

    acc = df_full[df_full['Personal Loan'] == 1]
    rej = df_full[df_full['Personal Loan'] == 0]

    findings = [
        ("💰 Income Gap", f"Acceptors earn on average **${acc['Income'].mean():.0f}K** vs **${rej['Income'].mean():.0f}K** for rejectors — a {(acc['Income'].mean()/rej['Income'].mean()-1)*100:.0f}% difference"),
        ("💳 Credit Card Spending", f"Acceptors spend **${acc['CCAvg'].mean():.2f}K/month** on credit cards vs **${rej['CCAvg'].mean():.2f}K** — nearly {acc['CCAvg'].mean()/rej['CCAvg'].mean():.1f}x more"),
        ("🏠 Mortgage", f"Acceptors hold higher mortgages: **${acc['Mortgage'].mean():.0f}K** vs **${rej['Mortgage'].mean():.0f}K**, indicating higher financial engagement"),
        ("🏦 CD Account", f"Customers with a CD account accept at **{df_full[df_full['CD Account']==1]['Personal Loan'].mean()*100:.1f}%** rate vs only **{df_full[df_full['CD Account']==0]['Personal Loan'].mean()*100:.1f}%** without one"),
        ("🎓 Education", f"Advanced degree holders accept at **{df_full[df_full['Education']==3]['Personal Loan'].mean()*100:.1f}%** vs Undergrads at **{df_full[df_full['Education']==1]['Personal Loan'].mean()*100:.1f}%**"),
    ]

    for title, text in findings:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Classification Model Performance</div>', unsafe_allow_html=True)

    with st.spinner("Training classification models... This may take a moment."):
        model_results, best_model_name, X_test, y_test, scaler, feature_cols = get_models(len(df_full))

    st.success(f"✅ Models trained! Best model: **{best_model_name}** (AUC = {model_results[best_model_name]['auc']:.4f})")

    # ─── Model Comparison Cards ───────────────────────────────────────────────
    cols_m = st.columns(len(model_results))
    for col, (name, res) in zip(cols_m, model_results.items()):
        with col:
            is_best = name == best_model_name
            best_badge = "⭐ Best" if is_best else ""
            st.markdown(f"""
            <div class="model-card {'best' if is_best else ''}">
                <div class="model-name">{best_badge} {name}</div>
                <div class="model-auc">{res['auc']:.3f}</div>
                <div class="model-label">ROC-AUC</div>
                <br>
                <div style="font-size:0.8rem;color:#555">
                    Accuracy: <b>{res['accuracy']*100:.1f}%</b><br>
                    F1 Score: <b>{res['f1']:.3f}</b><br>
                    CV AUC: <b>{res['cv_auc_mean']:.3f} ± {res['cv_auc_std']:.3f}</b>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── ROC Curves ───────────────────────────────────────────────────────────
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='gray', width=1),
            name='Random (AUC=0.5)', showlegend=True
        ))
        for name, res in model_results.items():
            fig_roc.add_trace(go.Scatter(
                x=res['fpr'], y=res['tpr'], mode='lines',
                name=f"{name} ({res['auc']:.3f})",
                line=dict(width=2.5)
            ))
        fig_roc.update_layout(
            title='ROC Curves — All Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=420, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20),
            legend=dict(x=0.5, y=0.05, bgcolor='rgba(255,255,255,0.8)')
        )
        fig_roc.update_xaxes(gridcolor='#F0F0F0')
        fig_roc.update_yaxes(gridcolor='#F0F0F0')
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_r2:
        # Best model confusion matrix
        best_res = model_results[best_model_name]
        cm = best_res['confusion']
        cm_labels = ['Rejected', 'Accepted']
        fig_cm = go.Figure(go.Heatmap(
            z=cm[::-1], x=cm_labels, y=cm_labels[::-1],
            colorscale=[[0, '#F8F9FA'], [1, COLORS['primary']]],
            text=cm[::-1], texttemplate='<b>%{text}</b>',
            textfont=dict(size=18),
            showscale=False,
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>'
        ))
        fig_cm.update_layout(
            title=f'Confusion Matrix — {best_model_name}',
            xaxis_title='Predicted', yaxis_title='Actual',
            height=420, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ─── Feature Importance ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

    col_fi1, col_fi2 = st.columns(2)
    models_with_fi = {k: v for k, v in model_results.items() if 'feature_importance' in v}

    for col, (name, res) in zip([col_fi1, col_fi2], models_with_fi.items()):
        with col:
            fi = res['feature_importance']
            fig_fi = px.bar(
                x=fi.values, y=fi.index, orientation='h',
                color=fi.values,
                color_continuous_scale=[[0, '#E8ECF0'], [1, COLORS['primary']]],
                title=f'Feature Importance — {name}',
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            fig_fi.update_layout(
                height=380, plot_bgcolor='white', paper_bgcolor='white',
                font=dict(family='DM Sans'), coloraxis_showscale=False,
                yaxis=dict(autorange='reversed'), margin=dict(t=50, b=20)
            )
            fig_fi.update_xaxes(gridcolor='#F0F0F0')
            st.plotly_chart(fig_fi, use_container_width=True)

    # ─── Classification Report ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Detailed Classification Report</div>', unsafe_allow_html=True)

    selected_model = st.selectbox("Select Model to View Report", list(model_results.keys()), key='report_model')
    report_dict = model_results[selected_model]['report']
    report_df = pd.DataFrame(report_dict).T.iloc[:-1].round(3)
    report_df = report_df.rename(index={'0': 'Rejected', '1': 'Accepted', 'macro avg': 'Macro Avg', 'weighted avg': 'Weighted Avg'})
    st.dataframe(report_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Prescriptive Recommendations</div>', unsafe_allow_html=True)
    st.markdown("Based on the diagnostic findings and model outputs, here are **actionable strategies** to improve personal loan campaign performance.")

    insights = get_prescriptive_insights(df_full)

    for ins in insights:
        badge_class = 'badge-very-high' if ins['impact'] == 'Very High' else ('badge-high' if ins['impact'] == 'High' else 'badge-medium')
        st.markdown(f"""
        <div class="prescriptive-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem">
                <span style="font-weight:700;color:{COLORS['primary']};font-size:1rem">🎯 {ins['category']}</span>
                <span class="{badge_class}">Impact: {ins['impact']}</span>
            </div>
            <p style="font-weight:600;color:{COLORS['text']};margin:0.3rem 0">{ins['insight']}</p>
            <p style="color:{COLORS['muted']};font-size:0.85rem;margin:0.2rem 0">{ins['detail']}</p>
            <div style="background:#F8F9FA;border-radius:8px;padding:0.5rem 0.8rem;margin-top:0.6rem">
                <span style="font-size:0.82rem;font-weight:600;color:{COLORS['primary']}">→ Action: </span>
                <span style="font-size:0.82rem;color:{COLORS['text']}">{ins['action']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ─── Campaign Simulation ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Campaign Targeting Simulation</div>', unsafe_allow_html=True)
    st.markdown("Simulate targeted campaign impact by selecting customer segments to focus on.")

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sim_min_income = st.slider("Minimum Income (K$)", 0, 200, 80, key='sim_income')
    with col_sim2:
        sim_edu = st.multiselect("Target Education Levels", ['Undergrad', 'Graduate', 'Advanced'],
                                  default=['Graduate', 'Advanced'], key='sim_edu')
    with col_sim3:
        sim_cd = st.checkbox("Must have CD Account", value=False, key='sim_cd')

    sim_df = df_full.copy()
    sim_df = sim_df[sim_df['Income'] >= sim_min_income]
    edu_map = {'Undergrad': 1, 'Graduate': 2, 'Advanced': 3}
    if sim_edu:
        sim_df = sim_df[sim_df['Education'].isin([edu_map[e] for e in sim_edu])]
    if sim_cd:
        sim_df = sim_df[sim_df['CD Account'] == 1]

    total_targeted = len(sim_df)
    expected_acceptors = sim_df['Personal Loan'].sum()
    sim_accept_rate = sim_df['Personal Loan'].mean() * 100 if total_targeted > 0 else 0
    baseline_rate = df_full['Personal Loan'].mean() * 100
    lift = sim_accept_rate / baseline_rate if baseline_rate > 0 else 1

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    sim_kpis = [
        ("Customers Targeted", f"{total_targeted:,}", "kpi-blue"),
        ("Expected Acceptors", f"{expected_acceptors:,}", "kpi-accept"),
        ("Simulated Accept Rate", f"{sim_accept_rate:.1f}%", "kpi-accent"),
        ("Campaign Lift", f"{lift:.2f}×", "kpi-accept"),
    ]
    for col, (label, val, cls) in zip([col_s1, col_s2, col_s3, col_s4], sim_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-delta">vs baseline {baseline_rate:.1f}%</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT A CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Predict Loan Acceptance for a New Customer</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details below to predict the likelihood of loan acceptance using the best-performing model.")

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        p_age = st.slider("Age", 20, 70, 40, key='p_age')
        p_experience = st.slider("Years of Experience", 0, 45, 15, key='p_exp')
        p_income = st.slider("Annual Income (K$)", 8, 225, 80, key='p_inc')
        p_family = st.selectbox("Family Size", [1, 2, 3, 4], index=1, key='p_fam')

    with col_p2:
        p_ccavg = st.slider("Monthly CC Spend (K$)", 0.0, 10.0, 1.5, step=0.1, key='p_cc')
        p_edu = st.selectbox("Education Level", ['Undergrad (1)', 'Graduate (2)', 'Advanced (3)'], key='p_edu')
        p_mortgage = st.slider("Mortgage Value (K$)", 0, 635, 0, key='p_mort')

    with col_p3:
        p_sec = st.checkbox("Has Securities Account", key='p_sec')
        p_cd = st.checkbox("Has CD Account", key='p_cd')
        p_online = st.checkbox("Uses Online Banking", value=True, key='p_online')
        p_cc = st.checkbox("Has Credit Card (UB)", key='p_cc2')

    edu_num = {'Undergrad (1)': 1, 'Graduate (2)': 2, 'Advanced (3)': 3}[p_edu]

    if st.button("🔮 Predict Loan Acceptance", use_container_width=True):
        customer = np.array([[p_age, p_experience, p_income, p_family, p_ccavg,
                              edu_num, p_mortgage, int(p_sec), int(p_cd), int(p_online), int(p_cc)]])

        best_res = model_results[best_model_name]
        best_model = best_res['model']

        if best_model_name in ['Logistic Regression', 'KNN']:
            customer_scaled = scaler.transform(customer)
            prob = best_model.predict_proba(customer_scaled)[0][1]
            pred = best_model.predict(customer_scaled)[0]
        else:
            prob = best_model.predict_proba(customer)[0][1]
            pred = best_model.predict(customer)[0]

        color = COLORS['accept'] if pred == 1 else COLORS['reject']
        verdict = "✅ Likely to Accept" if pred == 1 else "❌ Likely to Reject"
        confidence = prob * 100 if pred == 1 else (1 - prob) * 100

        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:2rem;border:2px solid {color};text-align:center;margin-top:1rem">
            <div style="font-size:1.1rem;font-weight:600;color:{COLORS['muted']};margin-bottom:0.5rem">Prediction Result — {best_model_name}</div>
            <div style="font-size:2.5rem;font-weight:700;color:{color};font-family:'Playfair Display',serif">{verdict}</div>
            <div style="font-size:1rem;color:{COLORS['muted']};margin-top:0.5rem">Probability of Acceptance: <b style="color:{COLORS['primary']}">{prob*100:.1f}%</b></div>
            <div style="margin-top:1rem;background:#F8F9FA;border-radius:8px;padding:0.8rem">
                <b>Confidence:</b> {confidence:.1f}% | <b>Model AUC:</b> {best_res['auc']:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=prob * 100,
            title={'text': 'Acceptance Probability (%)', 'font': {'family': 'DM Sans', 'size': 16}},
            delta={'reference': df_full['Personal Loan'].mean() * 100, 'suffix': '% (vs avg)'},
            number={'suffix': '%', 'font': {'size': 28, 'family': 'Playfair Display'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color, 'thickness': 0.25},
                'steps': [
                    {'range': [0, 30], 'color': '#FFE5E5'},
                    {'range': [30, 60], 'color': '#FFF9E5'},
                    {'range': [60, 100], 'color': '#E5F9F0'}
                ],
                'threshold': {'line': {'color': COLORS['primary'], 'width': 3}, 'value': 50}
            }
        ))
        fig_gauge.update_layout(
            height=320, paper_bgcolor='white',
            font=dict(family='DM Sans'), margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
