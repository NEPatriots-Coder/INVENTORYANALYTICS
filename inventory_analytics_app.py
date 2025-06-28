import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Inventory Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with light blue theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        font-weight: 600;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 2px solid #E8F4F8;
    }
    .metric-container {
        background: linear-gradient(135deg, #E8F4F8 0%, #B8D4E3 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #D1E7F0;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(46, 134, 171, 0.1);
    }
    .professional-metric {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        color: #5A6C7D;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .section-header {
        color: #2E86AB;
        border-left: 4px solid #2E86AB;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #F8FBFC, #E8F4F8);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Inventory Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Inventory Management & Performance Analysis")

# Sidebar
st.sidebar.markdown("## 📋 Data Upload")
st.sidebar.markdown("---")

# File uploaders
uploaded_files = {}
file_descriptions = {
    'cycle_count_accuracy': '📊 Cycle Count Accuracy Data',
    'inventory_accuracy_branches': '🏢 Branch Inventory Accuracy',
    'planner_scorecards': '👥 Planner Performance Scorecards'
}

for key, description in file_descriptions.items():
    uploaded_files[key] = st.sidebar.file_uploader(
        description,
        type=['csv'],
        key=f"upload_{key}"
    )

# Display file status
st.sidebar.markdown("### 📋 Upload Status")
for key, description in file_descriptions.items():
    status = "✅ Loaded" if uploaded_files[key] is not None else "⏳ Pending"
    st.sidebar.markdown(f"**{description}**: {status}")

# Main dashboard
if any(uploaded_files.values()):
    
    # Load data
    dataframes = {}
    for key, file in uploaded_files.items():
        if file is not None:
            try:
                dataframes[key] = pd.read_csv(file)
                st.sidebar.success(f"✅ {key} loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {key}: {str(e)}")
    
    # Data preview option
    if st.sidebar.checkbox("👀 Show Data Previews"):
        for key, df in dataframes.items():
            st.markdown(f"### Preview: {file_descriptions[key]}")
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)
            st.markdown("**Columns:** " + ", ".join(df.columns.tolist()))
            st.markdown("---")
    
    # INDIVIDUAL RECORDS SECTION
    st.markdown('<h2 class="section-header">📋 Data Records Overview</h2>', unsafe_allow_html=True)
    
    for key, df in dataframes.items():
        with st.expander(f"📊 View {file_descriptions[key]} Records", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.markdown(f"**Total Records:** {len(df)}")
    
    # PLANNER SCORECARDS ANALYSIS
    if 'planner_scorecards' in dataframes:
        st.markdown('<h2 class="section-header">👥 Planner Performance Analysis</h2>', unsafe_allow_html=True)
        
        df_planner = dataframes['planner_scorecards']
        
        if st.checkbox("🔍 Debug: Show Planner Scorecard Columns"):
            st.write("Available columns:", df_planner.columns.tolist())
            st.dataframe(df_planner.head(), use_container_width=True)
        
        # Find relevant columns
        planner_col = None
        accuracy_col = None
        total_col = None
        
        for col in df_planner.columns:
            if 'planner' in col.lower():
                planner_col = col
            elif 'accuracy' in col.lower() or 'rate' in col.lower():
                accuracy_col = col
            elif 'total' in col.lower() or 'count' in col.lower():
                total_col = col
        
        if planner_col and len(df_planner) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if accuracy_col:
                    fig = px.bar(
                        df_planner.sort_values(accuracy_col, ascending=False).head(15),
                        x=planner_col,
                        y=accuracy_col,
                        color=accuracy_col,
                        color_continuous_scale='Blues',
                        title="Planner Accuracy Performance",
                        labels={accuracy_col: 'Accuracy Rate'}
                    )
                    fig.update_layout(xaxis_tickangle=-45, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Accuracy column not found - showing available data")
                    st.dataframe(df_planner, use_container_width=True)
            
            with col2:
                if accuracy_col:
                    st.markdown("### 🏆 Top Performers")
                    top_performers = df_planner.nlargest(5, accuracy_col)[[planner_col, accuracy_col]]
                    st.dataframe(top_performers, use_container_width=True)
                    
                    st.markdown("### 🎯 Improvement Opportunities")
                    bottom_performers = df_planner.nsmallest(5, accuracy_col)[[planner_col, accuracy_col]]
                    st.dataframe(bottom_performers, use_container_width=True)
                
                st.markdown("### 📊 Summary Statistics")
                if accuracy_col:
                    avg_accuracy = df_planner[accuracy_col].mean()
                    st.markdown(f"**Average Accuracy:** {avg_accuracy:.2f}")
                    std_accuracy = df_planner[accuracy_col].std()
                    st.markdown(f"**Standard Deviation:** {std_accuracy:.2f}")
                
                total_planners = len(df_planner)
                st.markdown(f"**Total Planners:** {total_planners}")
    
    # CYCLE COUNT ACCURACY ANALYSIS
    if 'cycle_count_accuracy' in dataframes:
        st.markdown('<h2 class="section-header">📊 Cycle Count Accuracy Analysis</h2>', unsafe_allow_html=True)
        
        df_cycle = dataframes['cycle_count_accuracy']
        
        if st.checkbox("🔍 Debug: Show Cycle Count Columns"):
            st.write("Available columns:", df_cycle.columns.tolist())
            st.dataframe(df_cycle.head(), use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="professional-metric">{len(df_cycle):,}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Records</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Find variance/difference columns
        variance_cols = [col for col in df_cycle.columns if 'variance' in col.lower() or 'difference' in col.lower()]
        product_cols = [col for col in df_cycle.columns if 'product' in col.lower() or 'item' in col.lower()]
        location_cols = [col for col in df_cycle.columns if 'location' in col.lower() or 'branch' in col.lower()]
        
        if variance_cols:
            main_variance_col = variance_cols[0]
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                avg_variance = df_cycle[main_variance_col].abs().mean()
                st.markdown(f'<div class="professional-metric">{avg_variance:.1f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Avg Abs Variance</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                max_variance = df_cycle[main_variance_col].abs().max()
                st.markdown(f'<div class="professional-metric">{max_variance:.1f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Max Variance</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                perfect_counts = len(df_cycle[df_cycle[main_variance_col] == 0])
                perfect_pct = (perfect_counts / len(df_cycle)) * 100
                st.markdown(f'<div class="professional-metric">{perfect_pct:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Perfect Counts</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Variance distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df_cycle, 
                    x=main_variance_col,
                    nbins=50,
                    title="Variance Distribution",
                    labels={main_variance_col: 'Variance'},
                    color_discrete_sequence=['#2E86AB']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df_cycle,
                    y=main_variance_col,
                    title="Variance Box Plot - Outlier Detection",
                    color_discrete_sequence=['#2E86AB']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Location analysis
        if location_cols and variance_cols:
            location_col = location_cols[0]
            variance_col = variance_cols[0]
            
            st.markdown("### 📍 Location Performance")
            
            location_stats = df_cycle.groupby(location_col).agg({
                variance_col: ['count', 'mean', 'std', lambda x: (x == 0).sum()]
            }).round(2)
            
            location_stats.columns = ['Total_Counts', 'Avg_Variance', 'Std_Variance', 'Perfect_Counts']
            location_stats['Perfect_Rate'] = (location_stats['Perfect_Counts'] / location_stats['Total_Counts'] * 100).round(2)
            location_stats = location_stats.reset_index()
            
            fig = px.bar(
                location_stats,
                x=location_col,
                y='Perfect_Rate',
                color='Perfect_Rate',
                color_continuous_scale='Blues',
                title="Perfect Count Rate by Location"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(location_stats, use_container_width=True)
    
    # INVENTORY ACCURACY BRANCHES
    if 'inventory_accuracy_branches' in dataframes:
        st.markdown('<h2 class="section-header">🏢 Branch Accuracy Analysis</h2>', unsafe_allow_html=True)
        
        df_branches = dataframes['inventory_accuracy_branches']
        
        if st.checkbox("🔍 Debug: Show Branch Data Columns"):
            st.write("Available columns:", df_branches.columns.tolist())
            st.dataframe(df_branches.head(), use_container_width=True)
        
        # Find relevant columns
        branch_cols = [col for col in df_branches.columns if 'branch' in col.lower() or 'location' in col.lower()]
        accuracy_cols = [col for col in df_branches.columns if 'accuracy' in col.lower() or 'rate' in col.lower()]
        
        if branch_cols and accuracy_cols:
            branch_col = branch_cols[0]
            accuracy_col = accuracy_cols[0]
            
            fig = px.bar(
                df_branches.sort_values(accuracy_col, ascending=False),
                x=branch_col,
                y=accuracy_col,
                color=accuracy_col,
                color_continuous_scale='Blues',
                title="Branch Accuracy Comparison"
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🥇 Best Performing Branches")
                top_branches = df_branches.nlargest(5, accuracy_col)[[branch_col, accuracy_col]]
                st.dataframe(top_branches, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Improvement Opportunities")
                bottom_branches = df_branches.nsmallest(5, accuracy_col)[[branch_col, accuracy_col]]
                st.dataframe(bottom_branches, use_container_width=True)
        else:
            st.dataframe(df_branches, use_container_width=True)
    
    # ADVANCED ANALYTICS SECTION
    st.markdown('<h2 class="section-header">🤖 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    if len(dataframes) >= 2:
        st.markdown("### 🔗 Cross-Dataset Analysis")
        
        analysis_option = st.selectbox(
            "Choose Analysis:",
            ["Statistical Summary", "Correlation Heatmap", "Outlier Detection", "Distribution Comparison"]
        )
        
        if analysis_option == "Statistical Summary":
            for name, df in dataframes.items():
                st.markdown(f"#### {file_descriptions[name]} - Overall Statistics")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                st.markdown(f"#### {file_descriptions[name]} - Individual Records")
                st.dataframe(df, use_container_width=True)
                st.markdown("---")
        
        elif analysis_option == "Distribution Comparison":
            st.markdown("### 📊 Distribution Comparison Across Datasets")
            
            # Find common numeric columns across datasets
            all_numeric_cols = {}
            for name, df in dataframes.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                all_numeric_cols[name] = numeric_cols
            
            # Look for similar column patterns
            accuracy_cols = {}
            variance_cols = {}
            count_cols = {}
            
            for name, cols in all_numeric_cols.items():
                for col in cols:
                    if 'accuracy' in col.lower() or 'rate' in col.lower():
                        accuracy_cols[name] = col
                    elif 'variance' in col.lower() or 'difference' in col.lower():
                        variance_cols[name] = col
                    elif 'count' in col.lower() or 'total' in col.lower():
                        count_cols[name] = col
            
            # Create distribution plots
            col1, col2 = st.columns(2)
            
            # Accuracy distributions
            if accuracy_cols:
                with col1:
                    st.markdown("#### 🎯 Accuracy Rate Distributions")
                    fig = go.Figure()
                    
                    colors = ['#2E86AB', '#A23B72', '#F18F01']
                    for i, (name, col) in enumerate(accuracy_cols.items()):
                        df = dataframes[name]
                        if col in df.columns and len(df[col].dropna()) > 0:
                            fig.add_trace(go.Histogram(
                                x=df[col].dropna(),
                                name=f"{name} - {col}",
                                opacity=0.7,
                                nbinsx=20,
                                marker_color=colors[i % len(colors)]
                            ))
                    
                    fig.update_layout(
                        title="Accuracy Rate Distributions",
                        xaxis_title="Accuracy Rate",
                        yaxis_title="Frequency",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Variance/Count distributions  
            if variance_cols or count_cols:
                with col2:
                    st.markdown("#### 📈 Performance Metric Distributions")
                    fig = go.Figure()
                    
                    colors = ['#2E86AB', '#A23B72', '#F18F01']
                    color_idx = 0
                    
                    # Add variance distributions
                    for name, col in variance_cols.items():
                        df = dataframes[name]
                        if col in df.columns and len(df[col].dropna()) > 0:
                            fig.add_trace(go.Histogram(
                                x=df[col].dropna(),
                                name=f"{name} - {col}",
                                opacity=0.7,
                                nbinsx=20,
                                marker_color=colors[color_idx % len(colors)]
                            ))
                            color_idx += 1
                    
                    # Add count distributions
                    for name, col in count_cols.items():
                        df = dataframes[name]
                        if col in df.columns and len(df[col].dropna()) > 0:
                            fig.add_trace(go.Histogram(
                                x=df[col].dropna(),
                                name=f"{name} - {col}",
                                opacity=0.7,
                                nbinsx=20,
                                marker_color=colors[color_idx % len(colors)]
                            ))
                            color_idx += 1
                    
                    fig.update_layout(
                        title="Count/Variance Distributions",
                        xaxis_title="Value",
                        yaxis_title="Frequency", 
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Box plots for comparison
            st.markdown("#### 📦 Box Plot Comparisons")
            
            # Create box plots for all numeric columns
            for metric_type, cols_dict in [("Accuracy Metrics", accuracy_cols), 
                                          ("Variance Metrics", variance_cols),
                                          ("Count Metrics", count_cols)]:
                if cols_dict:
                    st.markdown(f"##### {metric_type}")
                    fig = go.Figure()
                    
                    colors = ['#2E86AB', '#A23B72', '#F18F01']
                    for i, (name, col) in enumerate(cols_dict.items()):
                        df = dataframes[name]
                        if col in df.columns and len(df[col].dropna()) > 0:
                            fig.add_trace(go.Box(
                                y=df[col].dropna(),
                                name=f"{name}",
                                boxpoints='outliers',
                                marker_color=colors[i % len(colors)]
                            ))
                    
                    fig.update_layout(
                        title=f"{metric_type} Box Plot Comparison",
                        yaxis_title="Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics comparison
            st.markdown("#### 📊 Statistical Comparison Table")
            
            comparison_data = []
            for name, df in dataframes.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if len(df[col].dropna()) > 0:
                        stats = df[col].describe()
                        comparison_data.append({
                            'Dataset': name,
                            'Column': col,
                            'Count': int(stats['count']),
                            'Mean': round(stats['mean'], 2),
                            'Std': round(stats['std'], 2),
                            'Min': round(stats['min'], 2),
                            'Max': round(stats['max'], 2),
                            'Median': round(stats['50%'], 2)
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("No numeric data found for comparison")
        
        elif analysis_option == "Correlation Heatmap":
            for name, df in dataframes.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    st.markdown(f"#### {file_descriptions[name]} Correlation Matrix")
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title=f"Correlation Heatmap - {name}",
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("---")
        
        elif analysis_option == "Outlier Detection":
            for name, df in dataframes.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.markdown(f"#### {file_descriptions[name]} Outlier Detection")
                    
                    for col in numeric_cols[:3]:
                        if df[col].std() > 0:
                            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                            outliers = df[z_scores > 3]
                            
                            if len(outliers) > 0:
                                st.markdown(f"**{col} - Found {len(outliers)} outliers:**")
                                st.dataframe(outliers.head(), use_container_width=True)
                            else:
                                st.markdown(f"**{col} - No significant outliers found**")
                    st.markdown("---")

else:
    st.markdown("## 📊 Welcome to Inventory Analytics")
    st.markdown("Upload your CSV files to begin comprehensive inventory analysis.")
    
    st.markdown("""
    ### 📋 How to Use This Dashboard:
    1. **Upload your CSV files** using the sidebar:
       - Cycle Count Accuracy data
       - Branch Inventory Accuracy data
       - Planner Performance Scorecards
    2. **Explore interactive visualizations** and detailed analytics
    3. **Gain actionable insights** from your inventory data
    
    ### 🎯 Key Features:
    - 📊 **Interactive visualizations** with professional charts
    - 🤖 **Advanced analytics** and statistical analysis
    - 📈 **Performance tracking** across all datasets
    - 🎯 **Planner performance** evaluation
    - 📍 **Branch/Location** comparisons
    - 🔍 **Flexible data handling** (adapts to different CSV structures)
    - 📦 **Distribution comparison** across multiple datasets
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Ready to analyze? Upload your files to get started!")

# Footer
st.markdown("---")
st.markdown("### 📊 *Professional Inventory Analytics Dashboard*")
st.markdown("*Powered by advanced data science and analytics*")