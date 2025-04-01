import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import lasio
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool, CustomJS, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.layouts import column, row
from io import StringIO
from DEM_Berryman import DEM
import Gassmann

# Page configuration
st.set_page_config(layout="wide", page_title="Xu-Payne Rock Physics Analyzer")

# Title and description
st.title("Xu-Payne Rock Physics Analyzer")
st.markdown("""
This app analyzes well log data using the Xu-Payne (2009) rock physics model.
Upload a LAS file, select your depth interval, and explore the relationships between Vp, porosity, and water saturation.
""")

# File uploader
uploaded_file = st.file_uploader("Upload LAS File", type=['las', 'LAS'])

if uploaded_file is not None:
    # Read LAS file
    las_text = StringIO(uploaded_file.getvalue().decode("utf-8"))
    log = lasio.read(las_text)
    df = log.df().reset_index()
    
    # Get available curves
    available_curves = [curve.mnemonic for curve in log.curves]
    
    # Create mapping between standard names and available curves
    curve_mapping = {
        'Depth': 'DEPT' if 'DEPT' in available_curves else available_curves[0],
        'Vp': 'P-VELOCITY' if 'P-VELOCITY' in available_curves else next((c for c in available_curves if 'VEL' in c or 'VP' in c), None),
        'Porosity': 'EFFECTIVE_POROSITY' if 'EFFECTIVE_POROSITY' in available_curves else next((c for c in available_curves if 'PHI' in c), None),
        'Sw': 'WATER_SATURATION' if 'WATER_SATURATION' in available_curves else next((c for c in available_curves if 'SW' in c), None)
    }
    
    # Check if we have all required curves
    if None in curve_mapping.values():
        missing = [k for k, v in curve_mapping.items() if v is None]
        st.error(f"Missing required curves: {', '.join(missing)}")
    else:
        # Create logfile dataframe
        logfile = pd.DataFrame({
            "Depth": log[curve_mapping['Depth']],
            "VP": log[curve_mapping['Vp']] * 0.001,  # Convert to km/s
            "Porosity": log[curve_mapping['Porosity']],
            "Saturation": log[curve_mapping['Sw']]
        }).dropna()
        
        # Depth range selector
        min_depth = float(logfile['Depth'].min())
        max_depth = float(logfile['Depth'].max())
        
        st.sidebar.header("Depth Selection")
        top_depth = st.sidebar.number_input("Top Depth", min_value=min_depth, max_value=max_depth, value=min_depth)
        base_depth = st.sidebar.number_input("Base Depth", min_value=min_depth, max_value=max_depth, value=max_depth)
        
        # Filter data
        filtered_data = logfile[(logfile['Depth'] >= top_depth) & (logfile['Depth'] <= base_depth)]
        
        # Xu-Payne model calculation
        def calculate_xu_payne():
            # Matrix properties
            Km_dol = 69.0  # GPa
            Gm_dol = 52.0  # GPa
            rhom_dol = 2.88  # g/cm3
            Km_lim = 77.0  # GPa
            Gm_lim = 32.0  # GPa
            rhom_lim = 2.71  # g/cm3

            # Fluid properties
            Kf = 2.24  # GPa
            rhof = 0.94  # g/cm3

            # Porosity
            phimax = 0.4

            # Aspect ratios
            alpha_ref = 0.11
            alpha_crack = 0.02
            alpha_stiff = 0.8

            # Create inclusion configurations
            fractions = [0.2, 0.4, 0.6, 0.8]
            crackandref_alphas = []
            crackandref_volumes = []
            stiffandref_alphas = []
            stiffandref_volumes = []
            
            for fraction in fractions:
                crackandref_alphas.append([alpha_crack, alpha_ref])
                crackandref_volumes.append([fraction * phimax, (1.0 - fraction) * phimax])
                stiffandref_alphas.append([alpha_stiff, alpha_ref])
                stiffandref_volumes.append([fraction * phimax, (1.0 - fraction) * phimax])

            alphas = [[alpha_crack]] + crackandref_alphas + [[alpha_ref]] + stiffandref_alphas + [[alpha_stiff]]
            volumes = [[phimax]] + crackandref_volumes + [[phimax]] + stiffandref_volumes + [[phimax]]
            alphas += [[alpha_ref]]
            volumes += [[phimax]]

            Kms = len(alphas) * [Km_lim]
            Gms = len(alphas) * [Gm_lim]
            Kms[-1] = Km_dol
            Gms[-1] = Gm_dol

            styles = ['m:'] + ['m:'] * len(crackandref_alphas) + ['b-'] + ['r--'] + ['r--'] * len(stiffandref_alphas) + ['g-']
            
            xu_payne_curves = []
            
            for inclusion_alphas, inclusion_volumes, style, Km, Gm in zip(alphas, volumes, styles, Kms, Gms):
                ni = len(inclusion_alphas)
                Kis = np.zeros(ni, dtype=float)
                Gis = np.zeros(ni, dtype=float)

                K, G, phi = DEM(Km, Gm, Kis, Gis, np.array(inclusion_alphas), np.array(inclusion_volumes))
                rho = (1.0 - phi) * rhom_lim + phi * rhof
                Ks = Gassmann.Ks(K, Km, Kf, phi)
                Vp = np.sqrt((Ks + 4.0 * G / 3.0) / rho)
                
                xu_payne_curves.append((phi, Vp, style))
            
            return xu_payne_curves
        
        # Calculate Xu-Payne curves
        xu_payne_curves = calculate_xu_payne()
        
        # Create Bokeh plots
        st.header("Interactive Cross-Plots")
        
        # Create ColumnDataSource
        source = ColumnDataSource(data=dict(
            depth=filtered_data['Depth'],
            vp=filtered_data['VP'],
            porosity=filtered_data['Porosity'],
            saturation=filtered_data['Saturation']
        ))
        
        # Create a color mapper
        color_mapper = LinearColorMapper(palette=Viridis256, 
                                       low=filtered_data['Saturation'].min(), 
                                       high=filtered_data['Saturation'].max())
        
        # Cross-plot (Vp vs Porosity)
        cross_plot = figure(width=600, height=500, 
                          title="Vp vs Porosity (Color by Sw)",
                          tools="pan,wheel_zoom,box_zoom,reset,hover,lasso_select")
        
        cross_plot.circle('porosity', 'vp', size=8, source=source,
                        fill_color={'field': 'saturation', 'transform': color_mapper},
                        line_color=None, alpha=0.6)
        
        # Add Xu-Payne curves
        style_mapping = {
            'm:': ('magenta', [4, 4]),
            'b-': ('blue', 'solid'),
            'r--': ('red', 'dashed'),
            'g-': ('green', 'solid')
        }
        
        for phi, Vp, style in xu_payne_curves:
            line_color, line_dash = style_mapping.get(style, ('black', 'solid'))
            cross_plot.line(phi, Vp, line_color=line_color, line_dash=line_dash, 
                          line_width=2, legend_label="Xu-Payne Curve")
        
        cross_plot.xaxis.axis_label = "Porosity"
        cross_plot.yaxis.axis_label = "Vp (km/s)"
        
        # Depth plot (Vp vs Depth)
        depth_plot = figure(width=600, height=500, 
                          title="Vp vs Depth (Color by Sw)",
                          tools="pan,wheel_zoom,box_zoom,reset,hover")
        
        depth_plot.circle('vp', 'depth', size=8, source=source,
                        fill_color={'field': 'saturation', 'transform': color_mapper},
                        line_color=None, alpha=0.6)
        
        depth_plot.yaxis.axis_label = "Depth"
        depth_plot.xaxis.axis_label = "Vp (km/s)"
        depth_plot.y_range.flipped = True  # Invert depth axis
        
        # Add hover tools
        hover = cross_plot.select_one(HoverTool)
        hover.tooltips = [
            ("Depth", "@depth{0.00}"),
            ("Vp", "@vp{0.000} km/s"),
            ("Porosity", "@porosity{0.000}"),
            ("Sw", "@saturation{0.000}")
        ]
        
        hover = depth_plot.select_one(HoverTool)
        hover.tooltips = [
            ("Depth", "@depth{0.00}"),
            ("Vp", "@vp{0.000} km/s"),
            ("Porosity", "@porosity{0.000}"),
            ("Sw", "@saturation{0.000}")
        ]
        
        # Create a source for highlighted points
        highlight_source = ColumnDataSource(data=dict(
            depth=[], vp=[], porosity=[], saturation=[]
        ))
        
        # Add highlighted points to both plots
        highlight_cross = cross_plot.circle('porosity', 'vp', size=10, 
                                          source=highlight_source,
                                          color='black', alpha=0.8)
        
        highlight_depth = depth_plot.circle('vp', 'depth', size=10, 
                                          source=highlight_source,
                                          color='black', alpha=0.8)
        
        # JavaScript callback for lasso selection
        callback = CustomJS(args=dict(source=source, highlight_source=highlight_source), code="""
            const indices = source.selected.indices;
            const depth = [];
            const vp = [];
            const porosity = [];
            const saturation = [];
            
            for (let i = 0; i < indices.length; i++) {
                depth.push(source.data['depth'][indices[i]]);
                vp.push(source.data['vp'][indices[i]]);
                porosity.push(source.data['porosity'][indices[i]]);
                saturation.push(source.data['saturation'][indices[i]]);
            }
            
            highlight_source.data = {
                depth: depth,
                vp: vp,
                porosity: porosity,
                saturation: saturation
            };
            highlight_source.change.emit();
        """)
        
        source.selected.js_on_change('indices', callback)
        
        # Display plots side by side
        col1, col2 = st.columns(2)
        with col1:
            st.bokeh_chart(cross_plot, use_container_width=True)
        with col2:
            st.bokeh_chart(depth_plot, use_container_width=True)
        
        # Display selected points info
        st.subheader("Selected Points Information")
        selected_info = st.empty()
        
        # Update selected points info when selection changes
        if source.selected.indices:
            selected_df = filtered_data.iloc[source.selected.indices]
            selected_info.dataframe(selected_df)
        else:
            selected_info.write("No points selected. Use the lasso tool to select points.")
        
        # Matplotlib version for static export
        st.header("Static Cross-Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Xu-Payne curves
        for phi, Vp, style in xu_payne_curves:
            ax.plot(phi, Vp, style)
        
        # Plot data points
        sc = ax.scatter(filtered_data['Porosity'], filtered_data['VP'], 
                       c=filtered_data['Saturation'], s=20, cmap='viridis')
        
        plt.colorbar(sc, label='Water Saturation (Sw)')
        ax.set_xlabel("Porosity")
        ax.set_ylabel("Vp (km/s)")
        ax.set_title("Vp vs Porosity with Xu-Payne Curves")
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Data export
        st.sidebar.header("Data Export")
        if st.sidebar.button("Export Selected Points to CSV"):
            if source.selected.indices:
                selected_df.to_csv("selected_points.csv", index=False)
                st.sidebar.success("Selected points exported to selected_points.csv")
            else:
                st.sidebar.warning("No points selected for export")
