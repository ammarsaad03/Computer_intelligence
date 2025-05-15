import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import random

from algorithms.som import StandardSOM
from algorithms.aco import ACO
from algorithms.pso import PSO
from algorithms.harmony_search import HarmonySearch
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.artificial_bee_colony import ArtificialBeeColony

from utils.training import train_algorithm
from utils.metrics import trustworthiness
from utils.visualization import create_original_data_visualization

# Add this function to set random seeds
def set_seeds(seed):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)

# Load breast_cancer dataset
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up Streamlit
st.title("Nature-Inspired Algorithms Comparison (Breast Cancer Dataset)")
st.markdown("Comparing different nature-inspired algorithms on the Breast Cancer dataset")

# Display dataset info
with st.expander("Dataset Information"):
    st.write("Breast Cancer Dataset:")
    st.write("- Number of samples:", len(X))
    st.write("- Number of features:", X.shape[1])
    st.write("- Number of classes:", len(np.unique(y)))
    st.write("\nFeatures:", breast_cancer.feature_names)
    st.write("\nTarget classes:", breast_cancer.target_names)
    st.write("\nDescription of the data :",pd.DataFrame(X, columns=breast_cancer.feature_names).describe())
    st.write("\nSample of the data:")
    st.dataframe(pd.DataFrame(X, columns=breast_cancer.feature_names).head())

# Show original data visualization with both PCA and t-SNE

# Sidebar controls
st.sidebar.header("General Parameters")
epochs = st.sidebar.slider("Epochs", 10, 100, 30)
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, help="Set a seed for reproducible results")

st.subheader("Original Data Visualization")
visualization_method = st.radio("Visualization Method", ["PCA", "t-SNE", "Both"], horizontal=True)

# Set perplexity for t-SNE if used
perplexity = 30

# Create and display original data visualization
original_vis = create_original_data_visualization(
    X_scaled, 
    y, 
    breast_cancer.target_names, 
    method=visualization_method.lower(),
    perplexity=perplexity,
    random_seed=random_seed
)
st.pyplot(original_vis)
# Algorithm selection
st.sidebar.header("Algorithm Selection")
selected_algorithms = st.sidebar.multiselect(
    "Select algorithms to run",
    ["Standard SOM", "Ant Colony Optimization", "Particle Swarm Optimization", 
     "Harmony Search", "Genetic Algorithm", "Artificial Bee Colony"],
    default=["Standard SOM"]
)

# Algorithm-specific parameters
algorithm_params = {}

if "Standard SOM" in selected_algorithms:
    with st.sidebar.expander("SOM Parameters"):
        algorithm_params["SOM"] = {
            "grid_size": st.slider("SOM Grid size", 5, 20, 10),
            "initial_sigma": st.slider("SOM Initial sigma", 0.1, 2.0, 1.0),
            "initial_lr": st.slider("SOM Initial learning rate", 0.01, 1.0, 0.5)
        }

if "Ant Colony Optimization" in selected_algorithms:
    with st.sidebar.expander("ACO Parameters"):
        algorithm_params["ACO"] = {
            "n_ants": st.slider("Number of ants", 5, 50, 10),
            "alpha": st.slider("Pheromone importance (alpha)", 0.1, 5.0, 1.0),
            "beta": st.slider("Heuristic importance (beta)", 0.1, 5.0, 2.0),
            "evaporation_rate": st.slider("Evaporation rate", 0.1, 0.9, 0.5),
            "q": st.slider("Pheromone deposit amount (q)", 1, 200, 100)
        }

if "Harmony Search" in selected_algorithms:
    with st.sidebar.expander("Harmony Search Parameters"):
        algorithm_params["Harmony"] = {
            "hm_size": st.slider("Harmony memory size", 5, 50, 10),
            "hmcr": st.slider("HM considering rate", 0.1, 1.0, 0.9),
            "par": st.slider("Pitch adjusting rate", 0.1, 0.9, 0.3),
            "bw": st.slider("Bandwidth", 0.01, 0.5, 0.05)
        }
if "Particle Swarm Optimization" in selected_algorithms:
    with st.sidebar.expander("PSO Parameters"):
        algorithm_params["PSO"] = {
            "n_particles": st.slider("Number of particles", 5, 50, 10),
            "inertia_weight": st.slider("Inertia weight (w)", 0.1, 1.0, 0.7),
            "cognitive_param": st.slider("Cognitive parameter (c1)", 0.1, 3.0, 1.5),
            "social_param": st.slider("Social parameter (c2)", 0.1, 3.0, 1.5)
        }

if "Genetic Algorithm" in selected_algorithms:
    with st.sidebar.expander("GA Parameters"):
        algorithm_params["GA"] = {
            "pop_size": st.slider("Population size", 5, 50, 20),
            "crossover_rate": st.slider("Crossover rate", 0.1, 1.0, 0.8),
            "mutation_rate": st.slider("Mutation rate", 0.01, 0.5, 0.1),
            "elite_size": st.slider("Elite size", 1, 5, 2),
            "selection_method": st.radio("Selection method", ["tournament", "parent"], index=0),
            "crossover_method": st.radio("Crossover method", ["multi_point", "uniform"], index=0),
        }

if "Artificial Bee Colony" in selected_algorithms:
    with st.sidebar.expander("ABC Parameters"):
        algorithm_params["ABC"] = {
            "n_bees": st.slider("Number of bees", 5, 50, 20),
            "limit": st.slider("Abandonment limit", 5, 50, 20),
            "max_cycles": st.slider("Maximum cycles", 10, 200, 100),
            "output_dim": st.slider("Output dimensions", 2, min(X.shape[1], 5), 2)
        }

# Initialize results storage
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'plots' not in st.session_state:
    st.session_state.plots = {}

# Clear previous results when running new algorithms
if st.sidebar.button("Run Selected Algorithms"):
    # Clear previous results
    st.session_state.results = {}
    st.session_state.plots = {}
    # Set global seeds
    set_seeds(random_seed)
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    
    # Start total timing
    total_start_time = time.time()
    
    # Function to update Streamlit progress
    def update_progress(message):
        status_text.text(message)
    
    for idx, name in enumerate(selected_algorithms):
        if name == "Standard SOM":
            params = algorithm_params.get("SOM", {})
            algo = StandardSOM(
                grid_size=params.get("grid_size", 10),
                input_dim=X.shape[1],
                sigma=params.get("initial_sigma", 1.0),
                lr=params.get("initial_lr", 0.5),
                random_seed=random_seed
            )
        elif name == "Ant Colony Optimization":
            params = algorithm_params.get("ACO", {})
            algo = ACO(
                n_ants=params.get("n_ants", 10),
                alpha=params.get("alpha", 1.0),
                beta=params.get("beta", 2.0),
                evaporation_rate=params.get("evaporation_rate", 0.5),
                q=params.get("q", 100),
                n_features_to_select=params.get("n_features", 2),
                random_seed=random_seed
            )
        elif name == "Harmony Search":
            params = algorithm_params.get("Harmony", {})
            algo = HarmonySearch(
                output_dim=params.get("output_dim", 2),
                hm_size=params.get("hm_size", 10),
                hmcr=params.get("hmcr", 0.9),
                par=params.get("par", 0.3),
                bw=params.get("bw", 0.05),
                random_seed=random_seed
            )
        elif name == "Particle Swarm Optimization":
            params = algorithm_params.get("PSO", {})
            algo = PSO(
                output_dim=params.get("output_dim", 2),
                n_particles=params.get("n_particles", 10),
                w=params.get("inertia_weight", 0.7),
                c1=params.get("cognitive_param", 1.5),
                c2=params.get("social_param", 1.5),
                random_seed=random_seed
            )
        elif name == "Genetic Algorithm":
            params = algorithm_params.get("GA", {})
            algo = GeneticAlgorithm(
                output_dim=params.get("output_dim", 2),
                pop_size=params.get("pop_size", 20),
                crossover_rate=params.get("crossover_rate", 0.8),
                mutation_rate=params.get("mutation_rate", 0.1),
                elite_size=params.get("elite_size", 2),
                random_seed=random_seed
            )
              # Store the selection and crossover methods as attributes
            algo.selection_method = params.get("selection_method", "tournament")
            algo.crossover_method = params.get("crossover_method", "multi_point")
        elif name == "Artificial Bee Colony":
            params = algorithm_params.get("ABC", {})
            algo = ArtificialBeeColony(
                output_dim=params.get("output_dim", 2),
                n_bees=params.get("n_bees", 20),
                limit=params.get("limit", 20),
                max_cycles=params.get("max_cycles", 100),
                random_seed=random_seed
            )
    
        result = train_algorithm(
            name=name, 
            algorithm=algo, 
            data=X_scaled, 
            labels=y, 
            epochs=epochs, 
            feature_names=breast_cancer.feature_names,
            class_names=breast_cancer.target_names,
            progress_callback=update_progress,
            random_seed=random_seed
        )
        trustworthiness_score = None
        if hasattr(algo, 'projected_data') and algo.projected_data is not None:
            trustworthiness_score = trustworthiness(X_scaled, algo.projected_data)
        elif hasattr(algo, 'reduced_data') and algo.reduced_data is not None:
            trustworthiness_score = trustworthiness(X_scaled, algo.reduced_data)
        
        # Update progress
        progress_bar.progress((idx + 1) / len(selected_algorithms))
        param_key_map = {
            "Standard SOM": "SOM",
            "Ant Colony Optimization": "ACO", 
            "Harmony Search": "Harmony",
            "Particle Swarm Optimization": "PSO",
            "Genetic Algorithm": "GA",
            "Artificial Bee Colony": "ABC"
        }
        
        # Store results
        st.session_state.results[name] = {
            'quantization_error': result['quantization_error'],
            'time': result['time'],
            'silhouette_score': result['silhouette_score'],
            'trustworthiness': trustworthiness_score,
            'parameters': algorithm_params.get(param_key_map.get(name, ""), {})
        }
        st.session_state.plots[name] = result['plot']
    
    # Calculate total time
    total_elapsed_time = time.time() - total_start_time
    
    progress_bar.empty()
    status_text.text("Training complete!")
    time_text.text(f"Total training time: {total_elapsed_time:.2f} seconds")

# Display results
if st.session_state.results:
    st.subheader("Results")
    
    # Create columns for the plots
    cols = st.columns(min(2, len(st.session_state.plots)))
    
    # Display plots and results
    for idx, (name, fig) in enumerate(st.session_state.plots.items()):
        with cols[idx % len(cols)]:
            st.pyplot(fig)
            st.write(f"Quantization Error: {st.session_state.results[name]['quantization_error']:.4f}")
            st.write(f"Silhouette Score: {st.session_state.results[name]['silhouette_score']:.4f}")
            if st.session_state.results[name].get('trustworthiness') is not None:
                st.write(f"Trustworthiness: {st.session_state.results[name]['trustworthiness']:.4f}")
            st.write(f"Training Time: {st.session_state.results[name]['time']:.2f}s")
            
            # Display parameters used
            with st.expander(f"Parameters used for {name}"):
                params = st.session_state.results[name].get('parameters', {})
                if params:
                    st.json(params)
                else:
                    st.write("No additional parameters")
    
    # Show comparison table
    st.subheader("Performance Comparison")
    results_df = pd.DataFrame([
        {
            "Algorithm": name,
            "Quantization Error": results['quantization_error'],
            "Silhouette Score": results['silhouette_score'],
            "Trustworthiness": results.get('trustworthiness', "N/A"),
            "Training Time (s)": results['time']
        }
        for name, results in st.session_state.results.items()
    ]).sort_values("Quantization Error")
    st.dataframe(results_df)

# Add explanation
with st.expander("How to interpret these results"):
    st.markdown("""
    **Algorithms Explained:**
    1. **Standard SOM**: Self-Organizing Map that creates a 2D representation of the data
    2. **Ant Colony Optimization**: Uses pheromone trails to find the most informative features
    3. **Particle Swarm Optimization**: Simulates bird flocking, particles move towards best solutions
    4. **Harmony Search**: Inspired by musical improvisation, creates a dimensional projection
    5. **Genetic Algorithm**: Uses evolutionary principles of selection, crossover and mutation
    
    **Visualization:**
    - For SOM: Shows the 2D grid with data points assigned to neurons
    - For ACO: Shows data points in the space of selected features
    - For Harmony Search: Shows data points in the projected space
    - For PSO and GA: Shows PCA-reduced data with cluster assignments
    - Lower Quantization Error (QE) is better (measures average distance to nearest neighbor)
    
    **Parameters:**
    - Each algorithm has specific parameters that can be adjusted in the sidebar
    - Parameters are shown in the expandable sections below each result
    """)