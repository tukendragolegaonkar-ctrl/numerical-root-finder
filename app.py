import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# --- Web Page Configuration ---
st.set_page_config(page_title="Numerical Root Finder", page_icon="🧮", layout="wide")

# --- Custom CSS for Visibility and Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    [data-testid="stMetric"] {
        background-color: white !important;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧮 Interactive Numerical Root Finder")
st.markdown("<p style='text-align: center;'>Visualize Bisection and False Position methods with real-time error tracking.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Settings")
method = st.sidebar.selectbox("Choose Method", ["Bracketing Method (Bisection)", "False Position"])
equation_input = st.sidebar.text_input("Enter Function f(x):", "x**3 - x - 2")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    a_input = st.number_input("Lower Guess (a):", value=1.0, format="%.4f")
with col_b:
    b_input = st.number_input("Upper Guess (b):", value=2.0, format="%.4f")

error_threshold = st.sidebar.slider("Error Threshold (%)", 0.1, 10.0, 1.0)

# --- Math Logic ---
def f(x, expr):
    # Dictionary to handle math functions safely
    safe_dict = {"x": x, "np": np, "sin": np.sin, "cos": np.cos, 
                 "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tan": np.tan}
    return eval(expr, {"__builtins__": None}, safe_dict)

def run_solver():
    a, b = a_input, b_input
    expr = equation_input
    
    # 1. Initial Sign Change Check
    try:
        f_a_init = f(a, expr)
        f_b_init = f(b, expr)
        if f_a_init * f_b_init >= 0:
            st.error("⚠️ **Error:** f(a) and f(b) must have opposite signs. Try different guesses.")
            return
    except Exception as e:
        st.error(f"❌ **Invalid Equation:** {e}")
        return

    iter_count = 1
    c_old = None
    error = 100.0
    
    st.subheader(f"🚀 Solution using {method}")
    results_container = st.container()

    while True:
        f_a = f(a, expr)
        f_b = f(b, expr)
        
        # 2. Calculate Estimate (cn)
        if "Bracketing" in method:
            # Bisection Formula
            c_new = (a + b) / 2 
        else:
            # False Position Formula: xr = (a*f(b) - b*f(a)) / (f(b) - f(a))
            # Mathematically equivalent to: (a*f_b - b*f_a) / (f_b - f_a)
            denominator = f_b - f_a
            if denominator == 0:
                st.error("Division by zero in False Position calculation.")
                break
            c_new = (a * f_b - b * f_a) / denominator
        
        f_c_new = f(c_new, expr)

        # 3. Error Calculation (Approximate Relative)
        if c_old is not None and c_new != 0:
            error = abs((c_new - c_old) / c_new) * 100
        
        # 4. Display UI for current Iteration
        with results_container:
            with st.expander(f"Iteration {iter_count} | cn = {c_new:.5f}", expanded=(iter_count == 1)):
                col_stats, col_graph = st.columns([1, 2])
                
                with col_stats:
                    st.write("**Step Analysis**")
                    st.metric("Lower (a)", f"{a:.5f}")
                    st.metric("Upper (b)", f"{b:.5f}")
                    st.metric("Estimate (cn)", f"{c_new:.5f}")
                    st.metric("f(cn)", f"{f_c_new:.5f}")
                    
                    err_display = f"{error:.4f}%" if c_old is not None else "---"
                    st.metric("Relative Error %", err_display)

                with col_graph:
                    plt.style.use('bmh')
                    fig, ax = plt.subplots(figsize=(7, 4))
                    
                    # Generate curve data
                    x_min, x_max = min(a, b, c_new), max(a, b, c_new)
                    x_range = np.linspace(x_min - 0.5, x_max + 0.5, 300)
                    y_range = [f(val, expr) for val in x_range]
                    
                    ax.plot(x_range, y_range, color='#34495e', linewidth=2, label='f(x)')
                    ax.axhline(0, color='black', linewidth=1) # X-axis
                    
                    # Draw interpolation line for False Position
                    if "False Position" in method:
                        ax.plot([a, b], [f_a, f_b], color='orange', linestyle='--', alpha=0.6, label='Secant Line')

                    def plot_point(x_val, y_val, label, color, position='bottom'):
                        ax.plot(x_val, y_val, marker='o', color=color, markersize=8, label=label)
                        ax.annotate(f"{label}", (x_val, y_val), textcoords="offset points", 
                                    xytext=(0, 10 if position=='top' else -15), ha='center', fontsize=9, fontweight='bold')

                    plot_point(a, f_a, 'a', '#e74c3c', 'top')
                    plot_point(b, f_b, 'b', '#3498db', 'top')
                    plot_point(c_new, f_c_new, 'cn', '#27ae60', 'bottom')
                    
                    ax.set_xlabel("X")
                    ax.set_ylabel("f(x)")
                    ax.legend(prop={'size': 8})
                    st.pyplot(fig)

        # 5. Convergence & Bounds Update (Sign Convention)
        if abs(error) < error_threshold and c_old is not None:
            st.success(f"🎊 **Success!** Root converged to **{c_new:.6f}**")
            st.balloons()
            break
            
        # Standard Bracket Selection
        if f_a * f_c_new < 0:
            b = c_new
        else:
            a = c_new
        
        c_old = c_new
        iter_count += 1
        
        if iter_count > 20: 
            st.warning("⚠️ Max iterations (20) reached to prevent long load times.")
            break

# --- Execution ---
if st.sidebar.button("Run Solver"):
    run_solver()
