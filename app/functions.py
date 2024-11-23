import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import cvxpy as cp
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus
from scipy.stats import beta, poisson, expon, nbinom, gamma, chi2_contingency, binom
import warnings
from scipy.optimize import linprog
from PIL import Image, ImageDraw, ImageFont

# Dictionary mapping table names to your CSV file paths
csv_files = {
    "Route": "Large_Expanded_Route.csv",
    "Order": "Large_Expanded_Order.csv"
}

# Function to load CSV data into MySQL
def load_csv_to_mysql(table_name, file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Upload data to MySQL table
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        print(f"Loaded data into '{table_name}' table successfully.")
    except Exception as e:
        print(f"Error loading data into '{table_name}': {e}")

def initialize(load_data=True):
    global route_data, order_data, engine

    # Ignore warnings
    warnings.filterwarnings('ignore')
    # MySQL connection setup
    engine = create_engine('mysql+pymysql://root:1234@localhost/newestdb')

    if load_data:
        # Iterate over CSV files and load them into MySQL
        for table_name, file_path in csv_files.items():
            load_csv_to_mysql(table_name, file_path)

    # Load data from MySQL for EDA
    route_data = pd.read_sql_table('route', con=engine)
    order_data = pd.read_sql_table('order', con=engine)

    # Run EDA functions
    #eda_travel_mode_distribution()
    #eda_cost_correlation_heatmap()
    #eda_time_correlation_heatmap()
    #eda_volumne_correlation_heatmap()
    #eda_cluster_analysis()
    #eda_feature_importance()
    #eda_geospacial_analysis()
    #eda_dependency_analysis()

############################################################################################################################

# Exploratory Data Analysis (EDA) Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_route_data():
    global route_data
    return route_data.head()

def get_order_data():
    global order_data
    return order_data.head()

def eda_missing_values():
    global route_data, order_data
    route_missing = route_data.isnull().sum()
    order_missing = order_data.isnull().sum()
    return route_missing, order_missing

def eda_numerical_summary():
    global route_data, order_data
    route_num_summary = route_data.describe()
    order_num_summary = order_data.describe()
    return route_num_summary, order_num_summary

def eda_travel_mode_distribution():
    column_name='Travel Mode'
    if column_name not in route_data.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")

    travel_mode_distribution = route_data[column_name].value_counts()
    plt.plot(travel_mode_distribution.index, travel_mode_distribution.values, marker='o', color='b')
    plt.title('Travel Mode Distribution in Route Data')
    plt.ylabel('Count')
    plt.xlabel('Travel Mode')
    plt.savefig('static/images/eda_3_travel_mode_distribution.png')

def eda_cost_correlation_heatmap():
    cost_columns = ['Fixed Freight Cost', 'Port/Airport/Rail Handling Cost', 'Bunker/ Fuel Cost', 'Documentation Cost']
    if not all(column in route_data.columns for column in cost_columns):
        missing_columns = [col for col in cost_columns if col not in route_data.columns]
        raise ValueError(f"Missing columns in the dataset: {missing_columns}")

    cost_correlation = route_data[cost_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cost_correlation, annot=True, cmap='viridis', fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Among Cost Attributes in Route Data')
    plt.savefig('static/images/eda_4_cost_correlation_heatmap.png')

def eda_time_correlation_heatmap():
    time_columns = ['Port/Airport/Rail Handling time (hours)', 'Transit time (hours)']
    if not all(column in route_data.columns for column in time_columns):
        missing_columns = [col for col in time_columns if col not in route_data.columns]
        raise ValueError(f"Missing columns in the dataset: {missing_columns}")

    time_correlation = route_data[time_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(time_correlation, annot=True, cmap='cividis', fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Among Time Attributes in Route Data')
    plt.savefig('static/images/eda_5_time_correlation_heatmap.png')

def eda_volumne_correlation_heatmap():
    volume_columns = ['Weight (KG)', 'Volume', 'Order Value']
    if not all(column in order_data.columns for column in volume_columns):
        missing_columns = [col for col in volume_columns if col not in order_data.columns]
        raise ValueError(f"Missing columns in the dataset: {missing_columns}")

    volume_correlation = order_data[volume_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(volume_correlation, annot=True, cmap='plasma', fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Among Volume and Value Attributes in Order Data')
    plt.savefig('static/images/eda_6_volume_correlation_heatmap.png')

def eda_cluster_analysis():
    data = route_data.copy()
    columns = ['Fixed Freight Cost', 'Bunker/ Fuel Cost']
    n_clusters = 3

    # Selecting relevant columns for clustering (only numeric)
    X = data[columns].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    data['Cluster'] = kmeans.labels_

    # Visualize the clusters
    plt.figure(figsize=(25, 15))
    sns.scatterplot(x=X[columns[0]], y=X[columns[1]], hue=data['Cluster'], palette='viridis', s=100)
    plt.title(f'Clustering Analysis on {columns[0]} and {columns[1]}')
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.legend(title='Cluster')
    plt.savefig('static/images/eda_7_cluster_analysis.png')

def eda_feature_importance(just_df=False):
    data = order_data.copy()
    target = 'Order Value'
    features = ['Weight (KG)', 'Volume', 'Tax Percentage']

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset.")
    if not all(column in data.columns for column in features):
        missing_columns = [col for col in features if col not in data.columns]
        raise ValueError(f"Missing feature columns in the dataset: {missing_columns}")

    rf = RandomForestRegressor(random_state=0)
    rf.fit(data[features], data[target])
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    feature_importance_df = feature_importance_df.set_index('Feature')
    feature_importance_df = feature_importance_df.to_dict()['Importance']

    if just_df:
        return feature_importance_df
    else:
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title(f'Feature Importance for predicting {target}')
        plt.savefig('static/images/eda_8_feature_importance.png')

def eda_geospacial_analysis():
    data = route_data.copy()
    source_col = 'Source'
    destination_col = 'Destination'

    if source_col not in data.columns or destination_col not in data.columns:
        raise ValueError(f"Columns '{source_col}' or '{destination_col}' not found in the dataset.")

    # Count occurrences of each route (Source-Destination pair)
    route_counts = data.groupby([source_col, destination_col]).size().reset_index(name='Count')

    # Reshape the data to make it suitable for heatmap
    route_pivot = route_counts.pivot(index=source_col, columns=destination_col, values='Count')

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(route_pivot, annot=True, cmap='cividis', fmt=".0f", cbar_kws={'label': 'Route Frequency'})
    plt.title('Geospatial Analysis of Routes')
    plt.xlabel('Destination')
    plt.ylabel('Source')
    plt.savefig('static/images/eda_9_geospacial_analysis.png')

def eda_dependency_analysis():
    data = route_data.copy()
    data['Long_Transit'] = (data['Transit time (hours)'] > data['Transit time (hours)'].median()).astype(int)
    target='Long_Transit'
    features=['Container Size', 'Fixed Freight Cost', 'Bunker/ Fuel Cost']

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset.")
    if not all(column in data.columns for column in features):
        missing_columns = [col for col in features if col not in data.columns]
        raise ValueError(f"Missing feature columns in the dataset: {missing_columns}")

    dt = DecisionTreeClassifier(random_state=0, max_depth=3)
    dt.fit(data[features], data[target])

    # Plot the decision tree
    plt.figure(figsize=(12, 8))
    from sklearn.tree import plot_tree

    plot_tree(dt, feature_names=features, class_names=True, filled=True, rounded=True)
    plt.title(f'Dependency Analysis for {target}')
    plt.savefig('static/images/eda_10_dependency_analysis.png')


# Managerial Statistics (MS) Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Bayesian Inference Using Beta Distribution
def ms_bayesian_inference_beta(column, alpha_prior=2, beta_prior=2):
    global route_data
    """
    Perform Bayesian inference using the Beta distribution for the given column in the dataset.

    Args:
        route_data (DataFrame): The dataset containing the column to analyze.
        column (str): The column name to use for determining the probability of the event.
        alpha_prior (int): The prior parameter for the Beta distribution. Default is 2.
        beta_prior (int): The prior parameter for the Beta distribution. Default is 2.

    Returns:
        None: Displays the posterior distribution plot.
    """
    if column not in route_data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    # Observed data
    long_transit_time = (route_data[column] > route_data[column].median()).sum()
    total_samples = len(route_data)

    # Update posterior based on data
    alpha_posterior = alpha_prior + long_transit_time
    beta_posterior = beta_prior + (total_samples - long_transit_time)

    # Plot the posterior distribution
    x = np.linspace(0, 1, 100)
    posterior_pdf = beta.pdf(x, alpha_posterior, beta_posterior)

    # Save the plot as an image
    plt.figure(figsize=(12, 6))
    plt.plot(x, posterior_pdf, label='Posterior', color='blue')
    plt.title('Posterior Distribution of Probability for Column: ' + column)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/images/ms_1_bayesian_inference_beta.png')

# 2. Poisson Distribution for Weekly Orders
def ms_poisson_distribution_analysis(source, destination, min_range=1, max_range=5):
    global route_data
    """
    Perform Poisson distribution analysis for weekly orders between specified source and destination.

    Args:
        route_data (DataFrame): The dataset containing route data.
        source (str): The source port for the analysis.
        destination (str): The destination port for the analysis.
        min_range (int): Minimum range for order count. Default is 1 to avoid empty range.
        max_range (int): Maximum range for order count. Default is 5.

    Returns:
        None: Displays the Poisson distribution plot.
    """
    if 'Source' not in route_data.columns or 'Destination' not in route_data.columns:
        raise ValueError("Source or Destination column not found in the dataset.")

    # Count the occurrences of each route (Source-Destination pair)
    route_counts = route_data.groupby(['Source', 'Destination']).size().reset_index(name='Count')

    # Select a specific route for analysis
    selected_route = route_counts[(route_counts['Source'] == source) & 
                                   (route_counts['Destination'] == destination)]['Count'].values

    if len(selected_route) == 0:
        raise ValueError(f"No data found for the route {source} to {destination}.")
    
    selected_route = selected_route[0]  # Get the count value for the selected route

    mu = max(selected_route, min_range)  # Set minimum to avoid empty range for small `mu`
    x = np.arange(0, max_range)  # Increase range to show distribution for meaningful order counts
    poisson_dist = poisson.pmf(x, mu)

    # Plot the Poisson distribution
    plt.figure(figsize=(10, 6))
    plt.stem(x, poisson_dist)
    plt.title(f'Poisson Distribution for Weekly Orders: {source} to {destination}')
    plt.xlabel('Number of Orders')
    plt.ylabel('Probability')
    plt.savefig('static/images/ms_2_poisson_distribution.png')

# 3. Exponential Distribution Analysis
def ms_exponential_distribution_analysis(num_points=100, time_range=(0, 15)):
    data = order_data.copy()

    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d-%b-%y')

    # Sort the data by 'Order Date'
    sorted_order_data = data.sort_values(by='Order Date')

    # Calculate the time differences between consecutive orders in days
    sorted_order_data['Time Between Orders'] = sorted_order_data['Order Date'].diff().dt.days

    # Drop NaN values and calculate the mean time between orders
    mean_time_between_orders = sorted_order_data['Time Between Orders'].dropna().mean()

    # Define the function for Exponential Distribution Analysis
    x = np.linspace(time_range[0], time_range[1], num_points)
    y = expon.pdf(x, scale=mean_time_between_orders)

    # Plot the exponential distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Exponential Distribution")
    plt.xlabel("Time Between Orders (days)")
    plt.ylabel("Probability Density")
    plt.title("Exponential Distribution of Time Between Orders")
    plt.legend()
    plt.savefig('static/images/ms_3_exponential_distribution.png')

# 4. Binomial Distribution for Success/Failure Events
def ms_binomial_distribution_analysis():
    data = order_data.copy()

    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d-%b-%y')

    # Sort the data by 'Order Date'
    sorted_order_data = data.sort_values(by='Order Date')
    
    sorted_order_data['Success'] = (sorted_order_data['Journey Type'] == "Domestic").astype(int)

    # Calculate the probability of success (Domestic orders)
    p_success = sorted_order_data['Success'].mean()

    # Define the number of trials as the total number of orders
    n_trials = len(sorted_order_data)

    x = np.arange(0, n_trials + 1)
    y = binom.pmf(x, n_trials, p_success)

    # Plot the binomial distribution
    plt.figure(figsize=(10, 6))
    plt.stem(x, y)
    plt.xlabel("Number of On-Time Deliveries")
    plt.ylabel("Probability")
    plt.title("Binomial Distribution for On-Time Deliveries")
    plt.savefig('static/images/ms_4_binomial_distribution.png')

# 5. Negative Binomial Distribution for Count Data with Variability
def ms_negative_binomial_distribution_analysis(max_orders=15):
    data = order_data.copy()

    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d-%b-%y')

    sorted_order_data = data.sort_values(by='Order Date')

    # Calculate mean and variance of the 'Order Value' column as an example of count data variability
    mean_orders = sorted_order_data['Order Value'].mean()
    variance_orders = sorted_order_data['Order Value'].var()

    r = mean_orders**2 / (variance_orders - mean_orders)  # Number of successes
    p = mean_orders / variance_orders  # Probability of success
    x = np.arange(0, max_orders)
    y = nbinom.pmf(x, r, p)

    # Plot the Negative Binomial distribution
    plt.figure(figsize=(10, 6))
    plt.stem(x, y)
    plt.xlabel("Number of Orders")
    plt.ylabel("Probability")
    plt.title("Negative Binomial Distribution for Weekly Orders")
    plt.savefig('static/images/ms_5_negative_binomial_distribution.png')

# 6. Gamma Distribution for Modeling Costs or Duration
def ms_gamma_distribution_analysis(time_range=(0, 15), num_points=100):
    data = route_data.copy()

    handling_time_mean = data['Port/Airport/Rail Handling time (hours)'].mean()
    handling_time_variance = data['Port/Airport/Rail Handling time (hours)'].var()

    # Calculate shape (k) and scale (θ) parameters
    shape_param = handling_time_mean**2 / handling_time_variance  # k = mean^2 / variance
    scale_param = handling_time_variance / handling_time_mean  # θ = variance / mean
    
    x = np.linspace(time_range[0], time_range[1], num_points)
    y = gamma.pdf(x, shape_param, scale=scale_param)

    # Plot the Gamma distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("Handling Time (hours)")
    plt.ylabel("Probability Density")
    plt.title("Gamma Distribution for Handling Time")
    plt.savefig('static/images/ms_6_gamma_distribution.png')


# Optimization Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Budget Transport Optimization Problem
def om_transportation_problem_with_budget(max_budget=500000000):
    o_data = order_data.copy()
    r_data = route_data.copy()


    # Ensure date columns are in datetime format
    o_data['Order Date'] = pd.to_datetime(o_data['Order Date'], format='%d-%b-%y')
    o_data['Required Delivery Date'] = pd.to_datetime(o_data['Required Delivery Date'], format='%d-%b-%y')

    # Get list of unique sources and destinations
    sources = o_data['Ship From'].unique()
    destinations = o_data['Ship To'].unique()

    # Calculate supply at each source
    supply = np.array([
        o_data[o_data['Ship From'] == src]['Weight (KG)'].sum() for src in sources
    ])

    # Calculate demand at each destination
    demand = np.array([
        o_data[o_data['Ship To'] == dst]['Weight (KG)'].sum() for dst in destinations
    ])

    # Create cost matrix
    cost_matrix = np.array([
        [
            r_data[
                (r_data['Source'] == src) & (r_data['Destination'] == dst)
            ]['Fixed Freight Cost'].mean()
            if not r_data[
                (r_data['Source'] == src) & (r_data['Destination'] == dst)
            ]['Fixed Freight Cost'].empty else np.inf
            for dst in destinations
        ]
        for src in sources
    ])

    # Replace NaN or inf costs with a high penalty
    penalty_cost = 1e6
    cost_matrix = np.nan_to_num(cost_matrix, nan=penalty_cost, posinf=penalty_cost)

    # Ensure feasibility
    total_supply = supply.sum()
    total_demand = demand.sum()

    if total_supply > total_demand:
        # Add a dummy destination
        demand = np.append(demand, total_supply - total_demand)
        cost_matrix = np.column_stack((cost_matrix, np.full(len(sources), penalty_cost)))
    elif total_demand > total_supply:
        # Add a dummy source
        supply = np.append(supply, total_demand - total_supply)
        cost_matrix = np.vstack((cost_matrix, np.full((1, len(demand)), penalty_cost)))

    # Number of sources and destinations
    num_sources = len(supply)
    num_destinations = len(demand)

    # Flatten the cost matrix
    costs = cost_matrix.flatten()

    # Build constraint matrices
    A_eq = []
    b_eq = []

    # Supply constraints
    for i in range(num_sources):
        constraint = np.zeros(num_sources * num_destinations)
        for j in range(num_destinations):
            constraint[i * num_destinations + j] = 1
        A_eq.append(constraint)
        b_eq.append(supply[i])

    # Demand constraints
    for j in range(num_destinations):
        constraint = np.zeros(num_sources * num_destinations)
        for i in range(num_sources):
            constraint[i * num_destinations + j] = 1
        A_eq.append(constraint)
        b_eq.append(demand[j])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Bounds for decision variables
    bounds = [(0, None) for _ in range(num_sources * num_destinations)]

    # Add budget constraint
    A_ub = [costs]  # Total cost constraint
    b_ub = [max_budget]

    # Solve using linprog
    result = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    # Check if the optimization succeeded
    if result.success:
        shipment_plan = result.x.reshape((num_sources, num_destinations))

        # Display results
        print("Optimal Total Cost:", result.fun)
        #print("Optimal Shipment Plan:\n", np.round(shipment_plan, 2))

        # Visualization
        def visualize_shipment_plan(shipment_plan, sources, destinations):
            plt.figure(figsize=(20, 8))
            sns.heatmap(shipment_plan, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Units Shipped"})
            plt.title(f"Optimal Shipment Plan Heatmap - Max Budget {max_budget}")
            plt.xlabel("Destinations")
            plt.ylabel("Sources")
            plt.xticks(ticks=np.arange(len(destinations)) + 0.5, labels=destinations, rotation=45)
            plt.yticks(ticks=np.arange(len(sources)) + 0.5, labels=sources, rotation=0)
            plt.tight_layout()
            plt.savefig('static/images/om_1_shipment_plan.png')

        visualize_shipment_plan(shipment_plan, sources, destinations)

        return result.fun
    else:
        print("Optimization failed:", result.message)

        # make a plot giving the error message and save it with the same name
        plt.figure(figsize=(10, 6))
        plt.title(f"Optimal Shipment Plan Heatmap - Max Budget {max_budget}")
        plt.text(0.5, 0.5, result.message, ha='center', va='center', fontsize=12)
        plt.text(0.5, 0.2,"Please Try Again with a different Maximum Budget",fontsize=12)
        plt.axis('off')
        plt.savefig('static/images/om_1_shipment_plan.png')

        return "Infeasible"
    
# 2. Optimization of Route Costs Using Gradient Descent
def om_optimize_route_cost(learning_rate=0.01, iterations=200, penalty=500, constant=2000):

    data = route_data.copy()

    # Define the cost function with penalties and fixed costs
    def cost_function_route(freight, fuel):
        weight_freight = 0.6
        weight_fuel = 0.4

        # Core cost function
        base_cost = weight_freight * freight**2 + weight_fuel * fuel**2 + freight * fuel

        # Penalty for costs outside practical bounds
        penalty_cost = penalty if (freight < 100 or freight > 5000 or fuel < 50 or fuel > 3000) else 0
        return base_cost + penalty_cost + constant

    # Initialize variables with mean values
    freight = data['Fixed Freight Cost'].mean()
    fuel = data['Bunker/ Fuel Cost'].mean()

    # Lists to store the values for plotting and table
    freight_vals, fuel_vals, total_costs = [freight], [fuel], [cost_function_route(freight, fuel)]

    # Gradient descent loop with constraints
    for i in range(iterations):
        grad_freight = 2 * 0.6 * freight + fuel  # Partial derivative wrt freight
        grad_fuel = 2 * 0.4 * fuel + freight     # Partial derivative wrt fuel

        # Update variables
        freight -= learning_rate * grad_freight
        fuel -= learning_rate * grad_fuel

        # Apply realistic constraints
        freight = max(min(freight, 5000), 100)  # Freight cost between 100 and 5000
        fuel = max(min(fuel, 3000), 50)         # Fuel cost between 50 and 3000

        # Adjust learning rate dynamically
        learning_rate *= 0.99 if i < iterations // 2 else 1.01

        # Calculate total cost
        total_cost = cost_function_route(freight, fuel)

        # Store values for visualization and table
        freight_vals.append(freight)
        fuel_vals.append(fuel)
        total_costs.append(total_cost)

    # Create a DataFrame to display the results in a table
    results_df = {
        'Optimal Freight Cost': freight,
        'Optimal Fuel Cost': fuel,
        'Minimum Total Cost': total_costs[-1]
    }

    # Display the final results
    print("Final Results:")
    print(f"Optimal Freight Cost: {freight}")
    print(f"Optimal Fuel Cost: {fuel}")
    print(f"Minimum Total Cost: {total_costs[-1]}")

    # Plotting cost values over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(total_costs)), total_costs, label="Cost", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Optimization of Cost Function")
    plt.legend()
    plt.savefig('static/images/om_2_cost_function.png')

    # Generate a contour plot to show convergence of gradient descent
    freight_range = np.linspace(min(freight_vals) - 100, max(freight_vals) + 100, 100)
    fuel_range = np.linspace(min(fuel_vals) - 50, max(fuel_vals) + 50, 100)
    F, Fu = np.meshgrid(freight_range, fuel_range)
    Z = np.vectorize(cost_function_route)(F, Fu)

    # Plot the contour plot with gradient descent path
    plt.figure(figsize=(10, 7))
    plt.contour(F, Fu, Z, levels=50, cmap="viridis")
    plt.plot(freight_vals, fuel_vals, marker='o', color='red', markersize=5, label="Gradient Descent Path")
    plt.xlabel("Freight Cost")
    plt.ylabel("Fuel Cost")
    plt.title("Contour Plot of Cost Function with Gradient Descent Convergence Path")
    plt.colorbar(label="Total Cost")
    plt.legend()
    plt.savefig('static/images/om_2_gradient_descent.png')
    
    return results_df

# 3. Balanced Knapsack Optimization for Load Distribution
def om_balanced_knapsack_optimization(vehicle_capacities):
    """
    Function to optimize load distribution across vehicles using a balanced knapsack approach.
    
    Parameters:
    - order_table (str): The name of the table in the database containing order data with weights.
    - vehicle_capacities (list of int): List of capacities for each vehicle in KG.
    
    Returns:
    - Prints optimized load distribution per vehicle.
    - Displays a bar chart showing the capacity utilization for each vehicle.
    """
    
    # Load order data from the database
    data = order_data.copy()

    # Define order weights from the database data
    order_weights = data['Weight (KG)'].to_numpy()  # Array of weights of each order
    num_orders = len(order_weights)
    num_vehicles = len(vehicle_capacities)
    total_order_weight = sum(order_weights)

    # Function to solve the knapsack problem for a single vehicle with load balancing in mind
    def knapsack_with_balance(vehicle_capacity, weights, target_load):
        n = len(weights)
        dp = np.zeros((n + 1, vehicle_capacity + 1))

        # Fill the DP table
        for i in range(1, n + 1):
            for w in range(vehicle_capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + weights[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]

        # Backtrack to find the items included in the optimal solution
        w = vehicle_capacity
        included_items = []
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                included_items.append(i - 1)  # Add the item index
                w -= weights[i - 1]

        # Calculate how close we are to the target load
        total_weight = dp[n][vehicle_capacity]
        load_balance_penalty = abs(total_weight - target_load)

        return included_items, total_weight, load_balance_penalty

    # Determine target load based on average capacity utilization
    target_load = total_order_weight / num_vehicles
    vehicle_allocations = {f"Vehicle {v+1}": [] for v in range(num_vehicles)}
    remaining_weights = list(order_weights)
    utilized_weights = []
    penalty_weights = []

    for v, vehicle_capacity in enumerate(vehicle_capacities):
        # Solve knapsack with balance for the current vehicle
        included_items, total_weight, penalty = knapsack_with_balance(vehicle_capacity, remaining_weights, target_load)
        utilized_weights.append(total_weight)
        penalty_weights.append(penalty)

        # Record allocations and remove allocated weights
        for idx in included_items:
            vehicle_allocations[f"Vehicle {v+1}"].append((f"Order {idx + 1}", remaining_weights[idx]))
            remaining_weights[idx] = 0  # Mark the order as assigned by setting its weight to 0

    # Print results
    print("Enhanced Optimization Results (Balanced Knapsack Optimization):")
    for v in range(num_vehicles):
        print(f"\nVehicle {v+1} - Total Weight Assigned: {utilized_weights[v]} KG (Penalty: {penalty_weights[v]:.2f})")
        for order, weight in vehicle_allocations[f"Vehicle {v+1}"]:
            print(f"  {order} - Weight: {weight} KG")

    # Store the printed information as an image

    # Create a blank image with white background
    img_width, img_height = 800, 100 + 150 * num_vehicles # Adjust height based on number of orders
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    # Define font and starting position
    font = ImageFont.load_default()
    x, y = 10, 10

    # Draw the text on the image
    draw.text((x, y), "Enhanced Optimization Results (Balanced Knapsack Optimization):", fill='black', font=font)
    y += 20
    for v in range(num_vehicles):
        draw.text((x, y), f"\nVehicle {v+1} - Total Weight Assigned: {utilized_weights[v]} KG (Penalty: {penalty_weights[v]:.2f})", fill='black', font=font)
        y += 40
        for order, weight in vehicle_allocations[f"Vehicle {v+1}"]:
            draw.text((x + 20, y), f"  {order} - Weight: {weight} KG", fill='black', font=font)
            y += 20

    # Crop the Excess blank space at the bottom of the image
    img = img.crop((0, 0, img_width, y+30))

    # Save the image
    img.save('static/images/om_3_description.png')

    # Visualization: Double Bar Graph for Capacity Utilization with balancing penalty
    vehicle_names = list(vehicle_allocations.keys())
    x_positions = np.arange(len(vehicle_names))  # x-axis positions

    plt.figure(figsize=(12, 6))
    bar1 = plt.bar(x_positions - 0.2, utilized_weights, width=0.4, label="Utilized Weight", color='skyblue')
    bar2 = plt.bar(x_positions + 0.2, vehicle_capacities, width=0.4, label="Total Capacity", color='orange', alpha=0.7)

    # Add text labels for the bars
    for bar in bar1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200, f'{bar.get_height():.0f}', ha='center', va='bottom')
    for bar in bar2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200, f'{bar.get_height():.0f}', ha='center', va='bottom')

    # Formatting the graph
    plt.xticks(x_positions, vehicle_names)
    plt.ylabel("Weight (KG)")
    plt.title("Vehicle Capacity Utilization (Balanced Knapsack Optimization)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/om_3_knapsack.png')

